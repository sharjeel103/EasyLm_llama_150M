import pprint
from functools import partial

import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import optax
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer, GenerationConfig, FlaxLogitsProcessorList
)

from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.serving import LMServer
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfigurator, FlaxLLaMAForCausalLM
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    param_dtype='bf16',
    dtype='bf16',
    input_length=1024,
    seq_length=2048,
    top_k=50,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    add_bos_token=True,
    load_checkpoint='',
    tokenizer='sharjeel103/16ktokenizer',
    llama=LLaMAConfigurator.get_default_config(),
    lm_server=LMServer.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


from collections import namedtuple
import sentencepiece as spm
import numpy as np

# Define a named tuple for tokenizer outputs
TokenizerOutput = namedtuple('TokenizerOutput', ['input_ids', 'attention_mask'])

class CustomSPMTokenizer:
    def __init__(self, model_path, truncation_side='right', padding_side='right'):
        # Load SentencePiece model
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_path)
        
        # Set up tokenizer properties
        self.truncation_side = truncation_side
        self.padding_side = padding_side
        self.pad_token_id = self.tokenizer.eos_id()  # Usually pad token same as EOS
        self.bos_token_id = self.tokenizer.bos_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.unk_token_id = self.tokenizer.unk_id()

    def encode(self, text, max_length=None, padding='max_length'):
        tokens = self.tokenizer.encode(text)

        # Truncate tokens if needed
        if max_length is not None:
            if self.truncation_side == 'left' and len(tokens) > max_length:
                tokens = tokens[-max_length:]
            elif self.truncation_side == 'right' and len(tokens) > max_length:
                tokens = tokens[:max_length]
        
        # Pad tokens if needed
        if padding == 'max_length' and max_length is not None:
            tokens = self._pad_sequence(tokens, max_length, self.padding_side)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
        if max_length is not None and len(tokens) < max_length:
            attention_mask += [0] * (max_length - len(tokens))

        # Return as a named tuple
        return TokenizerOutput(
            input_ids=np.array(tokens, dtype=np.int32),
            attention_mask=np.array(attention_mask, dtype=np.int32)
        )

    def _pad_sequence(self, tokens, max_length, padding_side='right'):
        if len(tokens) < max_length:
            pad_length = max_length - len(tokens)
            if padding_side == 'left':
                tokens = [self.pad_token_id] * pad_length + tokens
            elif padding_side == 'right':
                tokens += [self.pad_token_id] * pad_length
        return tokens

    def batch_encode(self, texts, max_length=None, padding='max_length'):
        if padding == 'longest':
            max_length = max(len(self.tokenizer.encode(text)) for text in texts)
        
        # Encode each text in the batch
        encoded_texts = [self.encode(text, max_length, padding=padding) for text in texts]
        
        # Stack all input IDs and attention masks into arrays
        return TokenizerOutput(
            input_ids=np.stack([e.input_ids for e in encoded_texts]),
            attention_mask=np.stack([e.attention_mask for e in encoded_texts])
        )
    def decode(self, tokens):
        # Decode tokens to string
        if isinstance(tokens, np.ndarray):  # Check if it's a NumPy array
            # Check if it's a 1D array (single sequence) and decode
            if tokens.ndim == 1:
                return self.tokenizer.decode(tokens.tolist())
            # Handle other dimensions (batch of sequences)
            return [self.tokenizer.decode(t.tolist()) for t in tokens]
        elif isinstance(tokens, (list, tuple)):  # Handle lists or tuples
            return self.tokenizer.decode(tokens)
        elif isinstance(tokens, (int, np.integer)):  # Handle single integer token
            return self.tokenizer.decode([tokens])
        else:
            raise ValueError(f"Unsupported token type: {type(tokens)}")
    
    def batch_decode(self, batch_tokens):
        # Decode batch of tokens
        if isinstance(batch_tokens, np.ndarray):  # Check if it's a NumPy array
            # If it's a 2D array, decode each row
            if batch_tokens.ndim == 2:
                return [self.decode(tokens) for tokens in batch_tokens]
            else:
                raise ValueError("Unsupported batch_tokens shape for batch decoding.")
        elif isinstance(batch_tokens, (list, tuple)):  # Handle lists or tuples
            return [self.decode(tokens) for tokens in batch_tokens]
        else:
            raise ValueError(f"Unsupported batch_tokens type: {type(batch_tokens)}")
    

    def __call__(self, text, padding='max_length', truncation=True, max_length=None, return_tensors=None):
        if isinstance(text, list):  # Handle batch input
            encoded = self.batch_encode(text, max_length=max_length, padding=padding)
        else:  # Handle single input
            encoded = self.encode(text, max_length, padding=padding)
        
        if return_tensors == 'np':
            # Return as NumPy arrays (already the case in this implementation)
            return encoded
        
        # Return as named tuples (default)
        return encoded


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)
    
    model_path = hf_hub_download(repo_id=FLAGS.tokenizer, filename="tok16k.model")
    
    prefix_tokenizer = CustomSPMTokenizer(
        model_path, truncation_side='left', padding_side='left'
    )
    tokenizer = CustomSPMTokenizer(
        model_path, truncation_side='right', padding_side='right'
    )
    
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)

    with jax.default_device(jax.devices("cpu")[0]):
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, disallow_trainstate=True
        )

        hf_model = FlaxLLaMAForCausalLM(
            llama_config,
            input_shape=(1, FLAGS.seq_length),
            seed=FLAGS.seed,
            _do_init=False,
            dtype=get_float_dtype_by_name(FLAGS.dtype),
            param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
        )

    model_ps = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), params
    )
    shard_fns, _ = make_shard_and_gather_fns(
        model_ps, get_float_dtype_by_name(FLAGS.param_dtype)
    )

    @partial(
        pjit,
        in_shardings=(model_ps, PS(), PS()),
        out_shardings=(PS(), PS(), PS())
    )
    def forward_loglikelihood(params, rng, batch):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        input_tokens = batch['input_tokens']
        output_tokens = batch['output_tokens']
        input_mask = batch['input_mask']
        output_mask = batch['output_mask']

        logits = hf_model.module.apply(
            params, input_tokens, attention_mask=input_mask,
            deterministic=True, rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        loglikelihood = -optax.softmax_cross_entropy_with_integer_labels(
            logits, output_tokens
        )
        loglikelihood = jnp.sum(loglikelihood * output_mask, axis=-1)
        match_count = jnp.sum(
            (jnp.argmax(logits, axis=-1) == output_tokens) * output_mask,
            axis=-1
        )
        total = jnp.sum(output_mask, axis=-1)
        is_greedy = match_count == total
        return loglikelihood, is_greedy, rng_generator()


    @partial(
        pjit,
        in_shardings=(model_ps, PS(), PS(), PS()),
        out_shardings=(PS(), PS())
    )
    def forward_generate(params, rng, batch, temperature):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            logits_processor=FlaxLogitsProcessorList(
                [FlaxTemperatureLogitsWarper(temperature)]
            ),
            generation_config=GenerationConfig(
                max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=FLAGS.do_sample,
                num_beams=FLAGS.num_beams,
                top_k=FLAGS.top_k,
                top_p=FLAGS.top_p,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    @partial(
        pjit,
        in_shardings=(model_ps, PS(), PS()),
        out_shardings=(PS(), PS())
    )
    def forward_greedy_generate(params, rng, batch):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            generation_config=GenerationConfig(
                max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        params = tree_apply(shard_fns, params)
        sharded_rng = next_rng()

    class ModelServer(LMServer):

       
        @staticmethod
        def generate(text, temperature):
            nonlocal sharded_rng
            # Tokenize input text
            inputs = prefix_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.input_length,
                return_tensors='np',
            )
            input_tokens = inputs.input_ids
            input_mask = inputs.attention_mask
        
            # Add BOS token if required
            if FLAGS.add_bos_token:
                input_tokens[:, 0] = tokenizer.bos_token_id
                input_mask[:, 0] = 1
        
            # Create input batch
            batch = dict(
                input_tokens=input_tokens,
                attention_mask=input_mask,
            )
        
            # Perform generation using mesh
            with mesh:
                output, sharded_rng = forward_generate(
                    params, sharded_rng, batch, temperature
                )
                output = jax.device_get(output)  # Retrieve output from JAX
        
                # Process output tokens
                output_text = []
                for tokens in output:
                    # Find EOS token index using NumPy
                    eos_indices = np.where(tokens == tokenizer.eos_token_id)[0]
                    if len(eos_indices) > 0:
                        eos_index = eos_indices[0]
                        tokens = tokens[:eos_index]  # Truncate at EOS
        
                    # Decode tokens to text
                    text = tokenizer.decode(tokens)
                    output_text.append(text)
        
            return output_text


       
if __name__ == "__main__":
    mlxu.run(main)
