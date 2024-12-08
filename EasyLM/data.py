import time
from functools import partial
import json
import base64
from multiprocessing import Pool

import mlxu
import numpy as np
from datasets import load_dataset


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()
        return mlxu.update_config_dict(config, updates)

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == 'huggingface':
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.fields_from_example = ''
        config.fields = 'text'
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.base64_token_dtype = 'i4'
        # config.regularize = True  # Default value for regularize (True)
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        conversation = example['conversation']  # a list of dictionaries
        token_buffer = []
        loss_mask_buffer = []

        # Add BOS token at the very start (as per instructions)
        # The instructions show a BOS at the start of the entire sequence.
        token_buffer.append(self.tokenizer.bos_id())
        loss_mask_buffer.append(0.0)

        # The pattern for each turn:
        # - Even index: user turn
        #   <|start_header_id|><|user|><|end_header_id|> + user_text + <|eot_id|>
        #   mask = 0.0 for all these tokens
        #
        # - Odd index: assistant turn
        #   <|start_header_id|><|assistant|><|end_header_id|> + assistant_text + <|eot_id|>
        #   mask = 0.0 for all headers and 1.0 for assistant_text and its closing <|eot_id|>

        start_header_id = self.tokenizer.piece_to_id('<|start_header_id|>')
        end_header_id = self.tokenizer.piece_to_id('<|end_header_id|>')
        # For user and assistant "headers", we assume they are single pieces known to tokenizer
        user_id = self.tokenizer.piece_to_id('<|user|>')
        assistant_id = self.tokenizer.piece_to_id('<|assistant|>')
        eot_id = self.tokenizer.eos_id()  # <|eot_id|>

        # Iterate through conversation turns
        for i, turn in enumerate(conversation):
            text = list(turn.values())[0]  # Extract the text from the dictionary regardless of key
            # Determine if user or assistant turn by index
            if i % 2 == 0:
                # User turn
                # Add: <|start_header_id|><|user|><|end_header_id|>
                token_buffer.append(start_header_id)
                loss_mask_buffer.append(0.0)
                token_buffer.append(user_id)
                loss_mask_buffer.append(0.0)
                token_buffer.append(end_header_id)
                loss_mask_buffer.append(0.0)

                # Encode user text
                user_tokens = self.tokenizer.encode(text, out_type=int)
                token_buffer.extend(user_tokens)
                loss_mask_buffer.extend([0.0]*len(user_tokens))

                # Add <|eot_id|>
                token_buffer.append(eot_id)
                loss_mask_buffer.append(0.0)

            else:
                # Assistant turn
                # Add: <|start_header_id|><|assistant|><|end_header_id|>
                token_buffer.append(start_header_id)
                loss_mask_buffer.append(0.0)
                token_buffer.append(assistant_id)
                loss_mask_buffer.append(0.0)
                token_buffer.append(end_header_id)
                loss_mask_buffer.append(0.0)

                # Encode assistant text
                assistant_tokens = self.tokenizer.encode(text, out_type=int)
                token_buffer.extend(assistant_tokens)
                # Assistant tokens have mask = 1.0
                loss_mask_buffer.extend([1.0]*len(assistant_tokens))

                # Add <|eot_id|> after assistant text with mask=1.0
                token_buffer.append(eot_id)
                loss_mask_buffer.append(1.0)


        return token_buffer, loss_mask_buffer, *aux


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = 'sharjeel103/Urdu_multi_turn_dataset'
        config.name = ''
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.batch_token_dtype = 'i4'
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )
        self.step_counter = 0
        self.total_tokens = 0

    def __iter__(self):
        total_tokens = 0
        while True:  # Infinite loop, managed by the external training loop
            batch_input = []
            batch_target = []
            batch_loss_mask = []

            for index, example in enumerate(self._dataset):
                tokens, loss_masks, *aux = self._text_processor(example)
                # Update total tokens
                self.total_tokens += len(tokens)

                # Truncate if longer than seq_length
                if len(tokens) > self.config.seq_length:
                    tokens = tokens[:self.config.seq_length]
                    loss_masks = loss_masks[:self.config.seq_length]

                # Pad if shorter than seq_length
                if len(tokens) < self.config.seq_length:
                    pad_length = self.config.seq_length - len(tokens)
                    tokens.extend([self._tokenizer.eos_id()] * pad_length)
                    loss_masks.extend([0.0] * pad_length)

                # Now form input_tokens, target_tokens, and loss_masks
                input_tokens = np.array(tokens, dtype=self.config.batch_token_dtype)
                target_tokens = np.zeros_like(input_tokens)
                target_loss_masks = np.zeros_like(loss_masks, dtype=np.float32)

                # Shift by one for target_tokens and target_loss_masks
                if self.config.seq_length > 1:
                    target_tokens[:-1] = input_tokens[1:]
                    target_loss_masks[:-1] = loss_masks[1:]

                # If always_start_with_bos is True, overwrite the first token with bos_id
                if self.config.always_start_with_bos:
                    input_tokens[0] = self._tokenizer.bos_id()

                # Append to the batch
                batch_input.append(input_tokens)
                batch_target.append(target_tokens)
                batch_loss_mask.append(target_loss_masks)

                # Once we have a full batch, yield it
                if len(batch_input) == self.config.batch_size:
                    metrics = {
                        'dataset_example_index': index,
                        'total_tokens': self.total_tokens,  # Include total_tokens in metrics
                    }
                    batch = {
                        'input_tokens': np.stack(batch_input, axis=0),
                        'target_tokens': np.stack(batch_target, axis=0),
                        'loss_masks': np.stack(batch_loss_mask, axis=0),
                    }
                    yield batch, metrics

                    # Reset batch buffers
                    batch_input = []
                    batch_target = []
                    batch_loss_mask = []

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(mlxu.ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:   # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_id()
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(mlxu.ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)
