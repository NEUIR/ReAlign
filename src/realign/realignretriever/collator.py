import torch
import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, ProcessorMixin
from realign.realignretriever.arguments import DataArguments
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    def __post_init__(self):
        self.is_qwen = 'qwen' in type(self.processor).__name__.lower()

    def _encode_images(self, images):
        if self.is_qwen:
            messages_list = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What is shown in this image?"}
                ]
            }] for img in images]
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                     for msg in messages_list]
            batch_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            input_ids_list = [batch_inputs['input_ids'][i].tolist() for i in range(len(images))]
            return input_ids_list, batch_inputs
        else:
            collated_list = [self.processor("<|image_1|>\nWhat is shown in this image?", img, return_tensors="pt")
                             for img in images]
            input_ids_list = [d['input_ids'][0].tolist() for d in collated_list]
            return input_ids_list, collated_list

    def _attach_image_tensors(self, target_dict, image_meta):
        if self.is_qwen:
            target_dict['pixel_values'] = image_meta['pixel_values']
            if 'image_grid_thw' in image_meta:
                target_dict['image_grid_thw'] = image_meta['image_grid_thw']
            elif 'image_sizes' in image_meta:
                target_dict['image_sizes'] = image_meta['image_sizes']
        else:
            target_dict['pixel_values'] = torch.stack([d['pixel_values'][0] for d in image_meta], dim=0)
            target_dict['image_sizes'] = torch.stack([d['image_sizes'][0] for d in image_meta], dim=0)

    def build_image_attention_mask(self, seq_len, input_lengths):
        image_attention_masks = []
        for input_len in input_lengths:
            image_attention_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
            image_attention_mask[input_len:, :input_len-1] = 0 
            image_attention_masks.append(image_attention_mask.unsqueeze(0))
        image_attention_masks = torch.cat(image_attention_masks, dim=0)
        return image_attention_masks

    def __call__(self, features: List[Tuple[str, List[str]]]):
        if len(features[0]) == 2:
            all_queries = [f[0] for f in features]
            all_images = [f[-1] for f in features]
            query_types = ['text'] * len(features)
            describes = [''] * len(features)
        else:
            all_queries = [f[0] for f in features]
            all_images = [f[1] for f in features]
            query_types = [f[2] for f in features]
            describes = [f[3] if len(f) > 3 else '' for f in features]

        query_type_ids = torch.tensor(
            [1 if qt == 'image' else 0 for qt in query_types],
            dtype=torch.long
        )

        q_collated = self.tokenizer(
            all_queries,
            padding=False, 
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        d_collated = {}
        d_input_ids, d_image_meta = self._encode_images(all_images)
        d_collated['input_ids'] = d_input_ids

        if self.data_args.append_eos_token:
            q_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_collated['input_ids']]
            d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]

        if self.data_args.pretrain:
            p_collated = {}
            all_input_ids, all_label_ids, input_lengths = [], [], []

            for i, ocr in enumerate(all_queries):
                prompt_input_ids = torch.tensor(d_collated['input_ids'][i]).unsqueeze(0)
                answer = f'{ocr}<|im_end|>' if self.is_qwen else f'{ocr}<|end|>\n<|endoftext|>'
                answer_input_ids = self.tokenizer(
                    answer, add_special_tokens=False, max_length=self.data_args.answer_max_len, truncation=True, return_tensors='pt')['input_ids']
                input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
                labels = torch.cat(
                    [
                        torch.tensor([-100] * len(prompt_input_ids[0])).unsqueeze(0),
                        answer_input_ids,
                    ],
                    dim=1,
                )
                all_input_ids.append(input_ids.squeeze(0).unsqueeze(1))
                all_label_ids.append(labels.squeeze(0).unsqueeze(1))
                input_lengths.append(len(prompt_input_ids[0]))

            input_ids = torch._C._nn.pad_sequence(
                all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).squeeze(2)
            labels = torch._C._nn.pad_sequence(
                all_label_ids, batch_first=True, padding_value=-100
            ).squeeze(2)

            p_collated['input_ids'] = input_ids
            p_collated['labels'] = labels

            if self.data_args.image_attention_mask:
                image_attention_mask = self.build_image_attention_mask(input_ids.size()[1], input_lengths)
                p_collated['attention_mask'] = image_attention_mask.unsqueeze(1)
        else:
            p_collated = None

        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )

        self._attach_image_tensors(d_collated, d_image_meta)
        if self.data_args.pretrain:
            p_collated['pixel_values'] = d_collated['pixel_values']
            if 'image_grid_thw' in d_collated:
                p_collated['image_grid_thw'] = d_collated['image_grid_thw']
            if 'image_sizes' in d_collated:
                p_collated['image_sizes'] = d_collated['image_sizes']

        all_describes_text = []
        for i in range(len(features)):
            if query_types[i] == 'image' and describes[i] and describes[i].strip():
                all_describes_text.append(describes[i])
            else:
                all_describes_text.append("N/A")

        q_describe_collated = self.tokenizer(
            all_describes_text,
            padding=False,
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.append_eos_token:
            q_describe_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_describe_collated['input_ids']]

        q_describe_collated = self.tokenizer.pad(
            q_describe_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return q_collated, d_collated, p_collated, q_describe_collated, query_type_ids

@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    def __post_init__(self):
        self.is_qwen = 'qwen' in type(self.processor).__name__.lower()

    def _encode_images(self, images):
        if self.is_qwen:
            messages_list = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What is shown in this image?"}
                ]
            }] for img in images]
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                     for msg in messages_list]
            batch_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            input_ids_list = [batch_inputs['input_ids'][i].tolist() for i in range(len(images))]
            return input_ids_list, batch_inputs
        else:
            collated_list = [self.processor("<|image_1|>\nWhat is shown in this image?", img, return_tensors="pt")
                             for img in images]
            input_ids_list = [d['input_ids'][0].tolist() for d in collated_list]
            return input_ids_list, collated_list

    def __call__(self, features: List[Tuple[str, str]]):
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[-1] for x in features]

        if self.data_args.encode_is_query:
            collated = self.tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
            image_meta = None
        else:
            collated = {}
            input_ids_list, image_meta = self._encode_images(images)
            collated['input_ids'] = input_ids_list

        if self.data_args.append_eos_token:
            collated['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated['input_ids']]

        collated = self.tokenizer.pad(
            collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if not self.data_args.encode_is_query and image_meta is not None:
            if self.is_qwen:
                collated['pixel_values'] = image_meta['pixel_values']
                if 'image_grid_thw' in image_meta:
                    collated['image_grid_thw'] = image_meta['image_grid_thw']
                elif 'image_sizes' in image_meta:
                    collated['image_sizes'] = image_meta['image_sizes']
            else:
                collated['pixel_values'] = torch.stack([d['pixel_values'][0] for d in image_meta], dim=0)
                collated['image_sizes'] = torch.stack([d['image_sizes'][0] for d in image_meta], dim=0)

        return text_ids, collated
