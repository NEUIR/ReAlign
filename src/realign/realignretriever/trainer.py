import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import ReAlignRetriever
from huggingface_hub import login
from .arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class ReAlignRetrieverTrainer(Trainer):
    def __init__(self, *args, data_args: DataArguments = None, **kwargs):
        super(ReAlignRetrieverTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.data_args = data_args

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (ReAlignRetriever,)
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if len(inputs) == 5:
            query, document, pair, query_describe, d_exist_ids = inputs
        elif len(inputs) == 3:
            query, document, pair = inputs
            query_describe = None
            d_exist_ids = None
        else:
            raise ValueError(f"Unexpected input format with {len(inputs)} elements")

        loss = model(
            query=query,
            document=document,
            pair=pair,
            query_describe=query_describe,
            d_exist_ids=d_exist_ids,
        ).loss

        return loss

    def training_step(self, *args):
        return super(ReAlignRetrieverTrainer, self).training_step(*args) / self._dist_loss_scale_factor
