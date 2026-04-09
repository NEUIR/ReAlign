from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F

from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from realign.realignretriever.arguments import ModelArguments, ReAlignRetrieverTrainingArguments as TrainingArguments, DataArguments

import logging
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class ReAlignRetriever(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 kl_loss_weight: float = 1.0,
                 is_qwen: bool = False,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.kl_loss_weight = kl_loss_weight
        self.is_qwen = is_qwen
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, 
                document: Dict[str, Tensor] = None, 
                pair: Dict[str, Tensor] = None, 
                query_describe: Dict[str, Tensor] = None,
                d_exist_ids: Tensor = None,
                use_cache: bool = True
        ):
        q_reps = self.encode_query(query, use_cache=use_cache) if query else None
        p_reps = self.encode_document(document, use_cache=use_cache) if document else None
        outputs = self.generate_output(pair, use_cache=use_cache) if pair else None

        q_describe_reps = self.encode_query(query_describe, use_cache=use_cache) if query_describe is not None else None

        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if q_describe_reps is not None:
                    q_describe_reps = self._dist_gather_tensor(q_describe_reps)
                if d_exist_ids is not None:
                    d_exist_ids = self._dist_gather_tensor(d_exist_ids)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss = self.compute_loss(scores / self.temperature, target)

            if d_exist_ids is not None:
                image_mask = (d_exist_ids == 1)
            else:
                image_mask = torch.zeros(q_reps.size(0), dtype=torch.bool, device=q_reps.device)

            num_image = image_mask.sum().item()
            if num_image > 0 and q_describe_reps is not None:
                image_q = q_reps[image_mask]
                image_p = p_reps[image_mask]
                image_describe = q_describe_reps[image_mask]
                image_scores = self.compute_similarity(image_q, image_p)
                describe_scores = self.compute_similarity(image_describe, image_p)
                kl_loss = self.compute_kl_loss(
                    image_scores / self.temperature,
                    describe_scores / self.temperature
                )
                loss = loss + self.kl_loss_weight * kl_loss

            if outputs:
                loss = loss + outputs.loss

            if self.is_ddp:
                loss = loss * self.world_size

            scores = self.compute_similarity(q_reps, p_reps)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def compute_kl_loss(self, teacher_scores, student_scores):
        teacher_probs = F.softmax(teacher_scores, dim=-1)
        student_log_probs = F.log_softmax(student_scores, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl_loss
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @staticmethod
    def _is_qwen(model_name_or_path: str) -> bool:
        return 'qwen' in model_name_or_path.lower()

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            data_args: DataArguments = None,
            **hf_kwargs,
    ):
        is_qwen = cls._is_qwen(model_args.model_name_or_path)
        hf_kwargs['trust_remote_code'] = True

        if is_qwen:
            transformer_cls = AutoModel
            logger.info(f"Loading Qwen model from {model_args.model_name_or_path}")
        else:
            transformer_cls = AutoModelForCausalLM

        base_model = transformer_cls.from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = (
                base_model.config.eos_token_id if is_qwen else 0
            )

        logger.info(f"Model config - hidden_size: {base_model.config.hidden_size}, "
                     f"pad_token_id: {base_model.config.pad_token_id}")

        kl_loss_weight = data_args.kl_loss_weight if data_args else 1.0

        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                logger.info(f"Loading LoRA weights from {model_args.lora_name_or_path}")
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                logger.info(f"Creating new LoRA config with r={model_args.lora_r}, "
                            f"alpha={model_args.lora_alpha}")
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False,
                )
                lora_model = get_peft_model(base_model, lora_config)
                lora_model.print_trainable_parameters()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_loss_weight=kl_loss_weight,
                is_qwen=is_qwen,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_loss_weight=kl_loss_weight,
                is_qwen=is_qwen,
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        is_qwen = cls._is_qwen(model_name_or_path)
        hf_kwargs['trust_remote_code'] = True
        transformer_cls = AutoModel if is_qwen else AutoModelForCausalLM
        base_model = transformer_cls.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = (
                base_model.config.eos_token_id if is_qwen else 0
            )
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize,
                is_qwen=is_qwen,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize,
                is_qwen=is_qwen,
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def encode_query(self, qry, use_cache=True):
        query_hidden_states = self.encoder(
            **qry,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=use_cache,
        )
        query_hidden_states = query_hidden_states.hidden_states[-1]
        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_document(self, doc, use_cache=True):
        return self.encode_query(doc, use_cache=use_cache)

    def generate_output(self, pair, use_cache=True):
        return self.encoder(**pair, use_cache=use_cache)

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
