# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trainer base class for RL training."""

from __future__ import annotations

import os
os.environ["WANDB_MODE"]="online"


import abc
import argparse
import copy
import itertools
from typing import Any, ClassVar

import deepspeed
import optree
import torch
import torch.distributed as dist
import torch.nn as nn
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler,
)
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.configs import ADAM_BETAS
from safe_rlhf.datasets import DummyDataset, PromptOnlyBatch, PromptOnlyDataset, SupervisedDataset
from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.trainers.base import TrainerBase
from safe_rlhf.utils import (
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    is_main_process,
    is_same_tokenizer,
    masked_mean,
    to_device,
)


class EDTrainer(TrainerBase):  # pylint: disable=too-many-instance-attributes
    """Trainer base class for RL training.

    Abstract methods:
        rollout: Rollout a batch of experiences.
        rl_step: Perform a single update step with RL loss.
        eval_step: Perform a single evaluation step.
    """

    TRAINING_TYPE: ClassVar[str] = 'rl'

    actor_model: deepspeed.DeepSpeedEngine
    actor_reference_model: deepspeed.DeepSpeedEngine
    reward_model: deepspeed.DeepSpeedEngine

    reward_tokenizer: PreTrainedTokenizerBase

    ds_train_config: dict[str, Any]
    ds_eval_config: dict[str, Any]

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        self.global_step = 0

        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

        self.reference_generation_config = GenerationConfig(
            max_length=self.args.max_length,
            num_return_sequences=self.args.num_return_sequences,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            renormalize_logits=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            # pad_token_id=self.tokenizer.unk_token_id,
        )

        self.actor_generation_config = GenerationConfig(
            max_length=self.args.max_length,
            num_return_sequences=self.args.rl_num_return_sequences,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            renormalize_logits=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            # pad_token_id=self.tokenizer.unk_token_id,
        )

        # Those value can be changed
        self.kl_coeff = self.args.kl_coeff
        self.clip_range_ratio = self.args.clip_range_ratio
        self.clip_range_score = self.args.clip_range_score
        self.clip_range_value = self.args.clip_range_value
        self.gamma = 1.0
        self.gae_lambda = 0.95

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)

        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)

        self.actor_model, self.tokenizer = load_pretrained_models(
            self.args.actor_model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
            use_peft=True,
        )

        self.actor_reference_model, _ = load_pretrained_models(
            self.args.actor_model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        self.reward_model, self.reward_tokenizer = load_pretrained_models(
            self.args.reward_model_name_or_path,
            model_max_length=self.args.max_length,
            auto_model_type=AutoModelForScore,
            padding_side='right',
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs={
                'score_type': 'reward',
                'do_normalize': self.args.normalize_reward,
            },
        )
        self.reward_model.set_normalize(self.args.normalize_reward)

        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        if (
            self.args.per_device_prompt_batch_size
            * self.args.num_return_sequences
            % self.args.per_device_train_batch_size
            != 0
        ):
            raise ValueError(
                'The number of prompt-only samples must be divisible by the micro batch size.',
            )

        prompt_only_dataset = PromptOnlyDataset(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                prompt_only_dataset, eval_dataset = prompt_only_dataset.split_train_test(
                    split_ratio=self.args.eval_split_ratio,
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = PromptOnlyDataset(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None

        self.sampler= DistributedSampler(prompt_only_dataset, shuffle=True)
        self.prompt_only_dataloader = DataLoader(
            prompt_only_dataset,
            collate_fn=prompt_only_dataset.get_collator(),
            sampler=self.sampler,
            batch_size=self.args.per_device_prompt_batch_size,
        )

        self.args.total_training_steps = int(
            len(self.prompt_only_dataloader)
            * self.args.epochs
            * self.args.update_iters
            * self.args.per_device_prompt_batch_size
            * self.args.rl_num_return_sequences
            // self.args.per_device_train_batch_size,
        )

    def _init_train_engine(
        self,
        model: nn.Module,
        weight_decay: float,
        lr: float,
        lr_scheduler_type: str,
        lr_warmup_ratio: float,
        total_training_steps: int,
        ds_config: dict[str, Any],
    ) -> deepspeed.DeepSpeedEngine:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay)
        if (
            ds_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=lr, betas=ADAM_BETAS)
        else:
            optimizer = FusedAdam(optimizer_grouped_parameters, lr=lr, betas=ADAM_BETAS)

        lr_scheduler_update_steps = total_training_steps // ds_config['gradient_accumulation_steps']
        num_warmup_steps = int(lr_scheduler_update_steps * lr_warmup_ratio)
        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=lr_scheduler_update_steps,
        )
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
        )
        return engine

    def _init_eval_engine(
        self,
        model: nn.Module,
        ds_config: dict[str, Any],
    ) -> deepspeed.DeepSpeedEngine:
        engine, *_ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        return engine

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        actor_ds_config = copy.deepcopy(self.ds_train_config)
        actor_total_training_steps = self.args.total_training_steps
        self.actor_model = self._init_train_engine(
            model=self.actor_model,
            weight_decay=self.args.actor_weight_decay,
            lr=self.args.actor_lr,
            lr_scheduler_type=self.args.actor_lr_scheduler_type,
            lr_warmup_ratio=self.args.actor_lr_warmup_ratio,
            total_training_steps=actor_total_training_steps,
            ds_config=actor_ds_config,
        )

        self.actor_reference_model = self._init_eval_engine(
            model=self.actor_reference_model,
            ds_config=self.ds_eval_config,
        )
        self.actor_reference_model.eval()

        self.reward_model = self._init_eval_engine(
            model=self.reward_model,
            ds_config=self.ds_eval_config,
        )
        self.reward_model.eval()

        if self.args.actor_gradient_checkpointing:
            self.actor_model.module.gradient_checkpointing_enable()

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        if mode:
            self.actor_model.train()

            if self.args.actor_gradient_checkpointing:
                self.actor_model.module.gradient_checkpointing_enable()
        else:
            self.actor_model.eval()

            if self.args.actor_gradient_checkpointing:
                self.actor_model.module.gradient_checkpointing_disable()

    def split_lambda_micro_batches(
        self,
        prompt_only_batch: PromptOnlyBatch,
    ) -> list[PromptOnlyBatch]:
        """Split a batch of lambda samples into micro-batches."""
        total_batch_size = prompt_only_batch['input_ids'].size(0)
        micro_batches = []
        for i in range(0, total_batch_size):
            micro_batch = optree.tree_map(
                # pylint: disable-next=cell-var-from-loop
                lambda tensor: tensor[i : i + 1],  # noqa: B023
                prompt_only_batch,
            )
            batch = self.rollout_reference(micro_batch)
            micro_batches.append(batch)
        return micro_batches
    
    def split_rl_micro_batches(
        self,
        prompt_only_batch: PromptOnlyBatch,
    ) -> list[PromptOnlyBatch]:
        """Split a batch of RL samples into micro-batches."""
        total_batch_size = prompt_only_batch['input_ids'].size(0)
        micro_batch_size = self.args.per_device_train_batch_size
        micro_batches = []
        for i in range(0, total_batch_size, micro_batch_size):
            micro_batch = optree.tree_map(
                # pylint: disable-next=cell-var-from-loop
                lambda tensor: tensor[i : i + micro_batch_size],  # noqa: B023
                prompt_only_batch,
            )
            micro_batches.extend(self.rollout(micro_batch))
        return micro_batches

    @torch.no_grad()
    def rollout(self, prompt_only_batch: PromptOnlyBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch['input_ids']
        # print(input_ids)
        sequences = self.actor_model.module.generate(
            input_ids=input_ids,
            attention_mask=prompt_only_batch['attention_mask'],
            generation_config=self.actor_generation_config,
            synced_gpus=True,
            do_sample=True,
        )
        sequences = (
            sequences.contiguous()
            .view(
                input_ids.size(0),
                self.args.rl_num_return_sequences,
                -1,
            )
            .transpose(0, 1)
        )

        return [
            self.post_rollout(
                input_ids,
                seq,
                attention_mask=torch.logical_and(
                    seq.not_equal(self.tokenizer.pad_token_id),
                    seq.not_equal(self.tokenizer.unk_token_id),
                ),
                probs=True,
            )
            for seq in sequences
        ]

    @torch.no_grad()
    def rollout_reference(self, prompt_only_batch: PromptOnlyBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences using reference model."""
        input_ids = prompt_only_batch['input_ids']
        sequences = self.actor_reference_model.module.generate(
            input_ids=input_ids,
            attention_mask=prompt_only_batch['attention_mask'],
            generation_config=self.reference_generation_config,
            synced_gpus=True,
            do_sample=True,
        )
        sequences = (
            sequences.contiguous()
            .view(
                input_ids.size(0),
                self.args.num_return_sequences,
                -1,
            )
            .transpose(0, 1)
        )

        return [
            self.post_rollout(
                input_ids,
                seq,
                attention_mask=torch.logical_and(
                    seq.not_equal(self.tokenizer.pad_token_id),
                    seq.not_equal(self.tokenizer.unk_token_id),
                ),
                probs=False,
            )
            for seq in sequences
        ]

    @abc.abstractmethod
    # @torch.no_grad()
    def post_rollout(
        self,
        prompt: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        """Post-process a rollout sample."""
        raise NotImplementedError

    @abc.abstractmethod
    def rl_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single update step with RL loss."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def lambda_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single update step with Lambda loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def eval_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        """Perform a single evaluation step."""
        raise NotImplementedError

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.args.total_training_steps,
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        if self.args.need_eval:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)

        # for prompt_only_batch in self.prompt_only_dataloader:
        #     prompt_only_batch = to_device(prompt_only_batch, self.args.device)

        #     # Step 2
        #     lambda_batches = self.split_lambda_micro_batches(prompt_only_batch)
        #     torch.cuda.empty_cache()
        #     for lambda_batch in lambda_batches:
        #         lambda_info = self.lambda_step(lambda_batch)
        #         self.logger.log(lambda_info, step=self.global_step)
        #         self.global_step += 1
        #         torch.cuda.empty_cache()
        #         print("Lambda value: ", lambda_info["train/lambda"])

        # print("*** lambda step done ***")
        # print("Lambda value: ", lambda_info["train/lambda"])

        for epoch in range(self.args.epochs):
            self.sampler.set_epoch(epoch)
            for prompt_only_batch in self.prompt_only_dataloader:
                prompt_only_batch = to_device(prompt_only_batch, self.args.device)

                # Step 3
                # generate batches
                rl_batches = self.split_rl_micro_batches(prompt_only_batch)

                self.set_train()
                for rl_batch in rl_batches:
                    rl_info = self.rl_step(rl_batch)
                    # torch.cuda.empty_cache()
                    self.logger.log(rl_info, step=self.global_step)

                    progress_bar.set_description(
                        f'Training {epoch + 1}/{self.args.epochs} epoch '
                        f'(reward {rl_info["train/reward"]:.4f})',
                    )
                    progress_bar.update(1)
                    self.global_step += 1

                    if self.global_step % self.args.save_interval == 0:
                        self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                        if is_main_process():
                            self.actor_model.save_checkpoint(
                                self.args.output_dir,
                                tag=self.global_step,
                            )

                        peft_dir = os.path.join(self.args.output_dir, "peft", str(self.global_step))
                        os.makedirs(peft_dir, exist_ok=True)
                        if is_main_process():
                            model = self.actor_model.module.merge_and_unload() 
                            model.save_pretrained(peft_dir, safe_serialization=True,)  

                        self.logger.print('Checkpoint saved.')

                    if (
                        self.args.need_eval
                        and self.args.eval_strategy == 'steps'
                        and self.global_step % self.args.eval_interval == 0
                    ):
                        self.logger.print(
                            f'\n***** Evaluating at step {self.global_step} *****',
                        )
                        self.logger.log(self.eval(), step=self.global_step)

            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}

        self.set_eval()
        prompts: list[str] = []
        generateds: list[str] = []
        scores: dict[str, list[torch.Tensor]] = {}

        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
        )

        for batch in eval_dataloader:
            batch = to_device(batch, self.args.device)
            with torch.no_grad():
                seq = self.actor_model.module.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.args.max_length,
                    synced_gpus=True,
                    do_sample=True,
                )

            dist.barrier()

            attention_mask = torch.logical_and(
                seq.not_equal(self.tokenizer.pad_token_id),
                seq.not_equal(self.tokenizer.unk_token_id),
            )
            for key, values in self.eval_step(seq, attention_mask).items():
                if key not in scores:
                    scores[key] = []
                scores[key].append(values)
            prompt = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            generated = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
            generated = [text[len(prompt[i]) :] for i, text in enumerate(generated)]
            prompts.extend(prompt)
            generateds.extend(generated)

        # Display result in main process
        if is_main_process():
            columns = ['Prompt', 'Generated', *list(scores.keys())]
            concatenated_scores = {
                key: torch.cat(value, dim=0).to(torch.float32) for key, value in scores.items()
            }
            concatenated_scores = {
                key: value.tolist() for key, value in concatenated_scores.items()
            }
            rows = list(zip(prompts, generateds, *concatenated_scores.values()))
            self.logger.print_table(
                title='Evaluating...',
                columns=columns,
                rows=rows,
                max_num_rows=5,
            )

        # Gather results from all processes
        for key, values in scores.items():
            scores[key] = torch.cat(values, dim=0).mean()
            dist.reduce(scores[key], dst=0, op=dist.ReduceOp.AVG)
            scores[key] = scores[key].mean().item()
        dist.barrier()

        self.set_train()

        return scores

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | PreTrainedModel | None = None,
        ds_config: dict | None = None,
    ) -> None:
        """Save model and tokenizer."""
        if model is None:
            model = self.actor_model
        if ds_config is None:
            ds_config = self.ds_train_config
        super().save(model=model, ds_config=ds_config)
