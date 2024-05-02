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

from __future__ import annotations

import argparse
from collections import deque
from typing import Any, List

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.trainers import EDTrainer
from safe_rlhf.utils import (
    batch_retokenize,
    gather_log_probabilities,
    get_all_reduce_max,
    get_all_reduce_mean,
    is_main_process,
    is_same_tokenizer,
    masked_mean,
)


class CPGTrainer(EDTrainer):
    TRAINING_TYPE = 'cpg'

    cost_model: deepspeed.DeepSpeedEngine

    cost_tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        super().__init__(args=args, ds_train_config=ds_train_config, ds_eval_config=ds_eval_config)

        # Lagrange multiplier
        self.lamb = torch.nn.Parameter(
            torch.tensor([0.0], device=self.args.device),
            requires_grad=True,
        )
        self.lambda_optimizer = torch.optim.Adam([self.lamb], lr=self.args.lambda_lr)
        self.episode_costs = deque(maxlen=self.args.episode_cost_window_size)

    def init_models(self) -> None:
        super().init_models()
        self.cost_model, self.cost_tokenizer = load_pretrained_models(
            self.args.cost_model_name_or_path,
            model_max_length=self.args.max_length,
            auto_model_type=AutoModelForScore,
            padding_side='right',
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs={
                'score_type': 'cost',
                'do_normalize': self.args.normalize_cost,
            },
        )
        self.cost_model.set_normalize(self.args.normalize_cost)

        if is_same_tokenizer(self.tokenizer, self.cost_tokenizer):
            self.cost_tokenizer = self.tokenizer

    def init_engines(self) -> None:
        super().init_engines()

        self.cost_model = self._init_eval_engine(
            model=self.cost_model,
            ds_config=self.ds_eval_config,
        )
        self.cost_model.eval()

    @torch.no_grad()
    def post_rollout(
        self,
        prompt: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.BoolTensor,
        probs: bool = False,
    ) -> dict[str, Any]:

        valids = torch.any(attention_mask[:, 1:], dim=1)
        sequence = sequence[valids]
        attention_mask = attention_mask[valids]

        if sequence.size(0) == 0:
            return {
                'prompt': prompt,
                'log_probs': None,
                'ref_log_probs': None,
                'reward': torch.tensor([]),
                'cost': torch.tensor([]),
                'input_ids': torch.tensor([]),
                'attention_mask': torch.tensor([]),
                'kl_divergence': None,
            }

        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            reward_seq = reward_tokenize_output['input_ids']
            reward_attention_mask = reward_tokenize_output['attention_mask']
        else:
            reward_seq = sequence
            reward_attention_mask = attention_mask

        if self.cost_tokenizer is not self.tokenizer:
            cost_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.cost_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            cost_seq = cost_tokenize_output['input_ids']
            cost_attention_mask = cost_tokenize_output['attention_mask']
        else:
            cost_seq = sequence
            cost_attention_mask = attention_mask        
        
        reward = self.reward_model(reward_seq, attention_mask=reward_attention_mask).end_scores
        cost = self.cost_model(cost_seq, attention_mask=cost_attention_mask).end_scores

        reward = reward.squeeze(dim=-1)
        cost = cost.squeeze(dim=-1)

        if probs:
            logits = self.actor_model(sequence, attention_mask=attention_mask).logits
            ref_logits = self.actor_reference_model(sequence, attention_mask=attention_mask).logits
            log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
            ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], sequence[:, 1:])
            kl_divergence = torch.sum(log_probs - ref_log_probs, dim=-1)

        self.episode_costs.extend(cost.tolist())
        return {
            'prompt': prompt,
            'log_probs': log_probs if probs else None,
            'ref_log_probs': ref_log_probs if probs else None,
            'reward': reward,
            'cost': cost,
            'input_ids': sequence,
            'attention_mask': attention_mask,
            'kl_divergence': kl_divergence if probs else None,
        }

    @torch.no_grad()
    def eval_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                input_ids,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            reward_input_ids = reward_tokenize_output['input_ids']
            reward_attention_mask = reward_tokenize_output['attention_mask']
        else:
            reward_input_ids = input_ids
            reward_attention_mask = attention_mask

        if self.cost_tokenizer is not self.tokenizer:
            cost_tokenize_output = batch_retokenize(
                input_ids,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.cost_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            cost_input_ids = cost_tokenize_output['input_ids']
            cost_attention_mask = cost_tokenize_output['attention_mask']
        else:
            cost_input_ids = input_ids
            cost_attention_mask = attention_mask

        reward = self.reward_model(
            reward_input_ids,
            attention_mask=reward_attention_mask,
        ).end_scores.squeeze(dim=-1)
        cost = self.cost_model(
            cost_input_ids,
            attention_mask=cost_attention_mask,
        ).end_scores.squeeze(dim=-1)
        return {
            'eval/reward': reward,
            'eval/cost': cost,
        }

    def actor_loss_fn(
        self,
        log_probs: torch.Tensor,  # size = (B, L - S)
        reward: torch.Tensor,  # size = (B, )
        cost: torch.Tensor,  # size = (B, )
        kl_divergence: torch.Tensor,  # size = (B, )
        mask: torch.BoolTensor,  # size = (B, L - S)
    ) -> torch.Tensor:  # size = ()
        objective = reward + cost * self.lamb - self.kl_coeff * kl_divergence
        loss = objective * torch.sum(log_probs * mask, dim=-1)
        return -loss.mean()

    def lambda_step(self, lambda_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        reward = torch.vstack([item['reward'] for item in lambda_batch])
        cost = -torch.vstack([item['cost'] for item in lambda_batch])
        objectives = (reward.squeeze(dim=-1) + cost @ self.lamb) / self.kl_coeff
        objectives = F.softmax(objectives, dim=0)
        gradient = objectives @ cost
        self.lambda_optimizer.zero_grad()
        self.lamb.grad = gradient
        self.lambda_optimizer.step()
        self.lamb = torch.max(self.lamb, torch.tensor([0.0], device=self.args.device))
        dist.barrier()
        return {
            'train/lambda': self.lamb.item(),
        }

    # pylint: disable-next=too-many-locals
    def rl_step(self, rl_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        episode_cost = torch.tensor(self.episode_costs).mean().to(self.args.device)
        prompt = rl_batch['prompt']
        reward = rl_batch['reward']
        cost = rl_batch['cost']
        attention_mask = rl_batch['attention_mask']
        input_ids = rl_batch['input_ids']
        kl_divergence = rl_batch['kl_divergence']
        start = prompt.size(-1) - 1
        sequence_mask = attention_mask[:, 1:]
        logits = self.actor_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

        actor_loss = self.actor_loss_fn(
            log_probs[:, start:],
            reward,
            cost,
            kl_divergence,
            sequence_mask[:, start:],
        )
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        dist.barrier()
        with torch.no_grad():
            mask = sequence_mask[:, start:]
            # kl_divergence = masked_mean(kl_divergence, mask)
            mean_generated_length = mask.sum(dim=-1).float().mean()
            max_generated_length = mask.sum(dim=-1).float().max()

            reward = reward.mean()
            cost = cost.mean()

            actor_loss = get_all_reduce_mean(actor_loss)
            reward = get_all_reduce_mean(reward)
            cost = get_all_reduce_mean(cost)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)

        return {
            'train/actor_loss': actor_loss.item(),
            'train/lambda': self.lamb.item(),
            'train/episode_cost': episode_cost.item(),
            'train/reward': reward.item(),
            'train/cost': cost.item(),
            # 'train/kl_divergence': kl_divergence.mean().item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
        }
