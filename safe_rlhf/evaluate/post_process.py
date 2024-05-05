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

import argparse
import csv
import os

import numpy as np
import torch


# def parse_arguments() -> argparse.Namespace:
#     """Parse the command-line arguments."""
#     parser = argparse.ArgumentParser(
#         prog='deepspeed --module safe_rlhf.evaluate.score',
#         description='Score the performance of a model.',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     # Model
#     model_parser = parser.add_argument_group('model')
#     model_parser.add_argument(
#         '--model_name_or_path',
#         type=str,
#         help='the name or path of the model to load from',
#         required=True,
#     )
#     model_parser.add_argument(
#         '--reward_model_name_or_path',
#         type=str,
#         help='the name or path of the reward model to load from',
#     )
#     model_parser.add_argument(
#         '--max_length',
#         type=int,
#         default=128,
#         help='The maximum sequence length of the model.',
#     )
#     model_parser.add_argument(
#         '--trust_remote_code',
#         type=str2bool,
#         default=False,
#         help='Whether to trust the remote code.',
#     )

#     # Evaluation
#     evaluation_parser = parser.add_argument_group('evaluation')
#     evaluation_parser.add_argument(
#         '--per_device_eval_batch_size',
#         type=int,
#         default=8,
#         help='Batch size (per device) for the evaluation dataloader.',
#     )
#     evaluation_parser.add_argument(
#         '--total_generate_size',
#         type=int,
#         default=3000,
#         help='The total amount of prompts to be completed.',
#     )
#     evaluation_parser.add_argument(
#         '--seed',
#         type=int,
#         default=42,
#         help='A seed for reproducible evaluation.',
#     )
#     evaluation_parser.add_argument(
#         '--fp8',
#         type=str2bool,
#         default=False,
#         help='Whether to use float8 precision.',
#     )
#     evaluation_parser.add_argument(
#         '--fp16',
#         type=str2bool,
#         default=False,
#         help='Whether to use float16 precision.',
#     )
#     evaluation_parser.add_argument(
#         '--bf16',
#         type=str2bool,
#         default=False,
#         help='Whether to use bfloat16 precision.',
#     )
#     evaluation_parser.add_argument(
#         '--tf32',
#         type=str2bool,
#         default=None,
#         help='Whether to use tf32 mix precision.',
#     )

#     # Logging
#     logging_parser = parser.add_argument_group('logging')
#     logging_parser.add_argument(
#         '--output_dir',
#         type=str,
#         default='output/score',
#         help='Where to store the evaluation output.',
#     )
#     logging_parser.add_argument(
#         '--input_dir',
#         type=str,
#         default='output/generate',
#         help='Where to store the evaluation output.',
#     )

#     # DeepSpeed
#     deepspeed_parser = parser.add_argument_group('deepspeed')
#     deepspeed_parser.add_argument(
#         '--local_rank',
#         type=int,
#         default=-1,
#         help='Local rank for distributed training on GPUs',
#     )
#     deepspeed_parser.add_argument(
#         '--zero_stage',
#         type=int,
#         default=0,
#         choices=[0, 1, 2, 3],
#         help='ZeRO optimization stage for models.',
#     )
#     deepspeed_parser.add_argument(
#         '--offload',
#         type=str,
#         default='none',
#         choices=['none', 'parameter', 'optimizer', 'all'],
#         help='Offload parameters and/or optimizer states to CPU.',
#     )

#     args = parser.parse_args()

#     return args

def main() -> None:  # pylint: disable=too-many-locals,too-many-statements

    model_name = 'beaver-7b-v1.0'
    total_generate_size = 3000
    reward_model_name = 'beaver-7b-v1.0-cost'
    file_path = 'output/score'+f'/{model_name}_{total_generate_size}_{reward_model_name}_scores.pt' # for test purpuse

    # scores = torch.load(file_path)
    scores = torch.load(file_path)
    print(scores[:,-1].mean())
    breakpoint()
    

if __name__ == '__main__':
    main()
