# LUT-GEMM
# Copyright (c) 2024-present NAVER Cloud Corp. All rights reserved.
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

import torch
import transformers
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    GPTQConfig,
    LlamaTokenizer

)

from rtn_parameter import RTNParameter
from bcq_parameter import BCQParameter


layers = ["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='facebook/opt-125m',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--qbits",
        type=int,
        default=4,
        help="Quantization Bits.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="quantization grouping size for weights",
    )
    args = parser.parse_args()

    return args

#def quant_model(model, module_to_not_convert:str = "lm_head"):
def quant_model(model, args):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            quant_model(module, args)

        if any(x in name for x in layers):
            print(name)
            original_weight = module.weight.clone().detach()
            # INT4 Quantization -> RTN
            w_rtn = RTNParameter(original_weight)
            scale, zero, w_quant, w_quant_shape = w_rtn.compress(
                in_ch_wise=False, qbits=args.qbits, group_size=args.group_size,
                perchannel=True, sym=False)

            # Convert INT4 -> BCQ4
            alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(
                scale, zero, w_quant, qbits=args.qbits,
                do_packing=False, in_ch_wise=False)

            print("Parameter size before packing")
            print("  alpha.size()  =", alpha.size())
            print("  binary.size() =", binary.size())
            print("="*30)

            # Packing BCQ4 -> Packed Weight (uint8)
            alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(
                scale, zero, w_quant, qbits=args.qbits,
                do_packing=True, in_ch_wise=False)

            print("Parameter size after packing")
            print("  alpha.size()  =", alpha.size())
            print("  binary.size() =", binary.size())
            print("="*30)

    return model

def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    model = quant_model(model, args)

if __name__ == "__main__":
    main()
