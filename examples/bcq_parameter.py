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
import torch.nn as nn

from bcq import quantize
from utils import CompressionParameter, PACKER

class BCQParameter(CompressionParameter):
    def compress(self, do_packing=False, in_ch_wise=False, **kwargs):
        global PACKER
        _, binary, alpha, _ = quantize(self.data, transpose=in_ch_wise, **kwargs)

        binary_shape = binary.shape
        if do_packing == True:
            binary, binary_shape = PACKER.pack(binary)
            binary = binary.to(self.data.device)

        return alpha, binary, binary_shape

    def decompress(self, alpha, binary, binary_shape, offset=None, do_packing=False, in_ch_wise=False):
        global PACKER

        if do_packing == True:
            binary = PACKER.unpack(binary, binary_shape, dtype=self.data.dtype)
            binary = binary.to(self.data.device)

        # w.shape = [out_ch, in_ch]
        # in_ch_wise == True
        #   -> binary.shape = [in_ch, out_ch//group_size, group_size, qbits]
        #   -> alpha.shape  = [in_ch, out_ch//group_size, qbits]
        #   -> offset.shape = [in_ch, out_ch//group_size, 1]
        # in_ch_wise == False
        #   -> binary.shape = [out_ch, in_ch//group_size, group_size, qbits]
        #   -> alpha.shape  = [out_ch, in_ch//group_size, qbits]
        #   -> offset.shape = [out_ch, in_ch//group_size, 1]

        if in_ch_wise == True:
            out_ch = binary_shape[1] * binary_shape[2]
            decomp_w = torch.einsum('iogb,iob->iog', (binary, alpha))
            if offset is not None:
                decomp_w = decomp_w + offset
            decomp_w = decomp_w.reshape([-1, out_ch]).T
        else:
            out_ch = binary_shape[0]
            decomp_w = torch.einsum('oigb,oib->oig', (binary, alpha))
            if offset is not None:
                decomp_w = decomp_w + offset
            decomp_w = decomp_w.reshape([out_ch, -1])
        self.data = decomp_w

class BCQTunedParameter:
    def __init__(self, alpha, binary, binary_shape, do_packing=False, in_ch_wise=False):
        #self.alpha = nn.Parameter(
        pass

if __name__ == '__main__':
    w_org = torch.randn(1024, 256)

    w_bcq = BCQParameter(w_org)
    alpha, binary, binary_shape = w_bcq.compress(do_packing=False, in_ch_wise=True, qbits=4, rounds=15, group_size=128)
    w_bcq.decompress(alpha, binary, binary_shape, do_packing=False, in_ch_wise=True)
    print(abs(w_org-w_bcq.data).mean())

    w_bcq = BCQParameter(w_org)
    alpha, binary, binary_shape = w_bcq.compress(do_packing=False, in_ch_wise=False, qbits=4, rounds=15, group_size=128)
    w_bcq.decompress(alpha, binary, binary_shape, do_packing=False, in_ch_wise=False)
    print(abs(w_org-w_bcq.data).mean())
