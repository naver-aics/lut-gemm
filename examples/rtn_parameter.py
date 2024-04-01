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
import numpy as np

#from clet.functions.rtn import Quantizer
from utils import CompressionParameter, PACKER, Quantizer
from bcq_parameter import BCQParameter

class RTNParameter(CompressionParameter):
    def compress(self, in_ch_wise=False, **kwargs):
        data_shape = self.data.shape
        group_size = -1
        if 'group_size' in kwargs:
            group_size = kwargs.pop('group_size')
        out_ch = data_shape[0]
        in_ch = data_shape[1]

        quant = Quantizer()
        quant.configure(**kwargs)
        if in_ch_wise == False:
            data = self.data
            if group_size > 0:
                data = data.reshape([-1, group_size])
            quant.find_params(data, weight=True)
            quant_data  = torch.clamp(torch.round(data / quant.scale) + quant.zero, 0, quant.maxq)
            quant_data  = quant_data.reshape([out_ch, -1, group_size]).to(torch.int)
            quant.scale = quant.scale.reshape([out_ch, -1, 1])
            quant.zero  = quant.zero.reshape([out_ch, -1, 1])
        else:
            data = self.data.T
            if group_size > 0:
                data = data.reshape([-1, group_size])
            quant.find_params(data, weight=True)
            quant_data = torch.clamp(torch.round(data / quant.scale) + quant.zero, 0, quant.maxq)
            quant_data = quant_data.reshape([in_ch, -1, group_size]).to(torch.int)
            quant.scale = quant.scale.reshape([in_ch, -1, 1])
            quant.zero  = quant.zero.reshape([in_ch, -1, 1])

        return quant.scale, quant.zero, quant_data, quant_data.shape

    def decompress(self, scale, zero, quant_data, quant_data_shape, in_ch_wise=False):
        # w.shape = [out_ch, in_ch]
        # in_ch_wise == True
        #   -> quant_data.shape = [in_ch, out_ch//group_size, group_size]
        #   -> scale.shape      = [in_ch, out_ch//group_size, 1]
        #   -> zero.shape       = [in_ch, out_ch//group_size, 1]
        # in_ch_wise == False
        #   -> quant_data.shape = [out_ch, in_ch//group_size, group_size]
        #   -> scale.shape      = [out_ch, in_ch//group_size, 1]
        #   -> zero.shape       = [out_ch, in_ch//group_size, 1]

        if in_ch_wise == True:
            out_ch = quant_data_shape[1] * quant_data_shape[2]
            decomp_w = scale * (quant_data - zero)
            decomp_w = decomp_w.reshape([-1, out_ch]).T
        else:
            out_ch = quant_data_shape[0]
            decomp_w = scale * (quant_data - zero)
            decomp_w = decomp_w.reshape([out_ch, -1])
        self.data = decomp_w

    def convert_bcq_format(self, scale, zero, quant_data, qbits, do_packing=False, in_ch_wise=False):
        global PACKER

        zero   = scale * zero
        upack  = torch.Tensor([[2**i for i in range(qbits)]])
        scale  = scale / 2.0
        scale  = torch.matmul(scale, upack)

        offset = scale.sum(-1).unsqueeze(-1) - zero

        binary = torch.zeros(list(quant_data.shape) + [qbits])
        binary_shape = binary.shape
        for i in range(qbits):
            binary[:, :, :, i] = ((quant_data >> i) & 1) * 2.0 - 1.0

        if do_packing == True:
            binary, binary_shape = PACKER.pack(binary)
            binary = binary.to(self.data.device)

        return scale, binary, binary_shape, offset

if __name__ == '__main__':
    w_org = torch.randn(1024, 256)

    # INT4 Quantization -> RTN
    w_rtn = RTNParameter(w_org)
    scale, zero, w_quant, w_quant_shape = w_rtn.compress(in_ch_wise=False, qbits=4, group_size=128, perchannel=True, sym=False)
    w_rtn.decompress(scale, zero, w_quant, w_quant_shape, in_ch_wise=False)
    print(abs(w_org-w_rtn.data).mean())

    # Convert INT4 -> BCQ4
    alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(scale, zero, w_quant, qbits=4, do_packing=False, in_ch_wise=False)

    # BCQ Decompress Check
    w_bcq = BCQParameter(w_org)
    w_bcq.decompress(alpha, binary, binary_shape, offset=offset, do_packing=False, in_ch_wise=False)
    print(abs(w_bcq.data - w_rtn.data).mean())