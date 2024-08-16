# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from model_basedon_huggingface.norms.mask_layernorm import LayerNorm
from model_basedon_huggingface.norms.mask_batchnorm import MaskBatchNorm
from model_basedon_huggingface.norms.mask_groupnorm import GroupNorm
from model_basedon_huggingface.norms.mask_syncbatchnorm import MaskSyncBatchNorm
from model_basedon_huggingface.norms.mask_powernorm import MaskPowerNorm

def NormSelect(norm_type, embed_dim, head_num=None, warmup_updates=1000):
    if norm_type == "layer":
        return LayerNorm(embed_dim)
    elif norm_type == "batch":
        # return MaskBatchNorm(embed_dim)
        return MaskSyncBatchNorm(embed_dim)
    elif norm_type == 'power':
        return MaskPowerNorm(embed_dim, group_num=head_num, warmup_iters=warmup_updates)
    
