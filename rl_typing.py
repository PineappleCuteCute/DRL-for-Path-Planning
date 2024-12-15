# -*- coding: utf-8 -*-
"""
Chú thích loại RL

Tạo: Ngày 12 tháng 12 năm 2024, 17:54:17
Cập nhật: Ngày 12 tháng 12 năm 2024

Tác giả: Đào Thành Mạnh
GitHub: https://github.com/PineappleCuteCute
"""
import numpy as np
import torch as th
from gym import spaces
from typing import Union, Literal, Optional


__all__ = [
    # Các loại chính thức
    "Union",
    "Optional",
    "Literal",

    # Khai báo loại
    "ListLike",
    "PathLike",
    "DeviceLike",
    "TorchLoss",
    "TorchOptimizer",
    "GymEnv",
    "GymBox",
    "GymDiscrete",
    "GymTuple",
    "GymDict",
    
    # Khai báo đầu vào và đầu ra
    "ObsSpace",
    "ActSpace",
    "Obs",
    "Act",
    "ObsBatch",
    "ActBatch",
]

#----------------------------- ↓↓↓↓↓ Khai báo loại ↓↓↓↓↓ ------------------------------#
from os import PathLike
ListLike = Union[list, np.ndarray]

DeviceLike = Union[th.device, str, None]
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import Optimizer as TorchOptimizer

from gym import Env as GymEnv
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple
from gym.spaces import Dict as GymDict

ObsSpace = spaces.Space                                             # Không gian trạng thái/quan sát: Bất kỳ
ActSpace = Union[spaces.Box, spaces.Discrete, spaces.MultiDiscrete] # Không gian hành động/kiểm soát: Box (liên tục), Discrete (mã hóa), MultiDiscrete (rời rạc)

_MetaObs = Union[int, np.ndarray]
_MixedObs = Union[dict[any, _MetaObs], tuple[_MetaObs, ...], list[_MetaObs]]
Obs = Union[_MetaObs, _MixedObs] # Trạng thái/Quan sát: int, array, hoặc kết hợp
Act = Union[int, np.ndarray]     # Hành động/Điều khiển: int là điều khiển mã hóa (DiscreteAct), array là điều khiển liên tục (BoxAct) hoặc rời rạc (MultiDiscreteAct)

_MetaObsBatch = th.FloatTensor
_MixedObsBatch = Union[dict[any, _MetaObsBatch], tuple[_MetaObsBatch, ...], list[_MetaObsBatch]]
ObsBatch = Union[_MetaObsBatch, _MixedObsBatch] # Đầu vào mạng nơ-ron: FloatTensor hoặc kết hợp của chúng
ActBatch = Union[th.FloatTensor, th.LongTensor] # Đầu ra mạng nơ-ron: FloatTensor cho điều khiển liên tục, LongTensor cho điều khiển mã hóa/rời rạc
