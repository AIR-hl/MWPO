from dataclasses import dataclass
from typing import Literal, Optional

from trl import DPOConfig


@dataclass
class MWPOConfig(DPOConfig):
    len_lambda: Optional[float] = None
    weighted: Optional[bool] = False
    weight_alpha: Optional[float] = None
    weight_beta: Optional[float] = None
    len_norm: Optional[bool] = False
    len_scale: Optional[float] = None