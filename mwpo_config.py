from dataclasses import dataclass
from typing import Literal, Optional

from trl import DPOConfig


@dataclass
class MWPOConfig(DPOConfig):
    weighted: Optional[bool] = False
    weight_alpha: Optional[float] = None
    weight_beta: Optional[float] = None
    len_norm: Optional[bool] = False
    len_scale: Optional[float] = None
    len_lambda: Optional[float] = None