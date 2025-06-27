from typing import Union

import numpy as np
from numpy._typing import NDArray

FloatArray = Union[np.ndarray, list[float], NDArray[np.floating]]
Shape3D = tuple[int, int, int]
BlockSize = tuple[float, float, float]
Vector = Union[tuple[float, float, float], list[float]]
Point = Union[tuple[float, float, float], list[float]]
MinMax = Union[tuple[float, float], list[float]]