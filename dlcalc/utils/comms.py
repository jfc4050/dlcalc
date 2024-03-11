
from .size import Size

def get_grad_reducescatter_volume(grad_size: Size, dp_size: int):
    n = grad_size
    p = dp_size
    return n * ((p - 1) / p)
