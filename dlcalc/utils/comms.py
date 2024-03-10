
from .size import Size

def get_grad_reducescatter_volume(grad_size: Size, dp_size: int):
    return grad_size * ((dp_size - 1) / dp_size)
