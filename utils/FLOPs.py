from numbers import Number
from typing import Any, Dict, List, Optional
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import Handle, elementwise_flop_counter
import torch
from math import prod
from utils import args


def get_shape(val: Any) -> Optional[List[int]]:
    """
    Get the shapes from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None


# def repeat_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
#     """
#     Count flops for fully connected layers.
#     """
#     # Count flop for nn.Linear
#     # inputs is a list of length 2.
#     input_shapes = [get_shape(v) for v in inputs]
#     assert len(input_shapes) == 2 and (input_shapes[0] is not None) and (input_shapes[1] is not None), input_shapes
#     if len(input_shapes[0]) == 0 or len(input_shapes[1]) == 0:
#         return 0
#     flops = min(prod(input_shapes[0]), prod(input_shapes[1]))
#     # batch_size, input_dim = input_shapes[0]
#     # output_dim = input_shapes[1][1]
#     # flops = batch_size * input_dim * output_dim
#     return flops


_ADDITIONAL_OPS: Dict[str, Handle] = {
    "aten::mul": None,
    "aten::mul_": None,
    "aten::tanh": elementwise_flop_counter(1, 0), 
    "aten::acos": elementwise_flop_counter(1, 0),
    "aten::repeat": elementwise_flop_counter(1, 0),
    "aten::cos": elementwise_flop_counter(1, 0),
    "aten::add": None,
    "aten::mean": elementwise_flop_counter(1, 0),
    "aten::gelu": None
}

from fvcore.nn import FlopCountAnalysis
def FLOP_counter(model, input_shape: tuple) -> int:
    counter = FlopCountAnalysis(model, torch.randn(size=input_shape).cuda(args.local_rank)).set_op_handle(**_ADDITIONAL_OPS)
    return counter.total()  