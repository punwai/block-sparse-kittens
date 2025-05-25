import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np

@cute.jit
def print_tensor_basic(x: cute.Tensor):
    print("Basic output:")
    cute.print_tensor(x)

@cute.jit
def print_tensor_verbose(x: cute.Tensor):
    print("Verbose output:")
    cute.print_tensor(x, verbose=True)

@cute.jit
def print_tensor_slice(x: cute.Tensor, coord : tuple):
    sliced_data = cute.slice_(x, coord)
    y = cute.make_fragment(sliced_data.layout, sliced_data.element_type)
    y.store(sliced_data.load())
    print("Slice output:")
    cute.print_tensor(y)


def tensor_print_example1():
    shape = (4,3,2)
    data = np.arange(24, dtype=np.float32).reshape(*shape)
    print_tensor_basic(from_dlpack(data))

def tensor_print_example2():
    shape = (4,3,2)
    data = np.arange(24, dtype=np.float32).reshape(*shape)
    print_tensor_verbose(from_dlpack(data))

def tensor_print_exmaple3():
    shape = (4, 3)
    data = np.arange(12, dtype=np.float32).reshape(*shape)

    print_tensor_slice(from_dlpack(data), (None, 0))
    print_tensor_slice(from_dlpack(data), (1, None))

tensor_print_exmaple3()
