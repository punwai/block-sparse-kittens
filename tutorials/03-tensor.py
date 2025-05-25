
import torch 

import cutlass
import cutlass.cute as cute
from cutlass.torch import dtype as torch_dtype
import cutlass.cute.runtime as cute_rt
from cutlass.cute.runtime import from_dlpack

@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5, 1))
    tensor = cute.make_tensor(ptr, layout)
    tensor.fill(1)
    cute.print_tensor(tensor)

def test_create_tensor():
    a = cutlass.Int32(10)
    b = cutlass.Int32(3)
    cute.printf("a: Int32({}), b: Int32({})", a, b)

@cute.jit
def print_tensor_dlpack(src: cute.Tensor):
    print(src)
    cute.print_tensor(src)

def test_print_dlpack():
    a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))
    print_tensor_dlpack(from_dlpack(a))

@cute.jit
def tensor_access_item(a: cute.Tensor):
    cute.printf("a[2] = {} (equivalent to a[{}])", a[2],
                cute.make_identity_tensor(a.layout.shape)[2])

    cute.printf("a[2,0] = {}", a[2,0])

@cute.kernel
def print_tensor_gpu(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5,1))
    tensor = cute.make_tensor(ptr, layout)

    tidx, _, _ = cute.arch.thread_idx()

    if tidx == 0:
        cute.print_tensor(tensor)

def test_access_item():
    data = torch.arange(0, 8*5, dtype=torch.float32).reshape(8, 5)
    tensor_access_item(from_dlpack(data))


test_access_item()
