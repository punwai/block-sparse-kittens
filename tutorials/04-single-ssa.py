import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack

@cute.jit
def apply_slice(src: cute.Tensor, dst: cute.Tensor, indices: cutlass.Constexpr):
    src_vec = src.load()
    dst_vec = src_vec[indices]
    print(f"{src_vec} -> {dst_vec}")

    if isinstance(dst_vec, cute.TensorSSA):
        dst.store(dst_vec)
        cute.print_tensor(dst)
    else: 
        dst[0] = dst_vec
        cute.print_tensor(dst)

def slice_1():
    src_shape = (4, 2, 3)
    dst_shape = (4, 3)
    indices = (None, 1, None)
    
    a = np.arange(np.prod(src_shape))\
        .reshape(*src_shape)\
        .astype(np.float32)
    
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)

def slice_2():
    src_shape = 4,2,3
    dst_shape = (1,)
    indices = 10
    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)

slice_2()