# 
# An implementation of a fast GEMM kernel
# in CuTE DSL that matches Torch's GEMM
# 

import cutlass
import cutlass.cute as cute
import cutlass.cute

# A simple tiling GEMM implementation of matrix multiplication
@cute.kernel
def gemm_naive_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    sA_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    
    sA = smem.allocate_tensor(
        A.element_type, sA_layout,
        byte_alignent=16
    )

    copy_atom    = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        A.element_type, 
        num_bits_per_copy=128
    )

    tiled_copy   = cute.make_tiled_copy_tv(
        copy_atom,
        cute.make_layout((16,16)),
        cute.make_layout((8,1))
    )

    thr_copy     = tiled_copy.get_slice(tidx)
    tG, tS       = thr_copy.partition_S(A), thr_copy.partition_D(sA)

    cute.copy(tiled_copy, tG, tS)

@cute.jit
def gemm_naive(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr
):
    kernel = gemm_naive_kernel(A, B, C)