import torch
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bdim, _, _ = cute.arch.block_dim()
    bidx, _, _ = cute.arch.block_idx()

    thread_idx = bdim * bidx + tidx

    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    gC[mi, ni] = a_val + b_val

@cute.jit
def naive_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    num_threads_per_block = 256
    
    m,n = mA.shape
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1),
        block=(num_threads_per_block, 1, 1)
    )

def benchmark(callable, *, num_warmups, num_iterations, tensor_size):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    for _ in range(num_warmups):
        callable()

    start_event.record(stream=torch.cuda.current_stream())

    for _ in range(num_iterations):
        callable()
    end_event.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iterations

    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Throughput: {(3 * tensor_size * 2) / (avg_time / 1000) / 1e9:.2f} Gb/s")


def test_naive_element_wise_add():
    shape = (2048, 2048)
    a = torch.randn(*shape, dtype=torch.float32, device="cuda")
    b = torch.randn(*shape, dtype=torch.float32, device="cuda")
    c = torch.zeros(*shape, dtype=torch.float32, device="cuda")
    


    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)     
    c_ = from_dlpack(c, assumed_align=16)

    naive_elementwise_add_ = cute.compile(
        naive_elementwise_add, 
        a_, b_, c_)
    naive_elementwise_add_(a_, b_, c_)

    benchmark(partial(naive_elementwise_add_, a_, b_, c_), tensor_size=a.numel(), num_warmups=5, num_iterations=100)

    torch.testing.assert_close(c, a + b)

test_naive_element_wise_add()

@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    m, n = gA.shape[1]
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()
    print(f"[DSL INFO] sliced gA = {gA[(None, (mi, ni))]}")
    print(f"[DSL INFO] sliced gB = {gB[(None, (mi, ni))]}")

    gC[(None, (mi, ni))] = a_val + b_val

@cute.jit
def vectorized_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    threads_per_block = 256
    
    gA = cute.zipped_divide(mA, (4,16))
    gB = cute.zipped_divide(mB, (4,16))
    gC = cute.zipped_divide(mC, (4,16))

    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")

    print(cute.size(gC, mode=[0]))

    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1)
    )

def test_vectorized_element_add():
    shape  = (2048, 2048)
    a = torch.randn(*shape, device="cuda", dtype=torch.float16)
    b = torch.randn(*shape, device="cuda", dtype=torch.float16)
    c = torch.randn(*shape, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled_func = cute.compile(vectorized_elementwise_add, a_, b_, c_)
    compiled_func(a_, b_, c_)

    torch.testing.assert_close(c, a+b)
    benchmark(partial(compiled_func, a_, b_, c_), tensor_size=a.numel(), num_warmups=5, num_iterations=100)

test_vectorized_element_add()