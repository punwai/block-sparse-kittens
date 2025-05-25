
import cutlass               
import cutlass.cute as cute  
import numpy as np

@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    
    if tidx == 0:
        cute.printf("Hello, world!")
    

@cute.jit
def hello_world():
    cute.printf("hello world")

    cutlass.cuda.initialize_cuda_context()

    kernel().launch(
        grid=(1, 1, 1),
        block=(32, 1, 1)
    )

print("Running hello_world()...")
hello_world()

print("Running compiled version...")
hello_world_compiled = cute.compile(hello_world)
hello_world_compiled()