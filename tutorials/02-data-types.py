from typing import List
import cutlass
import cutlass.cute as cute

x = cutlass.Int32(5)
y = cutlass.Float32(3.14)

# @cute.jit
# def foo(a: cutlass.Int32):
    
@cute.jit
def bar():
    a = cutlass.Float32(3.14)
    print("a(static =)", a)

    x = cutlass.Int32(42)
    y = x.to(cutlass.Float32)
    cute.printf(">>> {}", y)

    a = cutlass.Float32(3.14)
    b = a.to(cutlass.Int32)
    cute.printf("Float32({}) => Int32({})", a, b)

    c = cutlass.Int32(127)
    d = c.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({})", c, d)

    a = cutlass.Int32(10)
    b = cutlass.Int32(3)
    # cute.printf("a: Int32({}), b: Int32({})", a, b)
    div_result = a / b
    print("result type {}", type(div_result))

    a = cutlass.Int16(10)
    b = cutlass.Int16(3)
    div_result = a / b
    print("result type {}", type(div_result))

    a = cutlass.Int8(10)
    b = cutlass.Int8(3)
    div_result = a / b
    print("result type {}", type(div_result))



    


    
def test_bar():
    bar()

test_bar()