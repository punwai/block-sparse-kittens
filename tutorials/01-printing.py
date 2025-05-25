import cutlass
import cutlass.cute as cute

@cute.jit
def print_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    print(">>>", b)
    print(">>>", a)

    cute.printf(">?? a: {}", a)
    cute.printf(">?? b: {}", b)

    print(">>>", type(a))
    print(">>>", type(b))

    layout = cute.make_layout((a, b))
    print(">>>", layout)
    cute.printf(">?? {}", layout)

print_example(cutlass.Int32(8), 2)

print_example_compiled = cute.compile(print_example, cutlass.Int32(8), 2)

@cute.jit
def format_string_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    print(f"a: {a}, b: {b}")

    layout = cute.make_layout((a, b))
    print(f"layout: {layout}")

print("Direct run output:")
format_string_example(cutlass.Int32(8), 2)
