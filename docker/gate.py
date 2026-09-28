import torch
import triton
import triton.language as tl

assert torch.xpu.is_available(), "no XPU device"
dev = "xpu"

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 4096
x = torch.arange(n, dtype=torch.float32, device=dev)
y = torch.full((n,), 100.0, dtype=torch.float32, device=dev)
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.xpu.synchronize()
expected = x + y
ok = torch.equal(out, expected)
print("GATE_KERNEL_COMPILED_AND_RAN", "PASS" if ok else "FAIL")
print("sample:", out[:4].tolist(), "expected:", expected[:4].tolist())
assert ok, "vector-add result mismatch"
print("TRITON_XPU_GATE=PASS")
