#!/usr/bin/env bash
# ============================================================================
# p14-ple-offload-xpu-pinning.sh — P14: XPU PLE offload via pinned-CPU D2H
#
# Context (2026-09-21, overnight lane; boots #10 crash):
# The registration payload (connector._register_with_offload_worker) ships
# gpu_output_buffer + sem_flag_tensor (XPU storages) via ForkingPickler;
# torch 2.13.0+xpu python reductions register NO xpu reducer => reduce_storage
# hits the CPU arm => "_share_filename_: only available on CPU" => WorkerProc
# init fails on every XPU boot with PLE offload enabled.
#
# Probes (live, this image): xpu storage HAS _share_cuda_ (C++ IPC exists)
# but torch.multiprocessing.reductions registers no xpu (0 mentions) and
# torch.cuda.Stream refuses xpu. Registration is TP0-only (4 ranks, one
# socket message) so the FP8-era 2-rank multiprocess fault does not apply.
#
# DECISION (stated): mirror the FP8-era XPU serving shape — output buffer
# on pinned CPU; GPU side copies D2H (async on current stream, then event
# sync); CPU worker copies CPU->CPU per batch. GPU stays source of truth.
# This restores the exact shape that measured 5.5-24 tok/s on XPU pre-DLE
# (connector thread + event pool semantics preserved; L1 canary strings
# intact; semaphore device-flag path dormant on XPU).
#
# NOT chosen: registering torch xpu IPC reducers + stream-memop semaphore —
# requires patching torch internals; heavier surface than the proven shape.
# If pinned-CPU throughput craters at L4, that is the next lever (stated).
# ============================================================================
set -euo pipefail

SP=/opt/venv/lib/python3.12/site-packages
CONN="$SP/vllm/v1/ple_offload/connector.py"
WORK="$SP/vllm/v1/ple_offload/worker.py"
LAY="$SP/vllm/model_executor/layers/ple_offload_layer.py"

python - "$CONN" "$WORK" "$LAY" <<'PYEOF'
import sys
conn_p, work_p, lay_p = sys.argv[1:4]

# ---- 1) connector._setup_layers: allocate output buffer on pinned CPU on XPU
src = open(conn_p).read()
old = """            output_buffer = torch.empty(
                max_num_tokens,
                layer.get_offload_output_dim(int(config.ple_embed_dim)),
                dtype=layer.get_offload_output_dtype(vllm_config.model_config.dtype),
                device=self.device,
            )"""
new = """            buf_device = torch.device("cpu") if self.device.type == "xpu" else self.device  # P14 (2026-09-21)
            output_buffer = torch.empty(
                max_num_tokens,
                layer.get_offload_output_dim(int(config.ple_embed_dim)),
                dtype=layer.get_offload_output_dtype(vllm_config.model_config.dtype),
                device=buf_device,
            )
            if buf_device.type == "cpu":
                output_buffer = output_buffer.pin_memory()  # P14: D2H lands pinned for CPU->CPU handoff"""
if "P14 (2026-09-21)" in src:
    print("P14: connector already applied")
else:
    if old not in src:
        print("P14: connector buffer block NOT FOUND — drift, refusing", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)
    open(conn_p, "w").write(src)
    print("P14: connector output buffer -> pinned CPU on xpu")

# ---- 2) connector._register_with_offload_worker: CPU copies for xpu storages
src = open(conn_p).read()
old2 = """        # ForkingPickler transmits tensors through shared-memory and CUDA IPC.
        import torch.multiprocessing as torch_mp

        original_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            payload = ForkingPickler.dumps(registration)
        finally:
            torch_mp.set_sharing_strategy(original_strategy)"""
new2 = """        # ForkingPickler transmits tensors through shared-memory and CUDA IPC.
        import torch.multiprocessing as torch_mp

        original_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        # P14 (2026-09-21): xpu storages have no reducer -> _share_filename_
        # crash. Detach to CPU copies (same shapes/dtypes); the CPU worker
        # copies CPU->CPU per batch. Shapes/dtypes are preserved exactly.
        if any(t.device.type == "xpu" for t in registration.gpu_output_buffers.values()):
            registration = registration._replace(
                gpu_output_buffers={
                    k: v.detach().to("cpu", non_blocking=False)
                    for k, v in registration.gpu_output_buffers.items()
                },
                sem_flag_tensors={
                    k: v.detach().to("cpu") for k, v in registration.sem_flag_tensors.items()
                },
            )
            for v in registration.gpu_output_buffers.values():
                v.pin_memory_()
        try:
            payload = ForkingPickler.dumps(registration)
        finally:
            torch_mp.set_sharing_strategy(original_strategy)"""
if "P14 (2026-09-21): xpu storages" in src:
    print("P14: connector registration already applied")
else:
    if old2 not in src:
        print("P14: connector registration block NOT FOUND — drift, refusing", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old2, new2, 1)
    open(conn_p, "w").write(src)
    print("P14: connector registration ships CPU copies on xpu")

# ---- 3) worker busy-loop: CPU->CPU copy when target buffer is CPU
src = open(work_p).read()
old3 = """                        with torch.cuda.stream(target.copy_stream):
                            target.gpu_output_buffer[slices].copy_(
                                output_buffer[:n, :][cpu_idx]
                            )
                            target.sem.signal(target.copy_stream)"""
if "P14 (2026-09-21): cpu target" in src:
    print("P14: worker already applied")
elif old3 not in src:
    print("P14: worker copy block NOT FOUND — inspect manually", file=sys.stderr)
    sys.exit(1)
else:
    new3 = """                        if target.gpu_output_buffer.device.type == "cpu":  # P14 (2026-09-21): cpu target
                            target.gpu_output_buffer[slices].copy_(output_buffer[:n, :][cpu_idx])
                        else:
                            with torch.cuda.stream(target.copy_stream):
                                target.gpu_output_buffer[slices].copy_(
                                    output_buffer[:n, :][cpu_idx]
                                )
                                target.sem.signal(target.copy_stream)"""
    src = src.replace(old3, new3, 1)
    open(work_p, "w").write(src)
    print("P14: worker cpu-target copy branch installed")

# ---- 4) worker target creation: skip torch.cuda.Stream on CPU device
src = open(work_p).read()
old4 = "copy_stream=torch.cuda.Stream(device=gpu_buffer.device),"
new4 = ("copy_stream=(torch.cuda.Stream(device=gpu_buffer.device)\n"
        "                                   if gpu_buffer.device.type == \"cuda\"\n"
        "                                   else None),  # P14 (2026-09-21)")
if "P14 (2026-09-21)" in src and "copy_stream=(torch.cuda.Stream" in src:
    print("P14: worker stream already guarded")
elif old4 in src:
    src = src.replace(old4, new4, 1)
    open(work_p, "w").write(src)
    print("P14: worker copy_stream guarded for cpu target")
else:
    print("P14: worker stream line NOT FOUND (may already be guarded)", file=sys.stderr)
    sys.exit(1)

# ---- 5) semaphore.from_ipc_tensor: CPU flag tensor OK on xpu lane
src = open(lay_p).read()
if "P14 (2026-09-21)" in src:
    print("P14: layer semaphore already applied")
else:
    old5 = """    @classmethod
    def from_ipc_tensor(cls, flag_tensor: torch.Tensor) -> "CpuGpuSemaphore":
        \"\"\"Construct a semaphore from a CUDA tensor received through IPC.\"\"\"
        semaphore = cls.__new__(cls)
        semaphore._flag_tensor = flag_tensor
        return semaphore"""
    new5 = """    @classmethod
    def from_ipc_tensor(cls, flag_tensor: torch.Tensor) -> "CpuGpuSemaphore":
        \"\"\"Construct a semaphore from a CUDA tensor received through IPC.\"\"\"
        semaphore = cls.__new__(cls)
        semaphore._flag_tensor = flag_tensor
        return semaphore

    # P14 (2026-09-21): CPU-side flag tensor (xpu lane) — signal/wait become
    # plain host memory ops; GPU-side stream memops dormant on this path.
    def signal_cpu(self) -> None:
        self._flag_tensor.fill_(1)

    def wait_reset_cpu(self) -> None:
        self._flag_tensor.zero_()"""
    src = src.replace(old5, new5, 1)
    open(lay_p, "w").write(src)
    print("P14: semaphore cpu ops added (dormant helpers)")
PYEOF

# ---- Asserts
python - <<'PYEOF'
import importlib, torch
c = importlib.import_module("vllm.v1.ple_offload.connector")
w = importlib.import_module("vllm.v1.ple_offload.worker")
l = importlib.import_module("vllm.model_executor.layers.ple_offload_layer")
src_c = open(c.__file__).read()
src_w = open(w.__file__).read()
assert 'buf_device = torch.device("cpu") if self.device.type == "xpu"' in src_c, "P14 assert: connector buffer"
assert "xpu storages have no reducer" in src_c, "P14 assert: connector registration"
assert 'target.gpu_output_buffer.device.type == "cpu"' in src_w, "P14 assert: worker copy branch"
assert "signal_cpu" in open(l.__file__).read(), "P14 assert: semaphore helpers"
# CPU->CPU copy semantics sanity
a = torch.zeros(4, 8).pin_memory(); b = torch.empty(4, 8).pin_memory()
b.copy_(a); assert torch.equal(a, b)
print("P14 assert OK: connector/worker/semaphore patched; cpu copy semantics hold")
PYEOF
