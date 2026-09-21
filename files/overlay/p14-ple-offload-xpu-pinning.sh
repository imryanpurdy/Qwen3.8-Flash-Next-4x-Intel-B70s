#!/usr/bin/env bash
# ============================================================================
# p14-ple-offload-xpu-pinning.sh — P14 v2: XPU port of fork PLE offload
#                                  (v24h2 rollback-image doctrine)
#
# Context (2026-09-21, overnight lane; boots #10-#11 crash chain):
# The fork's PLE offload is CUDA-IPC by design; on XPU torch 2.13+xpu has no
# python reducer for xpu storages => ForkingPickler registration dies
# "_share_filename_: only available on CPU". Blueprint = stage-v24h2:rollback
# (the OLD BOX PROVEN implementation, ran this model on XPU):
#   - CpuGpuSemaphore: xpu => flag = torch.zeros(1,int32).share_memory_()
#     ("Level Zero does not expose CUDA-compatible stream-memory or
#     device-IPC primitives here") + blocking host polling.
#   - output buffer: cpu + share_memory_() on xpu; device IPC only on cuda.
#   - worker: copy_stream=None for CPU targets => plain copy_ + signal();
#     wait block skips copy_stream.synchronize(); pinned_bufs pin only when
#     a real stream exists.
#   - connector: pin_input_buffers + d2h_event_pool gated to cuda; xpu lane
#     stages inputs SYNCHRONOUSLY (blocking copies, no D2H event).
#   - layer consumer: after host-flag wait, .to(hidden_states.device) —
#     the v24c lesson (CPU buffer must convert to device for downstream
#     xpu matmuls). Graph-capture branch = graphs lane, next boot.
# Patch v1 failed in build 8 (worker-block anchor guess; also kept the
# event-pool machinery live on xpu, which dies at torch.cuda.current_stream).
# v2 uses exact anchor text pulled from the running image. Registration needs
# NO detach: buffers/flags are CPU-shared on xpu and pickle via file_system.
# ============================================================================
set -euo pipefail

SP=/opt/venv/lib/python3.12/site-packages
CONN="$SP/vllm/v1/ple_offload/connector.py"
WORK="$SP/vllm/v1/ple_offload/worker.py"
LAY="$SP/vllm/model_executor/layers/ple_offload_layer.py"

python - "$CONN" "$WORK" "$LAY" <<'PYEOF'
import sys

conn_p, work_p, lay_p = sys.argv[1:4]
TAG = "P14 (2026-09-21)"

def sub(path, old, new, label):
    src = open(path).read()
    if TAG in src and label in src:
        print(f"P14: {label} already applied")
        return
    if old not in src:
        print(f"P14: {label} anchor NOT FOUND in {path} — drift, refusing", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print(f"P14: {label} applied")

# ================= layer: semaphore xpu arm + host sync branches ===========
sub(lay_p,
'''    def __init__(self, device: torch.device) -> None:
        self._flag_tensor = torch.zeros(1, dtype=torch.int32, device=device)''',
'''    def __init__(self, device: torch.device) -> None:
        if device.type == "xpu":  # P14 (2026-09-21): v24h2 doctrine — shared CPU flag
            self._flag_tensor = torch.zeros(1, dtype=torch.int32).share_memory_()
        else:
            self._flag_tensor = torch.zeros(1, dtype=torch.int32, device=device)''',
"semaphore-init")

sub(lay_p,
'''    @property
    def flag_tensor(self) -> torch.Tensor:
        """Return the CUDA tensor used to share the semaphore through IPC."""
        return self._flag_tensor''',
'''    @property
    def flag_tensor(self) -> torch.Tensor:
        """Return the tensor used to share the semaphore through IPC."""
        return self._flag_tensor

    @property
    def is_host_synchronized(self) -> bool:  # P14 (2026-09-21)
        """Whether synchronization uses a shared host flag."""
        return self._flag_tensor.device.type == "cpu"''',
"semaphore-host-prop")

sub(lay_p,
'''    def reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=0)`` on ``stream``."""
        if stream is None:''',
'''    def reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=0)`` on ``stream``."""
        if self.is_host_synchronized:  # P14 (2026-09-21)
            self._flag_tensor.fill_(self.RESET_VALUE)
            return
        if stream is None:''',
"semaphore-reset")

sub(lay_p,
'''    def signal(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=1)`` on ``stream``."""
        if stream is None:''',
'''    def signal(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=1)`` on ``stream``."""
        if self.is_host_synchronized:  # P14 (2026-09-21)
            self._flag_tensor.fill_(self.DONE_VALUE)
            return
        if stream is None:''',
"semaphore-signal")

sub(lay_p,
'''    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WaitValue32(flag==0)`` on ``stream``."""
        if stream is None:''',
'''    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WaitValue32(flag==0)`` on ``stream``."""
        if self.is_host_synchronized:  # P14 (2026-09-21)
            import time as _time
            while self._flag_tensor.item() != self.RESET_VALUE:
                _time.sleep(0.0001)
            return
        if stream is None:''',
"semaphore-waitreset")

# ---- layer consumer: host-wait + device conversion (v24c lesson) ----------
sub(lay_p,
'''        if self._is_cpu_offloaded:
            torch.ops.vllm.ple_offload_wait(''',
'''        if self._is_cpu_offloaded:
            if getattr(self._sem, "is_host_synchronized", False):  # P14 (2026-09-21): xpu lane
                import time as _time
                while int(self._sem._flag_tensor.item()) != self._sem.DONE_VALUE:
                    _time.sleep(0.0001)
                return self._gpu_output_buffer[: input_ids.shape[0]].to(
                    device=hidden_states.device, non_blocking=False
                )
            torch.ops.vllm.ple_offload_wait(''',
"layer-forward")

# ---- custom-op wait impl: host branch for CPU flags ------------------------
sub(lay_p,
'''    """Wait for the CPU result without releasing its output buffer."""
    stream = torch.cuda.current_stream()''',
'''    """Wait for the CPU result without releasing its output buffer."""
    if sem_flag_tensor.device.type == "cpu":  # P14 (2026-09-21): host flag
        import time as _time
        while int(sem_flag_tensor.item()) != 1:
            _time.sleep(0.0001)
        return
    stream = torch.cuda.current_stream()''',
"wait-impl")

# ================= connector: xpu = synchronous lane ========================
sub(conn_p,
'''            # The CPU worker writes results here through CUDA IPC. The GPU
            # placeholder waits on the paired cross-process semaphore.
            output_buffer = torch.empty(
                max_num_tokens,
                layer.get_offload_output_dim(int(config.ple_embed_dim)),
                dtype=layer.get_offload_output_dtype(vllm_config.model_config.dtype),
                device=self.device,
            )
            layer.setup_cross_process_offload(''',
'''            # The CPU worker writes results here through CUDA IPC. The GPU
            # placeholder waits on the paired cross-process semaphore.
            # P14 (2026-09-21): xpu uses shared host output (Level Zero has
            # no CUDA-compatible stream-memop/device-IPC transport here).
            out_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            output_buffer = torch.empty(
                max_num_tokens,
                layer.get_offload_output_dim(int(config.ple_embed_dim)),
                dtype=layer.get_offload_output_dtype(vllm_config.model_config.dtype),
                device=out_device,
            )
            if output_buffer.device.type == "cpu":
                output_buffer.share_memory_()
            layer.setup_cross_process_offload(''',
"connector-buffers")

sub(conn_p,
'''                with torch.accelerator.device_index(self.device.index):
                    self._pin_input_buffers()
                    self._d2h_event_pool = queue.Queue(
                        maxsize=vllm_config.max_concurrent_batches
                    )
                    for _ in range(vllm_config.max_concurrent_batches):
                        self._d2h_event_pool.put_nowait(torch.cuda.Event())
                self._start_request_thread(ipc_addr)''',
'''                with torch.accelerator.device_index(self.device.index):
                    if self.device.type == "cuda":  # P14 (2026-09-21): xpu = sync lane
                        self._pin_input_buffers()
                        self._d2h_event_pool = queue.Queue(
                            maxsize=vllm_config.max_concurrent_batches
                        )
                        for _ in range(vllm_config.max_concurrent_batches):
                            self._d2h_event_pool.put_nowait(torch.cuda.Event())
                self._start_request_thread(ipc_addr)''',
"connector-init-gate")

sub(conn_p,
'''        assert self._d2h_event_pool is not None, "PLE D2H event pool is not initialized"
        try:
            d2h_done_event = self._d2h_event_pool.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(
                "PLE has more requests than configured concurrent batches"
            ) from exc
        self._enqueue_cuda_inputs(request, d2h_done_event)
        self._request_queue.put_nowait(
            _PendingPleOffloadRequest(request, d2h_done_event)
        )''',
'''        if self.device.type == "xpu":
            # P14 (2026-09-21): synchronous staging on xpu — blocking copies,
            # no D2H event pool; the connector thread publishes after copies.
            with torch.accelerator.device_index(self.device.index):
                self._input_ids_buf[: request.num_tokens].copy_(
                    self._input_ids_source[: request.num_tokens]
                )
                self._query_start_loc_buf[: request.num_reqs + 1].copy_(
                    self._query_start_loc_source[: request.num_reqs + 1]
                )
                if self._ngram_context_buf is not None:
                    assert self._ngram_context_source is not None
                    self._ngram_context_buf[: request.num_reqs].copy_(
                        self._ngram_context_source[: request.num_reqs]
                    )
            self._request_queue.put_nowait(
                _PendingPleOffloadRequest(request, None)
            )
            return
        assert self._d2h_event_pool is not None, "PLE D2H event pool is not initialized"
        try:
            d2h_done_event = self._d2h_event_pool.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(
                "PLE has more requests than configured concurrent batches"
            ) from exc
        self._enqueue_cuda_inputs(request, d2h_done_event)
        self._request_queue.put_nowait(
            _PendingPleOffloadRequest(request, d2h_done_event)
        )''',
"connector-launch")

sub(conn_p,
'''        event_pool = self._d2h_event_pool
        event = pending.d2h_done_event
        assert event_pool is not None, "PLE D2H event pool is not initialized"
        with torch.accelerator.device_index(self.device.index):
            event.synchronize()

        socket.send(msgspec.msgpack.encode(request))
        event_pool.put_nowait(event)''',
'''        if self.device.type == "xpu":  # P14 (2026-09-21): no D2H event on xpu
            socket.send(msgspec.msgpack.encode(request))
            return
        event_pool = self._d2h_event_pool
        event = pending.d2h_done_event
        assert event_pool is not None, "PLE D2H event pool is not initialized"
        with torch.accelerator.device_index(self.device.index):
            event.synchronize()

        socket.send(msgspec.msgpack.encode(request))
        event_pool.put_nowait(event)''',
"connector-process-request")

# ================= worker: CPU-target dual branch ===========================
sub(work_p,
'''    copy_stream: torch.cuda.Stream''',
'''    copy_stream: torch.cuda.Stream | None  # P14 (2026-09-21): None = CPU target''',
"worker-dataclass")

sub(work_p,
'''                    copy_stream=torch.cuda.Stream(device=gpu_buffer.device),''',
'''                    copy_stream=(
                        None
                        if gpu_buffer.device.type == "cpu"
                        else torch.cuda.Stream(device=gpu_buffer.device)
                    ),  # P14 (2026-09-21)''',
"worker-stream-guard")

sub(work_p,
'''                    dtype=self._layers[layer_name].get_offload_output_dtype(
                        self.vllm_config.model_config.dtype
                    ),
                    pin_memory=True,''',
'''                    dtype=self._layers[layer_name].get_offload_output_dtype(
                        self.vllm_config.model_config.dtype
                    ),
                    pin_memory=any(
                        target.copy_stream is not None for target in targets
                    ),  # P14 (2026-09-21)''',
"worker-pin-conditional")

sub(work_p,
'''                for target in targets:
                    target.copy_stream.synchronize()
                    target.sem.wait_reset(target.copy_stream)''',
'''                for target in targets:
                    if target.copy_stream is not None:  # P14 (2026-09-21)
                        target.copy_stream.synchronize()
                    target.sem.wait_reset(target.copy_stream)''',
"worker-wait-block")

sub(work_p,
'''                for target in targets:
                    with torch.cuda.stream(target.copy_stream):
                        target.gpu_output_buffer[slices].copy_(
                            result[slices], non_blocking=True
                        )
                        target.sem.signal(target.copy_stream)''',
'''                for target in targets:
                    if target.copy_stream is None:  # P14 (2026-09-21): CPU target
                        target.gpu_output_buffer[slices].copy_(result[slices])
                        target.sem.signal()
                    else:
                        with torch.cuda.stream(target.copy_stream):
                            target.gpu_output_buffer[slices].copy_(
                                result[slices], non_blocking=True
                            )
                            target.sem.signal(target.copy_stream)''',
"worker-copy-block")
PYEOF

# ---- Asserts ---------------------------------------------------------------
python - <<'PYEOF'
import importlib, torch
from multiprocessing.reduction import ForkingPickler
import torch.multiprocessing as torch_mp

l = importlib.import_module("vllm.model_executor.layers.ple_offload_layer")
c = importlib.import_module("vllm.v1.ple_offload.connector")
w = importlib.import_module("vllm.v1.ple_offload.worker")

sem = l.CpuGpuSemaphore(torch.device("xpu"))
assert sem.is_host_synchronized, "P14 assert: xpu semaphore must be host-sync"
sem.signal(); assert int(sem.flag_tensor.item()) == 1
sem.reset(); assert int(sem.flag_tensor.item()) == 0

torch_mp.set_sharing_strategy("file_system")
blob = ForkingPickler.dumps(sem.flag_tensor)
assert len(blob) > 0

a = torch.zeros(4, 8).share_memory_(); b = torch.empty(4, 8)
b.copy_(a[:4, :]); assert torch.equal(a, b)
print("P14 assert OK: xpu semaphore host-sync + pickle + cpu copy hold")
PYEOF
