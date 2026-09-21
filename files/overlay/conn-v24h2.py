# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exchange PLE data between a GPU worker and the CPU-offload process."""

import os
import queue
import threading
from multiprocessing.reduction import ForkingPickler
from typing import Any

import msgspec
import torch
import torch.nn as nn
import zmq

try:
    from cuda.bindings import driver as cuda_driver
except ImportError:
    cuda_driver = None  # type: ignore[assignment]

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    CpuGpuSemaphore,
    PleOffloadLayer,
)
from vllm.v1.ple_offload.protocol import (
    PleOffloadRegistration,
    PleOffloadRequest,
)
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


def _require_cuda_driver() -> None:
    """Fail closed if the CUDA PLE connector lacks its driver bindings."""
    if cuda_driver is None:
        raise RuntimeError(
            "PLE CUDA transport requires the cuda-python package; "
            "a non-CUDA transport must be selected on this platform"
        )


class PleOffloadConnector:
    """Connect a GPU runner to the shared PLE CPU worker.

    MRV1 and MRV2 share the same CPU-input and CUDA-output IPC protocol.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        device: torch.device,
        ipc_addr: str,
        *,
        input_ids_source: torch.Tensor,
        query_start_loc_source: torch.Tensor,
        ngram_context_source: torch.Tensor | None,
        input_ids_np=None,
        query_start_loc_np=None,
        ngram_context_np=None,
        staging_flag=None,
        staging_input_ids=None,
        staging_query_start_loc=None,
        staging_ngram_context=None,
    ) -> None:
        self.device = device
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_rank = get_tp_group().rank_in_group
        self._layers = self._setup_layers(vllm_config, model)

        # Both runner paths stage into the same shared buffers. TP0 registers
        # them with CUDA so MRV2 can use asynchronous D2H copies.
        scheduler_config = vllm_config.scheduler_config
        self._input_ids_buf = torch.empty(
            scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._query_start_loc_buf = torch.empty(
            scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._ngram_context_buf = None
        config = vllm_config.model_config.hf_text_config
        ngram_context_len = int(config.ngram_size) - 1
        if ngram_context_len > 0:
            self._ngram_context_buf = torch.empty(
                scheduler_config.max_num_seqs,
                ngram_context_len,
                dtype=torch.int32,
                device="cpu",
            ).share_memory_()

        # Runner input allocations are address-stable, so bind them once and
        # pass only batch sizes through the per-forward request queue.
        self._input_ids_source = input_ids_source
        self._query_start_loc_source = query_start_loc_source
        self._ngram_context_source = ngram_context_source
        # v2: address-stable numpy mirrors of the same pinned host allocations
        # (created at CpuGpuBuffer init before UVA re-typing). Copies between
        # numpy views never construct a torch op -> never enter the Level Zero
        # command path (root cause of the TP0 appendUSMMemcpy wedge).
        self._input_ids_np_src = input_ids_np
        self._query_start_loc_np_src = query_start_loc_np
        self._ngram_context_np_src = ngram_context_np
        # V3RUNNER: pinned double-buffered staging (runner-allocated).
        # Runner thread enqueues stream-ordered D2H copies then a flag
        # write; connector thread receives the (slot, seq) token inside
        # its own queue payload, so no shared mutable handoff state.
        self._staging_flag_t = staging_flag
        self._staging_input_ids_t = staging_input_ids
        self._staging_query_start_loc_t = staging_query_start_loc
        self._staging_ngram_context_t = staging_ngram_context
        self._staging_flag_np = staging_flag.numpy() if staging_flag is not None else None
        self._staging_input_ids_np = staging_input_ids.numpy() if staging_input_ids is not None else None
        self._staging_query_start_loc_np = staging_query_start_loc.numpy() if staging_query_start_loc is not None else None
        self._staging_ngram_context_np = staging_ngram_context.numpy() if staging_ngram_context is not None else None
        self._stage_ok = (
            self._staging_flag_t is not None
            and self._staging_input_ids_t is not None
            and self._staging_query_start_loc_t is not None
            and (self._ngram_context_buf is None
                 or self._staging_ngram_context_t is not None)
        )
        self._stage_slot = 0
        self._stage_seq = 0
        self._seq_dev = None
        # XPU runner V2 exposes its  mirror as a UVA accelerator view
        # (device.type == xpu), so gate on accelerator-ness, not is_cuda.
        self._uses_cuda_inputs = (
            self._input_ids_source.is_cuda
            or self._input_ids_source.device.type in ("xpu", "cuda")
        )
        self._validate_input_sources()
        # V3RUNNER miswire canary: wiring verdict printed every boot.
        logger.info(
            "PLE v3 staging wired: staged=%s env_host_stage=%s "
            "legacy_np_src=%s uses_cuda_inputs=%s tp_rank=%s",
            self._stage_ok,
            os.environ.get("VLLM_PLE_HOST_STAGE", "1"),
            self._input_ids_np_src is not None,
            self._uses_cuda_inputs,
            self.tp_rank,
        )

        self._pinned_input_buffers: list[torch.Tensor] = []
        # PLE rejects DBO, and each forward consumes its output before the
        # next launch, so one pending request is sufficient.
        self._request_queue: queue.Queue[PleOffloadRequest | None] = queue.Queue(
            maxsize=1
        )
        self._request_thread: threading.Thread | None = None
        self._request_thread_ready = threading.Event()
        self._zmq_ctx: zmq.Context | None = None
        self._registration_socket: zmq.Socket | None = None
        self._d2h_stream: torch.cuda.Stream | None = None
        self._input_ready_event: torch.cuda.Event | None = None
        self._d2h_done_event: torch.cuda.Event | None = None

        try:
            self._zmq_ctx = zmq.Context()
            self._registration_socket = self._zmq_ctx.socket(zmq.PUSH)
            self._registration_socket.connect(ipc_addr)
            self._register_with_offload_worker(vllm_config, ipc_addr)

            if self.tp_rank == 0:
                # ForkingPickler may replace CPU storage while converting its
                # sharing strategy, so register only the final addresses.
                with torch.accelerator.device_index(self.device.index):
                    if self.device.type == "cuda":
                        self._pin_input_buffers()
                    if self._uses_cuda_inputs:
                        # v22: XPU passes device-resident sources too.
                        # Under the runner's _torch_cuda_wrapper these
                        # map to torch.xpu.Stream/Event on XPU.
                        self._d2h_stream = torch.cuda.Stream(device=self.device)
                        self._input_ready_event = torch.cuda.Event()
                        self._d2h_done_event = torch.cuda.Event()
                        print(f"V22DIAG init events device={self.device.type} uses_cuda={self._uses_cuda_inputs}", flush=True)
                self._start_request_thread(ipc_addr)
        except Exception:
            self.close()
            raise

    def _setup_layers(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
    ) -> dict[str, PleOffloadLayer]:
        """Attach output buffers and semaphores to GPU PLE placeholders."""
        layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )

        config = vllm_config.model_config.hf_text_config
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        for layer in layers.values():
            # CUDA uses device IPC. XPU eager bring-up uses shared host output
            # because Level Zero device IPC and stream-memory waits are not
            # available through the CUDA transport used by this feature.
            output_device = self.device if self.device.type == "cuda" else "cpu"
            output_buffer = torch.empty(
                max_num_tokens,
                int(config.ple_embed_dim),
                dtype=layer.get_offload_output_dtype(vllm_config.model_config.dtype),
                device=output_device,
            )
            if output_buffer.device.type == "cpu":
                output_buffer.share_memory_()
            layer.setup_cross_process_offload(
                output_buffer,
                CpuGpuSemaphore(self.device),
            )
        return layers

    def _pin_input_buffers(self) -> None:
        """Page-lock shared input allocations without replacing their storage."""
        _require_cuda_driver()
        buffers = [self._input_ids_buf, self._query_start_loc_buf]
        if self._ngram_context_buf is not None:
            buffers.append(self._ngram_context_buf)
        for buffer in buffers:
            if buffer.device.type != "cpu" or not buffer.is_shared():
                raise RuntimeError("PLE input buffers must be shared CPU tensors")
            if not buffer.is_contiguous():
                raise RuntimeError("PLE input buffers must be contiguous")
            _cuda_check(
                cuda_driver.cuMemHostRegister(
                    buffer.data_ptr(),
                    buffer.numel() * buffer.element_size(),
                    cuda_driver.CU_MEMHOSTREGISTER_PORTABLE,
                ),
                "cuMemHostRegister(PLE input buffer)",
            )
            self._pinned_input_buffers.append(buffer)
            if not buffer.is_pinned():
                raise RuntimeError("CUDA did not page-lock a PLE input buffer")

    def _unpin_input_buffers(self) -> None:
        """Release CUDA registrations after the request thread has stopped."""
        _require_cuda_driver()
        for buffer in reversed(self._pinned_input_buffers):
            try:
                _cuda_check(
                    cuda_driver.cuMemHostUnregister(buffer.data_ptr()),
                    "cuMemHostUnregister(PLE input buffer)",
                )
            except RuntimeError:
                logger.exception("Failed to unregister a PLE input buffer")
        self._pinned_input_buffers.clear()

    def _register_with_offload_worker(
        self, vllm_config: VllmConfig, ipc_addr: str
    ) -> None:
        """Register CUDA IPC outputs and shared CPU inputs with the worker."""
        # Each GPU worker owns distinct output buffers, while TP0's shared
        # inputs become the request source for its DP rank.
        registration = PleOffloadRegistration(
            worker_id=(
                self.dp_rank * vllm_config.parallel_config.world_size
                + vllm_config.parallel_config.rank
            ),
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            gpu_output_buffers={
                name: layer._gpu_output_buffer for name, layer in self._layers.items()
            },
            sem_flag_tensors={
                name: layer._sem.flag_tensor for name, layer in self._layers.items()
            },
            input_ids_buf=self._input_ids_buf,
            query_start_loc_buf=self._query_start_loc_buf,
            ngram_context_buf=self._ngram_context_buf,
        )

        # ForkingPickler transmits tensors through shared-memory and CUDA IPC.
        import torch.multiprocessing as torch_mp

        original_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            payload = ForkingPickler.dumps(registration)
        finally:
            torch_mp.set_sharing_strategy(original_strategy)
        assert self._registration_socket is not None
        self._registration_socket.send(payload)

        logger.info(
            "PleOffload: registered %d PleOffloadLayer(s) "
            "(dp_rank=%d, tp_rank=%d, ipc_addr=%s): %s",
            len(self._layers),
            self.dp_rank,
            self.tp_rank,
            ipc_addr,
            sorted(self._layers),
        )

    def _start_request_thread(self, ipc_addr: str) -> None:
        """Start the thread that publishes batches after inputs are ready."""
        self._request_thread = threading.Thread(
            target=self._request_loop,
            args=(ipc_addr,),
            name=f"ple-offload-dp{self.dp_rank}",
            daemon=True,
        )
        self._request_thread.start()
        if not self._request_thread_ready.wait(timeout=10):
            raise RuntimeError("Timed out starting the PLE request thread")

    def _request_loop(self, ipc_addr: str) -> None:
        """Stage fixed runner inputs, then notify the CPU worker."""
        socket: zmq.Socket | None = None
        try:
            if self._zmq_ctx is None:
                raise RuntimeError("PLE ZMQ context closed before thread startup")
            socket = self._zmq_ctx.socket(zmq.PUSH)
            socket.connect(ipc_addr)
            self._request_thread_ready.set()
            while True:
                item = self._request_queue.get()
                if item is None:
                    return
                request, staging = item
                self._process_request(request, socket, staging)
        except Exception:
            logger.exception("PLE request thread failed")
            os._exit(1)
        finally:
            self._request_thread_ready.set()
            if socket is not None:
                socket.close(linger=0)

    def _process_request(
        self,
        request: PleOffloadRequest,
        socket: zmq.Socket,
        staging=None,
    ) -> None:
        """Stage one batch from fixed sources and publish its request."""
        if self._uses_cuda_inputs and self.device.type == "cuda":
            self._copy_cuda_inputs(request)
        elif self._uses_cuda_inputs:
            # V3RUNNER: consume the runner-staged mirror via numpy only.
            import time as _time
            import numpy as _np
            num_tokens = request.num_tokens
            num_reqs = request.num_reqs
            if (staging is not None
                    and os.environ.get("VLLM_PLE_HOST_STAGE", "1") != "0"):
                slot, seq = staging
                deadline = _time.monotonic() + 2.0
                while int(self._staging_flag_np[slot]) != seq:
                    if _time.monotonic() > deadline:
                        logger.error(
                            "PLE v3 staging flag timeout slot=%d seq=%d; "
                            "legacy copy_ fallback", slot, seq)
                        self._copy_legacy_cuda(request)
                        break
                    _time.sleep(0.0002)
                else:
                    _np.copyto(self._input_ids_buf.numpy()[:num_tokens],
                               self._staging_input_ids_np[slot, :num_tokens])
                    _np.copyto(
                        self._query_start_loc_buf.numpy()[: num_reqs + 1],
                        self._staging_query_start_loc_np[slot, : num_reqs + 1])
                    if self._ngram_context_buf is not None:
                        _np.copyto(
                            self._ngram_context_buf.numpy()[:num_reqs],
                            self._staging_ngram_context_np[slot, :num_reqs])
            else:
                self._copy_legacy_cuda(request)
        else:
            self._copy_cpu_inputs(request)

        with record_function_or_nullcontext("ple_offload.send_request"):
            socket.send(msgspec.msgpack.encode(request))

    def _copy_cpu_inputs(self, request: PleOffloadRequest) -> None:
        """Stage MRV1's existing CPU mirrors in the notifier thread."""
        num_tokens = request.num_tokens
        num_reqs = request.num_reqs
        with record_function_or_nullcontext("ple_offload.copy_input_ids"):
            self._input_ids_buf[:num_tokens].copy_(self._input_ids_source[:num_tokens])
        with record_function_or_nullcontext("ple_offload.copy_query_start_loc"):
            self._query_start_loc_buf[: num_reqs + 1].copy_(
                self._query_start_loc_source[: num_reqs + 1]
            )
        if self._ngram_context_buf is not None:
            assert self._ngram_context_source is not None
            with record_function_or_nullcontext("ple_offload.copy_ngram_context"):
                self._ngram_context_buf[:num_reqs].copy_(
                    self._ngram_context_source[:num_reqs]
                )

    def _validate_input_sources(self) -> None:
        """Validate fixed runner sources against shared input buffers."""
        sources = [
            ("input_ids", self._input_ids_source, self._input_ids_buf),
            (
                "query_start_loc",
                self._query_start_loc_source,
                self._query_start_loc_buf,
            ),
        ]
        if (self._ngram_context_source is None) != (self._ngram_context_buf is None):
            raise ValueError("PLE ngram_context source and buffer must match")
        if self._ngram_context_source is not None:
            assert self._ngram_context_buf is not None
            sources.append(
                (
                    "ngram_context",
                    self._ngram_context_source,
                    self._ngram_context_buf,
                )
            )

        expected_device = self.device if self._uses_cuda_inputs else torch.device("cpu")
        for name, source, buffer in sources:
            if (
                source.device != expected_device
                or source.dtype != buffer.dtype
                or source.ndim != buffer.ndim
                or source.shape[0] < buffer.shape[0]
                or source.shape[1:] != buffer.shape[1:]
            ):
                raise ValueError(
                    f"PLE {name} source is incompatible: "
                    f"src_dev={source.device} exp={expected_device} "
                    f"src_dtype={source.dtype} buf_dtype={buffer.dtype} "
                    f"src_ndim={source.ndim} buf_ndim={buffer.ndim} "
                    f"src0={source.shape[0]} buf0={buffer.shape[0]} "
                    f"src_rest={tuple(source.shape[1:])} buf_rest={tuple(buffer.shape[1:])}"
                )

    def _copy_cuda_inputs(self, request: PleOffloadRequest) -> None:
        """Stage MRV2 inputs on the background D2H stream."""
        if (
            self._d2h_stream is None
            or self._input_ready_event is None
            or self._d2h_done_event is None
        ):
            raise RuntimeError("PLE D2H resources are not initialized")

        with torch.accelerator.device_index(self.device.index):
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(self._input_ready_event)
                with record_function_or_nullcontext("ple_offload.copy_input_ids"):
                    self._input_ids_buf[: request.num_tokens].copy_(
                        self._input_ids_source[: request.num_tokens],
                        non_blocking=True,
                    )
                with record_function_or_nullcontext("ple_offload.copy_query_start_loc"):
                    self._query_start_loc_buf[: request.num_reqs + 1].copy_(
                        self._query_start_loc_source[: request.num_reqs + 1],
                        non_blocking=True,
                    )
                if self._ngram_context_buf is not None:
                    assert self._ngram_context_source is not None
                    with record_function_or_nullcontext(
                        "ple_offload.copy_ngram_context"
                    ):
                        self._ngram_context_buf[: request.num_reqs].copy_(
                            self._ngram_context_source[: request.num_reqs],
                            non_blocking=True,
                        )
                self._d2h_done_event.record(self._d2h_stream)
            with record_function_or_nullcontext("ple_offload.wait_d2h"):
                self._d2h_done_event.synchronize()

    def stage_from_device(self):
        """V3RUNNER: runner-thread stream-ordered D2H staging + seq flag.

        Enqueues on the current stream in program order: full-width
        mirror copies device->pinned host, THEN the flag write carrying
        the per-boot monotonic seq. In-order queue => flag completes
        after the mirrors (behaviorally verified 300/300 trials,
        test-inorder.py, idle engine; campaign is the real proof).
        Returns the (slot, seq) token for the queue payload, or None.
        """
        if not self._stage_ok or self.tp_rank != 0:
            return None
        slot = self._stage_slot
        if self._seq_dev is None:
            self._seq_dev = torch.zeros(1, dtype=torch.int32, device=self.device)
        with torch.accelerator.device_index(self.device.index):
            st = self._staging_input_ids_t[slot]
            st.copy_(self._input_ids_source[: st.shape[0]], non_blocking=True)
            stq = self._staging_query_start_loc_t[slot]
            stq.copy_(self._query_start_loc_source[: stq.shape[0]], non_blocking=True)
            if self._staging_ngram_context_t is not None:
                stn = self._staging_ngram_context_t[slot]
                stn.copy_(self._ngram_context_source[: stn.shape[0]], non_blocking=True)
            self._seq_dev.fill_(self._stage_seq + 1)
            self._staging_flag_t[slot : slot + 1].copy_(self._seq_dev, non_blocking=True)
        self._stage_seq += 1
        self._stage_slot = (slot + 1) % 2
        return (slot, self._stage_seq)

    def _copy_legacy_cuda(self, request: PleOffloadRequest) -> None:
        """V3RUNNER fallback: original connector-thread copy_ path."""
        num_tokens = request.num_tokens
        num_reqs = request.num_reqs
        self._input_ids_buf[:num_tokens].copy_(
            self._input_ids_source[:num_tokens])
        self._query_start_loc_buf[: num_reqs + 1].copy_(
            self._query_start_loc_source[: num_reqs + 1])
        if self._ngram_context_buf is not None:
            assert self._ngram_context_source is not None
            self._ngram_context_buf[:num_reqs].copy_(
                self._ngram_context_source[:num_reqs])

    def _launch(
        self,
        num_reqs: int,
        num_tokens: int,
        staging=None,
    ) -> None:
        """Queue one batch while keeping staging off the model thread."""
        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        if self.tp_rank != 0:
            return

        if self._uses_cuda_inputs:
            assert self._input_ready_event is not None
            # v22: bare .record() targets the current stream and works on
            # both CUDA and XPU (torch.cuda.current_stream is CUDA-only
            # outside the runner's wrapper).
            self._input_ready_event.record()
        request = PleOffloadRequest(
            dp_rank=self.dp_rank,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        )
        # V3RUNNER: the (slot, seq) staging token rides in the payload;
        # bounded blocking put — a bare put_nowait would crash the engine
        # with queue.Full if the connector thread is briefly occupied.
        self._request_queue.put((request, staging), timeout=10)

    def prepare_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        dummy_run: bool,
    ) -> None:
        """Submit real inputs or satisfy the PLE wait for a dummy forward."""
        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        # V3RUNNER: stream-ordered D2H staging + flag on the runner
        # thread; the (slot, seq) token rides in the queue payload.
        staging = self.stage_from_device()
        self._launch(num_reqs, num_tokens, staging)

    def signal_dummy_outputs(self, num_tokens: int) -> None:
        """Locally satisfy PLE waits for dummy and capture forwards."""
        # Dummy and capture forwards do not send CPU requests, but every PLE
        # placeholder still waits for a completed output semaphore.
        stream = (
            torch.cuda.current_stream(self.device)
            if self.device.type == "cuda"
            else None
        )
        for layer in self._layers.values():
            layer._gpu_output_buffer[:num_tokens].zero_()
            layer._sem.signal(stream)

    def release_outputs(self) -> None:
        """Mark GPU output buffers reusable after the model consumes them."""
        # Reset only after the consumer forward so the CPU worker cannot
        # overwrite an output that a GPU PLE placeholder may still read.
        stream = (
            torch.cuda.current_stream(self.device)
            if self.device.type == "cuda"
            else None
        )
        for layer in self._layers.values():
            layer.release_offloaded_output(stream)

    def close(self) -> None:
        """Stop request transport and release host registrations."""
        request_thread = self._request_thread
        if request_thread is not None and request_thread.is_alive():
            try:
                self._request_queue.put(None, timeout=5)
            except queue.Full:
                logger.error("Timed out stopping the PLE request thread")
            request_thread.join(timeout=5)
        if request_thread is not None and request_thread.is_alive():
            # The thread may still access the registered buffers or ZMQ context.
            logger.error("PLE request thread did not stop; deferring resource cleanup")
            return
        self._request_thread = None

        if self._pinned_input_buffers:
            with torch.accelerator.device_index(self.device.index):
                self._unpin_input_buffers()
        self._d2h_done_event = None
        self._input_ready_event = None
        self._d2h_stream = None
        if self._registration_socket is not None:
            self._registration_socket.close(linger=0)
            self._registration_socket = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
