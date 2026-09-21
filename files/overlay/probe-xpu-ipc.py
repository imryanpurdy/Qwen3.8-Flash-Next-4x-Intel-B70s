import torch
t = torch.zeros(2, device="xpu")
print("share_cuda_", hasattr(t.storage(), "_share_cuda_"))
import torch.multiprocessing.reductions as R
src = open(R.__file__).read()
print("xpu_mentions_in_reductions:", src.count("xpu"))
print("xpu_ipc_api:", [a for a in dir(torch.xpu) if "ipc" in a.lower()])
s = torch.cuda.Stream(device=t.device)
print("cuda.Stream on xpu ok:", s is not None)
e = torch.cuda.Event()
print("cuda.Event ok:", e is not None)
print("memops:", [a for a in dir(torch.cuda) if "Write" in a or "Wait" in a])
import inspect
print("CpuGpuSemaphore wait uses:", "cuda" in inspect.getsource(__import__("vllm.model_executor.layers.ple_offload_layer", fromlist=["CpuGpuSemaphore"]).CpuGpuSemaphore.wait_reset) if True else "?")
