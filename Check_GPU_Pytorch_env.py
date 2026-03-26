import torch
import subprocess

print("==== Driver ====")
subprocess.run(["nvidia-smi"])

print("\n==== PyTorch CUDA ====")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    GPU = torch.cuda.get_device_name(0)
    VRAM = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    print(f"{GPU} is available with {VRAM}GB V-RAM")
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
else:
    print("No GPU available")