"""Check GPU availability and CUDA setup."""
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB")
else:
    print("\n⚠️  CUDA is NOT available!")
    print("This usually means PyTorch was installed without CUDA support.")
    print("\nTo fix this, reinstall PyTorch with CUDA:")
    print("pip uninstall torch")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
