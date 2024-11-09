# Testing torch
import torch_geometric
import torch
print("Testing torch")
print(f"torch=={torch.__version__}")
x = torch.rand(5, 3)
print(x)

# Testing CUDA
print("Testing CUDA")
print(f"CUDA available=={torch.cuda.is_available()}")

# Testing torch-geometric
print("Testing torch-geometric")
print(f"torch-geometric=={torch_geometric.__version__}")
