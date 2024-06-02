import torch

# Check if CUDA is available
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("CUDA is available. Using GPU.")
else:
	device = torch.device("cpu")
	print("CUDA is not available. Using CPU.")

# Create a tensor
X = torch.zeros(4, 2, dtype=torch.long)
print(f"Tensor: {X}")