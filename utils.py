import torch

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M1 Metal GPU acceleration!")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVidia Cuda acceleration!")
print()
