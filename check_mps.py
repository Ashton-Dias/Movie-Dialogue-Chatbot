import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Set *before* importing torch

import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device: {device}")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")
