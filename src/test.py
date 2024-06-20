import torch

max_sequence_length = 512  # Example value

# Create a tensor with values from 0 to max_sequence_length - 1
sequence_tensor =  even_i = torch.arange(0, 10, 2)
stacked = torch.stack([sequence_tensor, sequence_tensor], dim=1)
print(stacked)