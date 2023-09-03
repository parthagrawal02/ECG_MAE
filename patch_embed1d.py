import torch
from torch import nn
import torch.nn.functional as F

from trace_utils import _assert

class PatchEmbedECG(nn.Module):
    def __init__(
        self,
        seq_length: int = 12000,  # Assuming 10 seconds sampled at 100 Hz
        patch_size: int = 50,   # Adjust this based on your requirement
        embed_dim: int = 128,
        norm_layer: nn.Module = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_length // patch_size

        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        _assert(
            L % self.patch_size == 0,
            f"Input sequence length ({L}) should be divisible by patch size ({self.patch_size})."
        )

        x = self.proj(x)
        x = x.transpose(1, 2)  # B, L', C
        x = self.norm(x)
        return x

# Example usage
seq_length = 1000
patch_size = 100
embed_dim = 128

# Create a PatchEmbed module for ECG data
patch_embed_ecg = PatchEmbedECG(seq_length=seq_length, patch_size=patch_size, embed_dim=embed_dim)

# Generate random 12-lead ECG data
x = torch.randn(1, 1, seq_length)  # Shape: (batch_size, 1, sequence_length)

# Get patch embeddings
patch_embeddings = patch_embed_ecg(x)
print("Patch embeddings shape:", patch_embeddings.shape)
