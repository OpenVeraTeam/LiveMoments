import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from diffusers.models.modeling_utils import ModelMixin

class MotionEncoder(ModelMixin):
    def __init__(
        self,
        conditioning_channels: int = 2,
        out_channels: Tuple[int] = (1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536)
    ):
        super(MotionEncoder, self).__init__()
        
        self.blocks = nn.ModuleList()
        current_channels = conditioning_channels
        
        layers = [
            nn.Conv2d(current_channels, 512, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(1024, 1536, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(1536, 1536, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        ]
        self.blocks.append(nn.Sequential(*layers))
        current_channels = 1536

        for _ in range(11):  
            self.blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, 1536, kernel_size=1, stride=1, padding=0),
                nn.SiLU()
            ))

    def forward(self, x):
        embeddings = []
        embedding = x
        for block in self.blocks:
            embedding = block(embedding)
            B, C, H, W = embedding.shape 
            embeddings.append(embedding.view(B, C, H * W).transpose(1, 2))
            embeddings.append(embedding.view(B, C, H * W).transpose(1, 2))
        return embeddings