from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import ResNetConfig, ResNetForImageClassification
from transformers.models.resnet.modeling_resnet import ResNetModel
from transformers import ConvNextImageProcessor
from torchvision.transforms import (
    Compose, 
    RandomHorizontalFlip, 
    RandomCrop, 
    ToTensor, 
    Normalize
)

# 1. Define a CIFAR-specific Embedding layer (The "Stem")
# Standard ResNet shrinks image 4x immediately. We want 1x (no shrinking).
class ResNetCifarEmbeddings(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        # CIFAR ResNet uses a 3x3 conv, stride 1, padding 1
        self.embedder = nn.Conv2d(
            config.num_channels, 
            config.embedding_size, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.norm = nn.BatchNorm2d(config.embedding_size)
        self.activation = nn.ReLU()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Allow fp16 inputs if needed
        if pixel_values.dtype == torch.float16 and self.embedder.weight.dtype == torch.float32:
            pixel_values = pixel_values.to(torch.float32)
            
        x = self.embedder(pixel_values)
        x = self.norm(x)
        x = self.activation(x)
        return x

# 2. Define the Custom Model Class
class ResNet20ForCIFAR(ResNetForImageClassification):
    def __init__(self, config: ResNetConfig) -> None:
        super(ResNetForImageClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.resnet = ResNetModel(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )

        # SWAP: Replace the standard ImageNet embedder with our CIFAR embedder
        self.resnet.embedder = ResNetCifarEmbeddings(config)

        # initialize weights and apply final processing
        self.post_init()

        self.loss_type = "ForSequenceClassification"
        self._train_batchnorm = True  # Track BatchNorm training state
        self.frozen_model: list[ResNet20ForCIFAR] = []

    def train(self, mode: bool = True) -> ResNet20ForCIFAR:
        """
        Override train() to respect the BatchNorm training setting.
        """
        super().train(mode)
        if mode and not self._train_batchnorm:
            # If model is set to train mode but BatchNorm should be frozen
            self.set_batchnorm_training(False)
        return self
    
    def set_batchnorm_training(self, train_batchnorm: bool) -> None:
        """
        Control whether BatchNorm2d layers should be trained or frozen.
        
        Args:
            train_batchnorm (bool): If False, BatchNorm layers will use running statistics
                                   and not update them, even when model is in train mode.
                                   If True, BatchNorm layers behave normally.
        """
        self._train_batchnorm = train_batchnorm
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if train_batchnorm:
                    # Normal training behavior: use batch stats and update running stats
                    module.train()
                else:
                    # Freeze behavior: use running stats and don't update them
                    module.eval()
    
    # In your custom ResNet20ForCIFAR class
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Any:
        output = super().forward(
            pixel_values=pixel_values, 
            labels=labels, 
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict
        )
        return output

# 3. Configure for ResNet-20
# ResNet-20 = 3 stages of 3 blocks each. 
# Typical CIFAR widths are [16, 32, 64].
config = ResNetConfig(
    depths=[3, 3, 3],
    downsample_in_first_stage=False,
    embedding_size=16,
    hidden_act="relu",
    hidden_sizes=[16, 32, 64],

    num_labels=10,  
    label2id={"plane": 0, "car": 1, "bird": 2, "cat": 3, "deer": 4, 
              "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9},
    id2label={0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"},
    layer_type="basic",
    model_type="resnet",
    num_channels=3,
)
# Define the processor for 32x32 CIFAR-10 images
image_processor = ConvNextImageProcessor(
    size={"height": 32, "width": 32},
    image_mean=[0.4914, 0.4822, 0.4465],
    image_std=[0.2023, 0.1994, 0.201],
    do_resize=False,                      # Input is usually already 32x32
    do_rescale=True,                      # Rescale 0-255 to 0-1
    do_normalize=True                     # Apply mean/std
)

# 1. Training Transforms (Augmentation + Normalization)
train_transforms = Compose([
    RandomHorizontalFlip(),
    RandomCrop(32, padding=4),
    ToTensor(),
    # We manually apply the normalization matching the processor
    Normalize(mean=image_processor.image_mean, std=image_processor.image_std) 
])

# 2. Validation Transforms (Just Normalization)
val_transforms = Compose([
    ToTensor(),
    Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])
