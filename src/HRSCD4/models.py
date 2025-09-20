import torch
import torch.nn as nn
import torch.nn.functional as F

class HRSCD4(nn.Module):
    def __init__(self, input_channels, num_semantic_classes, num_cd_classes):
        super().__init__()

        # Shared Encoder
        self.encoder = nn.Sequential(
            # Initial block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Land Cover Mapping Decoder (shared weights)
        self.lcm_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_semantic_classes, kernel_size=4, stride=2, padding=1)
        )

        # Change Detection Decoder
        self.cd_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_cd_classes, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x1, x2):
        # Ensure input images are the same size
        assert x1.shape == x2.shape, "Input images must have the same dimensions"

        # Encode both images
        enc1 = self.encoder(x1)
        enc2 = self.encoder(x2)

        # Land Cover Mapping for both time periods
        lcm1 = self.lcm_decoder(enc1)
        lcm2 = self.lcm_decoder(enc2)

        # Change Detection (using difference of encodings)
        cd_input = torch.abs(enc1 - enc2)  # Or torch.cat([enc1, enc2], dim=1)
        cd_output = self.cd_decoder(cd_input)

        return lcm1, lcm2, cd_output