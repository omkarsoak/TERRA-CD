import torch
import segmentation_models_pytorch as smp

class HRSCD3:
    """Combined CD and LCM model with checkpoint management"""
    def __init__(self, cd_architecture='unet', lcm_architecture='unet',
                 cd_encoder='resnet34', lcm_encoder='resnet34',
                 input_channels=3, num_cd_classes=13, num_semantic_classes=4):
        # Initialize CD model
        self.cd_model = self._create_cd_model(
            architecture=cd_architecture,
            encoder=cd_encoder,
            input_channels=input_channels
        )
        # Initialize LCM model
        self.lcm_model = self._create_lcm_model(
            architecture=lcm_architecture,
            encoder=lcm_encoder,
            input_channels=input_channels,
            num_semantic_classes=num_semantic_classes
        )

    def _create_cd_model(self, architecture, encoder, input_channels):
        """Create binary change detection model"""
        if architecture.lower() == 'unet':
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=input_channels*2,  # Concatenated images
                classes=1,  # Binary output,
                encoder_depth=4,  # Reduce depth (def=5)
                decoder_channels=(256, 128, 64, 32)  # Reduce channels(def=(256, 128, 64, 32, 16))

            )
        elif architecture.lower() == 'deeplabv3plus':
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=input_channels*2,
                classes=1,
            )
        # Add more architectures as needed
        return model

    def _create_lcm_model(self, architecture, encoder, input_channels, num_semantic_classes=4):
        """Create land cover mapping model"""
        if architecture.lower() == 'unet':
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=input_channels,
                classes=num_semantic_classes,  # 4 land cover classes
            )
        elif architecture.lower() == 'deeplabv3plus':
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=input_channels,
                classes=num_semantic_classes,
            )
        # Add more architectures as needed
        return model

    def to(self, device):
        """Move models to device"""
        self.cd_model = self.cd_model.to(device)
        self.lcm_model = self.lcm_model.to(device)
        return self

    def train(self):
        """Set models to training mode"""
        self.cd_model.train()
        self.lcm_model.train()

    def eval(self):
        """Set models to evaluation mode"""
        self.cd_model.eval()
        self.lcm_model.eval()