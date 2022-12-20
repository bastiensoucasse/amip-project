import torch.nn as nn
import torchvision.models as models


class SuperResolutionLoss(nn.Module):
    def __init__(self, use_pixel_loss: bool = False):
        super(SuperResolutionLoss, self).__init__()

        # Load the pre-trained VGG-16 model
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Extract the layers of the VGG-16 model that we will use for feature loss computation
        vgg_layers = list(vgg16.features)[:23]

        self.vgg_layers = nn.Sequential(*vgg_layers)
        self.mse_loss = nn.MSELoss()
        self.use_pixel_loss = use_pixel_loss

    def forward(self, generated_img, high_res_img):
        # Extract the feature maps from the VGG-16 model for the high resolution reference image and the generated image
        gen_features = self.vgg_layers(generated_img)
        ref_features = self.vgg_layers(high_res_img)

        # Compute the MSE between the feature maps as the feature loss
        feature_loss = 0
        for gen_f, ref_f in zip(gen_features, ref_features):
            feature_loss += self.mse_loss(gen_f, ref_f)

        # Return the feature loss if the improvement is not used
        if not self.use_pixel_loss:
            return feature_loss

        # Compute the pixel-wise loss between the generated image and the high resolution reference image
        pixel_loss = self.mse_loss(generated_img, high_res_img)

        # Return the total loss
        return feature_loss + pixel_loss


class ImageTransformer(nn.Module):
    def __init__(self, num_channels=3, num_filters=64, num_residual_blocks=16):
        super(ImageTransformer, self).__init__()

        # Initialize the input convolutional layer
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=3, stride=1, padding=1)

        # Initialize the residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_residual_blocks)])

        # Initialize the output convolutional layer
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.Upsample(scale_factor=4, mode="nearest")(x)

        # Apply the input convolutional layer
        x = self.conv1(x)

        # Apply the residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        # Apply the output convolutional layer
        x = self.conv2(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()

        # Initialize the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Save the input to the residual block
        residual = x

        # Apply the first convolutional layer
        x = self.conv1(x)
        x = nn.functional.relu(x)

        # Apply the second convolutional layer
        x = self.conv2(x)

        # Add the input to the output of the second convolutional layer
        x += residual

        return x