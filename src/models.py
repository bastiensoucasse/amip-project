import torch
import torch.nn as nn
import torchvision.models as models

import copy


class SuperResolutionLoss(nn.Module):
    def __init__(self, use_pixel_loss: bool = False) -> None:
        super(SuperResolutionLoss, self).__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        print(list(vgg.features)[:9])
        self.vgg = nn.Sequential(*list(vgg.features)[:9]).eval()
        self.criterion = nn.MSELoss()
        self.use_pixel_loss = use_pixel_loss
        for param in self.vgg.parameters():
            param.requires_grad = False

    

    def forward(self, gen_img: torch.Tensor, hr_img: torch.Tensor) -> nn.MSELoss:
        # Extract the feature maps from the VGG-16 model for the high resolution reference image and the generated image
        gen_features = self.vgg(gen_img)
        hr_features = self.vgg(hr_img)

        # Compute the MSE between the feature maps as the feature loss
        feature_loss = 0
        # for gen_f, hr_f in zip(gen_features, hr_features):
        #     feature_loss += self.criterion(gen_f, hr_f)
        feature_loss = torch.mean(torch.abs(gen_features - hr_features))

        # Return the feature loss if the improvement is not used
        if not self.use_pixel_loss:
            return feature_loss

        # Compute the pixel-wise loss between the generated image and the high resolution reference image
        pixel_loss = self.criterion(gen_img, hr_img)

        # Return the total loss
        return feature_loss + pixel_loss


class ImageTransformer(nn.Module):
    def __init__(self, scaling_factor: int) -> None:
        super(ImageTransformer, self).__init__()

        # Set parameters
        num_channels = 3
        num_filters = 64
        num_residual_blocks = 4

        # Save scaling factor
        self.scaling_factor = scaling_factor

        # Initialize the upsample layer
        self.reflection_pad = nn.ReflectionPad2d(1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=self.scaling_factor)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.batch_norm64 = nn.BatchNorm2d(num_filters)

        # Initialize the input convolutional layer
        self.conv_in = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=9, stride=1, padding=4)

        # Initialize the residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_residual_blocks)])

        # Initialize the middle convolutional layer
        self.conv_middle = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0)
        self.conv_middle1 = nn.ConvTranspose2d(in_channels=num_filters, out_channels=num_filters, kernel_size=4, stride=2, padding=1)

        # Initialize the output convolutional layer
        self.conv_out = nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=9, stride=1, padding=4)

        # Initialize activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # # Apply the upsample layer
        # x = self.upsample(x)

        # Apply the input convolutional layer
        # print("in " + str(x.shape))
        x = self.relu(self.batch_norm64(self.conv_in(x)))
        # print("conv_in " + str(x.shape))
        

        # Apply the residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        

        # print("residual_blocks " + str(x.shape))
        # Apply the set of middle convolutional layers
        # if self.scaling_factor == 4:
        #     x = self.relu(self.batch_norm64(self.conv_middle(self.reflection_pad(self.upsample(x)))))
        #     # print("middle1 " + str(x.shape))
        #     x = self.relu(self.batch_norm64(self.conv_middle(self.reflection_pad(self.upsample(x)))))
        # elif self.scaling_factor == 8:
        #     x = self.relu(self.batch_norm64(self.conv_middle(self.reflection_pad(self.upsample(x)))))
        #     # print("middle1 " + str(x.shape))
        #     x = self.relu(self.batch_norm64(self.conv_middle(self.reflection_pad(self.upsample(x)))))
        #     # print("middle2 " + str(x.shape))
        #     x = self.relu(self.batch_norm64(self.conv_middle(self.reflection_pad(self.upsample(x)))))

        if self.scaling_factor == 4:
            x = self.relu(self.batch_norm64(self.conv_middle1(x)))
            # print("middle1 " + str(x.shape))
            x = self.relu(self.batch_norm64(self.conv_middle1(x)))
        elif self.scaling_factor == 8:
            x = self.relu(self.batch_norm64(self.conv_middle1(x)))
            # print("middle1 " + str(x.shape))
            x = self.relu(self.batch_norm64(self.conv_middle1(x)))
            # print("middle2 " + str(x.shape))
            x = self.relu(self.batch_norm64(self.conv_middle1(x)))

        # print("middle " + str(x.shape))
        # Apply the output convolutional layer
        x = self.tanh(self.conv_out(x))

        x = torch.add(x, 1.)
        x = torch.mul(x, 0.5)

        # print("out " + str(x.shape))
        # print("----------------------")
        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int) -> None:
        super(ResidualBlock, self).__init__()

        # Initialize the convolutional layer
        self.conv = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_filters)

        # Initialize activation function
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Apply the first convolutional layer
        out = self.relu(self.batch_norm(self.conv(x)))

        # Apply the second convolutional layer
        out = self.batch_norm(self.conv(out))

        return x + out
