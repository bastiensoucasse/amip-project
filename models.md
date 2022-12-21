# Models

## Explanations

In the paper by Johnson et al. (2016), the authors use the output of the first 23 layers of the VGG-16 model as the feature maps for computing the feature loss. These layers correspond to the convolutional and pooling layers of the model, and contain high-level information about the image such as edges, textures, and patterns.

The reason for using only the first 23 layers is that the later layers of the VGG-16 model contain increasingly fine-grained information about the image, which may not be relevant for the task of super-resolution. Using only the first 23 layers allows the model to focus on the high-level features of the image, rather than getting bogged down in the details.

"For our feature loss, we use the same layers as in Gatys et al. (2015) and use the VGG-16 model (Simonyan & Zisserman, 2014) as a fixed feature extractor. Specifically, we use the same layers as in Gatys et al. (2015) – the output of the layers conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1 – and pass the generated HR image and the ground truth HR image through the VGG-16 network. We then compute the L2 distance between the corresponding feature maps as our feature loss."

The layers mentioned in this excerpt are the first 5 convolutional layers of the VGG-16 model, which corresponds to the first 23 layers in PyTorch's implementation of the model.

## Improvements

### Sum the feature loss and the pixel loss

The sum of the feature loss and the pixel loss is used as the total loss because both losses contribute to the quality of the generated image.

The pixel loss measures the difference between the generated image and the high resolution reference image in terms of pixel values. It is computed using the mean squared error (MSE) loss function, which measures the average squared difference between the pixel values of the two images. The pixel loss helps to ensure that the generated image is visually similar to the high resolution reference image, and that it has similar pixel values.

The feature loss, on the other hand, measures the difference between the generated image and the high resolution reference image in terms of the features extracted by the pre-trained VGG-16 model. The VGG-16 model has been trained on a large dataset of images, and has learned to extract features that are useful for various vision tasks, such as object recognition. By computing the feature loss between the generated image and the high resolution reference image, we can ensure that the generated image has similar features to the high resolution reference image, and that it is "realistic" in the sense that it is similar to natural images that the VGG-16 model has seen before.

In summary, the pixel loss helps to ensure that the generated image is visually similar to the high resolution reference image, while the feature loss helps to ensure that the generated image is realistic and has similar features to natural images. By adding the pixel loss and the feature loss, we can optimize for both of these factors, and achieve a high quality generated image.
