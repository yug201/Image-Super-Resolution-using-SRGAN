# Image Super-Resolution using SRGAN
## Overview

<img width="537" height="266" alt="image" src="https://github.com/user-attachments/assets/3cd37274-1ba2-4baa-97f9-61dbbe191b9b" />


This project implements an **Image Super-Resolution GAN (SRGAN)** using PyTorch.

The goal is to generate a high-resolution image from a low-resolution input. The model performs **2× super-resolution**, converting `128×128` images into `256×256` images.

The system contains three main components:

* **Generator** — converts LR images into super-resolved images.
* **Discriminator** — distinguishes real HR images from generated images.
* **VGG19** — provides perceptual features used in the Generator loss.

## Workflow

```text
LR Image (128×128)
        ↓
    Generator
        ↓
SR Image (256×256)
        ↓
 ┌──────┴──────┐
 ↓             ↓
Discriminator  VGG19
 ↓             ↓
Adversarial   Perceptual
   Loss          Loss
 └──────┬──────┘
        ↓
 Update Generator
```

## Notebook Flow

1. Import required libraries and define configuration.
2. Load paired low-resolution and high-resolution images.
3. Apply image preprocessing and create a custom Dataset.
4. Create DataLoaders and verify a sample batch.
5. Build the Generator using convolutional and residual blocks.
6. Upsample the image using PixelShuffle.
7. Build the Discriminator for real/fake image classification.
8. Load pretrained VGG19 features for perceptual loss.
9. Train the Generator and Discriminator together.
10. Generate super-resolved images from validation samples.
11. Visualize LR, SRGAN output, and ground-truth HR images.
12. Save the trained model weights.
13. Use the trained Generator for new images.

## Main Technologies

* Python
* PyTorch
* Torchvision
* CNNs
* GANs
* Residual Learning
* PixelShuffle
* VGG19 Perceptual Loss

The final Generator can be used independently to convert new low-resolution images into higher-resolution outputs.
