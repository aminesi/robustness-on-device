# robustness-on-device

This repository contains all the source code and scripts used in the paper "Robustness Analysis of Deep Learning Frameworks on Mobile Platforms".

This study empirically compares two on-device deep learning frameworks (TensorFlow Lite and PyTorch Mobile) with three adversarial attacks (both white-box and black-box) on three different model architectures (MobileNetV2, ResNet50, and InceptionV3). The study also uses both the quantized and unquantized variants for each architecture, resulting in 36 configurations on mobile devices and 18 configurations on PC as a baseline for comparison.
The results show that (a) neither of the deep learning frameworks is better than the other in terms of robustness, and the results depend on the model and other factors. This finding was indeed observed both on mobile and PC frameworks, (b) in terms of robustness, there is not a significant difference between the PC and mobile frameworks either. However, in some cases, like the Boundary attack, the mobile version is more robust than the PC version. Finally (c), results demonstrate that quantization improves robustness in all cases when moving from PC to mobile.

The repository consists of two main parts. There is a PC section that contains codes to create (and quantize) models that produce adversarial samples and replicate the results on PC.
There is also a mobile section that contains source code for testing the robustness on mobile.

It is worth noting that datasets and models are not added to the repository because of their size, and they have to be downloaded separately from the related sites. 
