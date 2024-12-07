# Abstract



# Introduction

- Background and motivation of the project
- State the concrete project goal
- Comprehensive review of existing literature (recent advancements and state-of-the-art)

Semantic segmentation, a foundational task in computer vision, involves partitioning an image at the pixel level into semantically meaningful regions. This task has evolved from earlier image classification and object detection methods to instance and panoptic segmentation. Semantic segmentation has wide applications in fields like medical imaging and autonomous driving. The development of semantic segmentation can be divided into three major stages. Early methods relied on hand-crafted features such as edge detection or color segmentation. While straightforward, these approaches struggled with complex images and image variations.

Recent advancements in this field have primarily been driven by deep learning. The introduction of the Fully Convolutional Network (FCN) marked a significant shift in 2015, as it enabled end-to-end segmentation by replacing fully connected layers with convolutional layers and leveraging upsampling techniques. U-Net refined this approach by incorporating an encoder-decoder structure with "short-cuts", combining low-level feature maps (rich in detail) with high-level ones (rich in semantic information). This architecture improved the precision of segmentation masks and excelled in medical imaging tasks. Meanwhile, the DeepLab series introduced multi-scale feature fusion and dilated convolutions, which enhanced the model’s ability to adapt to complex and extended the receptive field without increasing parameters.

More recently, Transformer-based models have achieved state-of-the-art performance in semantic segmentation. Vision Transformer (ViT) replaced convolutional operations with a self-attention mechanism, enabling the modeling of global dependencies within an image. It introduced a tokenized approach to redefine image tasks in a manner similar to textual tasks and highlighted the potential of Transformers for multimodal data. SegFormer incorporated multi-level Transformer decoder blocks with multi-layer perceptrons (MLPs) to balance accuracy and efficiency. Swin Transformer introduced a hierarchical sliding-window attention mechanism, which significantly reduced computational complexity while preserving the ability to capture global dependencies. These advancements have positioned Transformer-based architectures as the dominant paradigm in semantic segmentation, achieving superior performance across various benchmarks.

This project aims to leverage semantic segmentation techniques to address a specific challenge in image editing. At Apple’s recent iPhone launch event, the company introduced a feature called \textit{Photographic Styles}, which allows users to adjust image colors while preserving characteristic colors. This feature ensures that editing the background of an image does not overly alter a subject’s skin tone or brighten their hair. It relies on advanced hardware for real-time image segmentation when taking photos, making it unavailable on older iPhones and Android devices. The objective of this project is to develop a similar functionality that can be applied on existing images. To achieve this, a pre-trained MobileViT+DeepLabV3 model will be fine-tuned on a labeled human parsing dataset. Subsequently, a Look-Up Table (LUT) filter will be selectively applied to the image, modifying all pixels except those corresponding to the skin. The Great Lakes HPC at University of Michigan is utilized to accelerate the training process.

# Method

- Problem formulation: learning problem formulation (input, output), dataset description and model formulation
- Walk through the methodologies used

We applied a series of data augmentation techniques to enhance the diversity of the training dataset and improve the model’s generalization ability. The augmentations included random horizontal flipping, Gaussian noise addition, and image downscaling to simulate low-quality inputs. Additionally, we introduced variations in brightness, contrast, saturation, and hue to account for different lighting conditions, as well as structural distortions like pixel dropout and gravel noise to mimic real-world imperfections. These transformations were applied probabilistically to the training data, while the validation data underwent only basic preprocessing to ensure consistent evaluation. This augmentation strategy was designed to make the model more robust to diverse and challenging scenarios.



# Results

- Data pipeline or model set up.
- Figures that present numerical simulation results
- Interpretation of the results

# Conclusion

