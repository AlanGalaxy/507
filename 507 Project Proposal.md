# Overview

During Apple’s fall iPhone event this year, the company introduced a new feature called **Photographic Styles**, which offers great creative flexibility for adjusting image colors while preserving key “characteristic colors,” such as skin tones and hair colors. Humans are particularly sensitive to these colors, as deviations can make photos look unnatural. This feature ensures that editing the background of an image does not overly alter a subject’s skin tone or brighten their hair. 

However, this feature is exclusive to Apple’s latest devices, as it relies on advanced hardware to analyze photos during capture. The algorithm segments parts of the image, such as skin and hair, into distinct layers, making it incompatible with older iPhones or Android devices. My goal is to develop a similar feature that works across older iPhones and other Android phones, enabling color adjustments while preserving key characteristics.

To achieve this, I plan train a supervised learning model that is capable of generating a mask image from the input photos. Each pixel in the mask will represent a specific class (e.g., face, hair, arms, legs). This task, known as _semantic segmentation_, involves identifying and classifying individual image components. I plan to fine-tune Apple’s MobileViT + DeepLabV3 model on a labeled human parsing dataset, leveraging pre-trained segmentation models for efficient and accurate training.

After successfully identifying semantic features in the image, I will apply some Lookup Table (LUT) filters to modify other pixels while keeping the characteristic colors (e.g., skin and hair) unaffected. This part of the project will involve the use of OpenCV and NumPy libraries for image processing.

Through this project, I aim to develop skills in Python programming and the application of deep learning models. Additionally, I seek to gain a comprehensive understanding of semantic segmentation advancements, the principles of image filter application, and the operational workflow of Great Lakes HPC computing clusters.

# Prior Work

Semantic segmentation is a foundational task in computer vision, aiming to partition an image at the pixel level into semantically coherent regions. It evolves from earlier tasks such as image classification and object recognition and has been extended by instance segmentation and panoptic segmentation. Semantic segmentation has wide applications in fields like medical imaging and autonomous driving.

The development of semantic segmentation can be divided into three major stages. Early methods relied on hand-crafted features such as edge detection or color segmentation. While straightforward, these approaches struggled with complex images and image variations.

The rise of deep learning, marked by the success of AlexNet in 2012, introduced convolutional neural networks (CNNs) to segmentation tasks. In 2015, the Fully Convolutional Network (FCN) replaced fully connected layers with convolutional layers, enabling input images of arbitrary sizes. FCN also introduced upsampling techniques, which achieved end-to-end semantic segmentation for the first time. Later, U-Net refined this approach by incorporating an encoder-decoder structure with "short-cuts", combining low-level feature maps rich in detail with high-level feature maps rich in semantic information. This architecture improved the precision of segmentation masks and excelled in medical imaging tasks. Meanwhile, the DeepLab series introduced atrous (dilated) convolutions, extending the receptive field without increasing parameters. This was combined with multi-scale feature fusion, enhancing adaptability to complex scenes.

More recently, the introduction of Transformers has driven further advancements in semantic segmentation. Transformers replaced local convolutions with self-attention mechanism, enabling the modeling of global dependencies and contextual information. Vision Transformer (ViT) was the first to demonstrate the application of Transformers to image data, dividing images into patches (16x16 pixels) and flatten them. This approach redefined visual tasks in a manner similar to textual tasks and highlighted the potential of Transformers for multimodal data. SegFormer combines multi-level transformer decoder blocks with multi-layer perceptrons (MLPs), balancing accuracy and efficiency. Swin Transformer introduces a hierarchical structure with a sliding-window attention mechanism, significantly reducing computational complexity while retaining the ability to capture global context. These Transformer-based models currently dominate semantic segmentation benchmarks, achieving state-of-the-art performance across various datasets.

For my project, I chose Apple’s MobileViT + DeepLabV3-small model as the initial framework. MobileViT, pre-trained on PASCAL VOC with a resolution of 512x512, serves as the backbone, while the DeepLabV3 head is integrated for semantic segmentation tasks. This combination provides a strong starting point for building an efficient and accurate model.

随着 Transformer 的发明，基于 Transformer 的方法进一步推动了语义分割的表现。Transformer 模型不依赖局部的卷积，而是通过自注意力机制建模全局特征。这使得其特别适合捕捉长距离依赖和图像的上下文信息。Vision Transformer (ViT) 最早提出了使用 Transformer 处理图像的框架，通过将图像分割成16\*16的 patch 展平并将其作为输入序列，有效地将视觉任务转化为类文本问题。证明了 transformer 模型具有处理多模态数据的能力，为后续模型研究奠定了基础。SegFormer 将不同层级的 transformer decoder block 输入一个 MLP，实现了语义分割精度与效率的平衡。Swin Transformer 则通过滑窗注意力机制引入层次化特征提取，有效降低计算复杂度，同时保持对全局信息的捕获能力。基于 Transformer 的模型在多个语义分割数据集上都处于 state-of-the-art 的位置。

我的项目选择来自苹果的 MobileViT + DeepLabV3 small 模型作为初始模型。MobileViT 模型使用 PASCAL VOC 以 512x512 的分辨率进行了预训练，DeepLab V3 的 head 添加到 MobileViT backbone for semantic segmentation. 

# Preliminary Results

For the project, I chose Apple’s MobileViT + DeepLabV3-small model as the initial model. MobileViT was pre-trained on PASCAL VOC with a resolution of 512x512, and a DeepLabV3 head is integrated for semantic segmentation tasks. This combination provides a strong starting point for building an efficient and accurate model. This model has a relatively small parameter size of 6.4 million, which makes it easier and more suitable for deployment on mobile devices with limited computational power.

I used the Human Parsing Data dataset for fine-tune, which consists of 17,706 samples of fashionably dressed individuals. Each sample includes an 600\*400 original image and a corresponding mask image. The mask has 18 categories that segment the image pixels into regions such as background, hair, face, arms, and legs. This dataset is highly suitable for my project as it provides detailed labels of human skin and hair, which is crucial for the task of fine-tuning the model. However, the large number of images in the dataset made the training process slow without hardware acceleration. To overcome this, I utilized the Great Lakes HPC at the university, leveraging GPU acceleration to speed up the training process.

In terms of implementation, I used basic Python knowledge and the PyTorch library to fine-tune the model. For visualizing the results, I used the Matplotlib library to plot images. I also explored NumPy and OpenCV for image manipulation and applying filters. The performance of the model is evaluated using Intersection over Union (IoU), a common metric in semantic segmentation. Figure 1 presents a comparison between performance of the original model and the fine-tuned model on a same image. It shows how fine-tuning improves the model’s ability to segment the image.

, especially in capturing finer details such as facial features, hair, and limbs, with an increase in IoU scores for the respective regions. This demonstrates the model’s enhanced ability to preserve key features after fine-tuning.

# Project Deliverables

- What will a successful project produce?

Successfully fine tune the model. Get a good performance on segmentation.

- What are the sub-goals?

能力上：熟悉 Git 的流程。掌握 great lakes 的使用。学习如何对现有深度学习模型进行微调。学习图像语义分割的主要模型。练习python 和pytorch的使用。

ability to fine tune a model

python ability with pytorch

learn more about (get familiar with) apple silicon metal acceleration

# Timeline

- Week 1-2: 通过ChatGPT学习git的版本管理。浏览 hugging face网站，寻找可能的项目模型和数据集。学习密歇根大学 great lakes  HPC 的使用方法。苹果 Metal 加速的方法。
- Week 3-4: 对所选项目进行literature review，学习前人的成果。确定我们将要使用的模型。开始写fine tune 模型的代码。完成 research proposal
- Week 5: 整理 fine tune 代码到 notebook。完成 final report。