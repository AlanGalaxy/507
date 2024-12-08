# Stats507-coursework

Stats 507 mainly concerns about python programming and some machine learning 
packages. This repository is to store the coursework and controls the version.

The course project is about fine-tuning a semantic segmentation model 
(Apple’s MobileViT + DeepLabV3-small) on Human Parsing Data dataset, and 
apply Lut on the images while protects some characteristic colors, such as 
skin tones and hair colors.

The report for the project is `507_Project_Report.pdf`. The code can be found 
in `507_Project.ipynb`. Be sure to run the first two code blocks to install the 
necessary libraries.

The folder `model` contains the fine-tuned model. The folder `resources` 
contains example image and Look-Up Tables (LUTs). These will be downloaded when 
you run the inference code blocks in `507_Project.ipynb`.