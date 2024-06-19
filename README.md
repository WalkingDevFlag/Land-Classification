# Land-Classification

## Overview

This project aims to develop a deep learning model for land segmentation using satellite imagery. The model uses a U-Net architecture with an EfficientNet-B7 encoder and is trained using a Dice Loss function. The model is designed to classify different land cover types, including urban land, agriculture land, rangeland, forest land, water, barren land, and unknown.

## Requirements
- Python 3.6+
- PyTorch 1.9+
- PyTorch Lightning 1.4+
- Albumentations 0.2.0+
- OpenCV 4.5.1+
- Matplotlib 3.5.1+
- NumPy 1.20.0+
- Pandas 1.3.5+
- Scikit-learn 1.0.2+


## Dataset

The dataset consists of satellite images and their corresponding segmentation masks, organized into images and mask directories. The class definitions are stored in a 'class_dict.csv' file.

I've used the DeepGlobe Land-Cover Dataset


## Model Architecture
The project uses a U-Net model with an EfficientNet-B7 encoder, pretrained on ImageNet. The model is trained to predict the segmentation masks for the input images.


## Training and Evaluation
The code sets up PyTorch Lightning modules for training, validation, and testing the model. It uses a DiceLoss function and various segmentation metrics (accuracy, IoU, precision, recall, F1-score) to evaluate the model's performance.


## Usage
Prepare Data:
- Organize your dataset into the following directories
```
images: contains the satellite images
mask: contains the corresponding segmentation masks
class_dict.csv: contains the class definitions
```

## Run the Code
- Install the required dependencies using pip:
```
bash
pip install torch torchvision albumentations opencv-python matplotlib numpy pandas scikit-learn
```

- Run the provided Python script to train and evaluate the model.


## Configuration
- Batch Size: Set the batch size in the batch_size variable.
- Number of Epochs: Set the number of epochs in the max_epochs variable.
- Early Stopping: Set the early stopping patience in the early_stop_callback variable.
- Model Checkpointing: Set the model checkpoint directory in the checkpoint_callback variable.


## Results
The trained model can be used for inference on new satellite images. The model's performance can be evaluated using various metrics, including accuracy, IoU, precision, recall, and F1-score.
