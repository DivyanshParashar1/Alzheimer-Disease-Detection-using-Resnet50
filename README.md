
# Alzheimer’s Disease MRI Classification with ResNet50

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

This project utilizes Deep Learning to classify Alzheimer's Disease stages from MRI scans. It implements a **Transfer Learning** approach using a pre-trained **ResNet50** architecture, fine-tuned on the Augmented Alzheimer MRI dataset. The training pipeline was optimized using a dual-GPU setup on Kaggle.

## Project Overview

* **Goal:** Classify MRI scans into 4 stages of Dementia.
* **Model:** ResNet50 (Pre-trained on ImageNet).
* **Technique:** Transfer Learning + Fine-Tuning.
* **Accuracy:** ~95.8% (Validation), ~99.6% (Training).

## Dataset

The dataset used is the [Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset) provided by Uraninjo on Kaggle.

It consists of MRI images organized into four classes:
1.  **Non Demented**
2.  **Very Mild Demented**
3.  **Mild Demented**
4.  **Moderate Demented**

*> **Note:** The training pipeline includes a custom `safe_loader` to robustly handle non-image files or corrupt data points often found in raw datasets.*

## Methodology

1.  **Data Preprocessing:**
    * Images resized to `224x224`.
    * Normalized using ImageNet standards (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
    * Safe loading mechanism to prevent training crashes on corrupt files.

2.  **Architecture:**
    * **Backbone:** ResNet50.
    * **Head:** Custom fully connected layer:
        * `Linear(2048 -> 512)`
        * `ReLU` + `Dropout(0.3)`
        * `Linear(512 -> 4)`

3.  **Training Strategy:**
    * **Stage 1 (Feature Extraction):** Frozen backbone, trained only the classifier head (Achieved ~64% accuracy).
    * **Stage 2 (Fine-Tuning):** Unfrozen all layers, retrained with a low learning rate (`1e-5`) to adapt to specific medical imaging features (Achieved ~95% accuracy).

## Usage

### 1. Installation
Ensure you have PyTorch installed. You can install the dependencies via pip:

```bash
pip install torch torchvision matplotlib pillow scikit-learn

```

### 2. Loading the Model (`.pth`)

The model was trained using `nn.DataParallel` (multiple GPUs). If you are loading the `.pth` file on a machine with a single GPU or CPU, you must remove the `module.` prefix from the keys.

Use the following snippet to load the model correctly:

```python
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

# 1. Define the Architecture
def get_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 4)
    )
    return model

# 2. Load Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
state_dict = torch.load('alzheimer_resnet50.pth', map_location=device)

# 3. Handle 'DataParallel' keys (remove 'module.' prefix)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") 
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

print("Model loaded successfully.")

```

### 3. Inference Example

To predict the class of a new image:

```python
from PIL import Image
from torchvision import transforms

# Define Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def predict(image_path, model):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
        
    return classes[predicted.item()]

# Usage
# print(predict('path/to/mri.jpg', model))

```

## Results

| Metric | Score |
| --- | --- |
| **Training Accuracy** | 99.57% |
| **Validation Accuracy** | 95.76% |
| **Training Loss** | 0.0130 |
| **Validation Loss** | 0.1366 |

## Credits

* **Dataset:** [Uraninjo (Kaggle)](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)
* **Framework:** PyTorch

