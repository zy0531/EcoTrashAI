# EcoTrashAI

## Inroduction

Classifying trash types is essential for proper disposal, helping to protect the ecosystem and facilitate recycling. This project develops a trash classifier using state-of-the-art computer vision technology. The classifier can be used locally as a Gradio app or deployed on Hugging Face as a web app.

## Dataset

[GARBAGE CLASSIFICATION 3 Dataset v2](https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2): GARBAGE CLASSIFICATION 3 Dataset v2 is a Yolo (You Only Look Once) dataset with images and annotations with the form `<object-class> <x_center> <y_center> <width> <height>`. We use only the `<object-class>` as the target variable. Images are categorized into six classes, labeled as `BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, and `PLASTIC`.

## Exploratory Data Analysis (EDA)
A visualization of the 6 trash classes.

![](/images/class.jpg)

The proportions of classes in the training set, validation set, and the test set are different.

![](/images/data.jpg)

## Metrics and Modeling

- Metric
  - score
    - [x] precision: if false positive is an issue
    - [x] recall: if false negative is an issue
    - [x] $F1$: balance
    - [ ] beta $F1$: $(1+\beta^2) \frac{\mathrm{precision} \cdot \mathrm{recall}}{\beta^2 \mathrm{precision} + \mathrm{recall} }$, $\beta = 1$ is F1, $\beta < 1$, precision is weighted more, $\beta > 1$, recall is weighted more.
  - average method (e.g. precission)
    - [x] weighted average: take into account the class distribution, a balanced evaluation for imbalanced dataset
    - [ ] micro average: $\Sigma_i$ TP_i / ($\Sigma_i$ TP_i + $\Sigma_i$ FP_i), treats each instance equally, can be dominated by the majority.
    - [ ] macro average: 1/n $\Sigma_i$ Precision_i, treats each class equally, can highlight the minority class
- Loss function
  - cross entropy
    
- Optimizer
  - `optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)`
  - learning curve: https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/

- Scheduler
  - `ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.1, verbose=True)`

  
- Early stopping
  - stop training if the loss function does not decrease after 8 epochs.

- Model:
  -  EfficientNet is an architecture which applys `compound scaling` to systematically scale a model's dimentions to achieve a balance between model performance and computational cost. The model's width (number of channels; patterns and features), depth (number of layers; intrinsic representation of data), and resolution (image details) are scaled by factors $d=\alpha^\phi$, $w = \beta^\phi$, $r = \gamma^\phi$ s.t. $\alpha \beta^2 \gamma^2 ≈ 2$ and $\alpha \geq 1$, $\beta \geq 1$, $\gamma \geq 1$. The architecture begins with a base model. Next, user defines a compund coefficient $\phi$. Then the optimal $\alpha$, $\beta$, and $\gamma$ are chosen using optimization methods or grid search. Finally, scalings are applied to the base model to get an EfficientNet with a compund scaling $\phi$.
  
  - Base Model: Mobile Inverted Bottleneck (MBConv) layers, it first appliess pointwise $1 \times 1$ convolution to increase the number of layers, next, it applies depthwise convolution, keep the number of layers unchanges, then, it applies pointwise $1 \times 1$ convolution to reduce the number of channels back to the original, finally, residual connection (skip connection) is added to facilitate gradient flow.

  - Use case: There are eight EfficientNets, from `b0` to `b7`. We use `b0` in this work..
    - [x] b0, 20.17 MB, mobile app, fast inference, low power consumption
    - [ ] b1, 29.73 MB, mobile app,
    - [ ] b2, 34.75 MB, between b1 and b3
    - [ ] b3, 46.67 MB, web app, balance between accuracy and computation
    - [ ] b4, 73.78 MB, web app
    - [ ] b5, 115.93 MB, between b4 and b6
    - [ ] b6, 164.19 MB, high accuray
    - [ ] b7, 253.10 MB, high accuray

  - Links:
    - [Original paper](https://arxiv.org/abs/1905.11946)
    - [What is EfficientNet? The Ultimate Guide](https://blog.roboflow.com/what-is-efficientnet/)
    - [MobileNetV2](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5)
    - [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)

- Experiments
  - b0: pre-trained torchvision model
  - b0_long: pre-trained model with a modified classifier
    - modified classifier:
      ```
      classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=1280, out_features=512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, 6))
      ```
  - b0_aug: pre-trained model with data augmentation on the training set
    - data agumentation on the training set
      ```
      transform_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      ```
  - b0_long_aug: pretrained model with a modified classifier and data augmentation

## Analysis

|  | precision | recall | F1 |
|----------|----------|----------|----------|
| b0 | torchvision model| b0 with a | Row 1 Col 3 |
| b0_long | Row 2 Col 2 | Row 2 Col 3 | |
| b0_aug | Row 3 Col 2 | Row 3 Col 3 | |
| b0_long_aug | torchvision model| b0 with a | Row 1 Col 3 |

## Deployment

## Acknowledgement
- 
-
