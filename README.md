# EcoTrashAI

## Inroduction

Classifying trash types is essential for proper disposal, helping to protect the ecosystem and facilitate recycling. This project develops a trash classifier using state-of-the-art computer vision technology. The classifier can be used locally as a Gradio app.

## Demo
- app link: https://huggingface.co/spaces/shl159/EcoTrashAI

![](images/demo.gif)

## Dataset

[GARBAGE CLASSIFICATION 3 Dataset v2](https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2): GARBAGE CLASSIFICATION 3 Dataset v2 is a Yolo (You Only Look Once) dataset with images and annotations with the form `<object-class> <x_center> <y_center> <width> <height>`. We use only the `<object-class>` as the target variable. Images are categorized into six classes, labeled as `BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, and `PLASTIC`.

## Exploratory Data Analysis (EDA)
A visualization of the 6 trash classes.

<img src="/images/class.jpg" alt="class" width="500"/>

The proportions of classes in the training set, validation set, and the test set are different.

<img src="/images/data.jpg" alt="data" width="800"/>

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
  - algorithm
    - get gradient: $g_t = \nabla f$
    - update biased first moment and second moment:
      - $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t = \sum_{i=1}^t \beta_1^{t-i}(1-\beta_1)g_i$
      - $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
    - compute unbiased moments: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
    - update parameter: $\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
  - pros
    - combine AdaGrad and RMSProp
    - adjust learning rate for each parameter individually
    - use momentum to speed up convergence and avoid oscillation
    - correct bias in the early stages of training
  - cons
    - converge to local minimum
    - require hyperparameter tuning
      - 

- Scheduler
  - [x] `ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.1, verbose=True)`  if the monitored metric does not improve for 4 epochs, the learning rate will be multiplied by 0.1, adaptive, may find local minima
  - [ ] `StepLR` reduce the learning rate by a factor(gamma every step_size epochs, easy to implement, not adaptive, does not consider the model performance
  - [ ] `ExponentialLR` decrease the learning rate by a factor gamma every epoch, continuous and smooth decrease, not adaptive, need careful selection of gamma
  - [ ] `CosineAnnealingLR(optimizer, T_max=50)` adjusts the learning rate according to a cosine function, decreasing it gradually to a minimum and then potentially restarting, can work cyclically
  - [ ] `CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000)` cycles the learning rate between two boundaries (base_lr and max_lr) with a certain frequency, avoid local minima, works well with large dataset
  - [ ] `OneCycleLR` increases the learning rate up to a maximum value (max_lr) and then decreases it again, adjust the momentum of the optimizer in an inverse manner, following a 1-cycle policy (Warm-Up Phase, Annealing Phase, Momentum Scheduling, Momentum Scheduling), fast convergence and better generalization, overkill for small datasets and simple models
  - [ ] `MultiplicativeLR` multiply the lerning rate by a customized function, flexible, not adaptive
  - [ ] `LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)`, customized schedule, flexible, not adaptive

  
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

- Learning Curve (https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
- <img src="/images/lc.jpg" alt="learning curve" width="800"/>

- Scores
- <img src="/images/score.jpg" alt="scores" width="500"/>
|  | accuracy | precision | recall | F1 score|
|----------|----------|----------|----------|----------|
| b0 | 0.493947 | 0.846667 | 0.916385	| 0.846667 | 0.875846 |
| b0_long | 0.381804 | 0.878333 | 0.946747 | 0.878333 | 0.908103 |
| b0_aug | 0.579488	| 0.825000 | 0.899522	| 0.825000 | 0.853236 |
| b0_long_aug | 0.501828 | 0.850000	| 0.887555 | 0.850000	| 0.858136 |

- Confusion Matrix of the Best model
<img src="/images/confusion_matrix.jpg" alt="confusion matrix" width="500"/>

- Conclusions
  - The validation set is easier than the training set. Models tend to perform better on the validation set.
  - The b0_long model has the best learning curve. The asymptotic behaviors on the training set and validation set are close.
  - The modified classier helps the b0_long model to learn the patten in the training set.
  - Data augmentation does not help improving the performance. Perhaps the model is too simple. Even leaning the default training set requires a modified classifier (b0_long).
  - A longer classifier can be tested in the future for performance improvement.

## Gradio App

- App Interface
<img src="/images/app.jpg" alt="scores" width="1000"/>

- classify an image
<img src="/images/classified.jpg" alt="scores" width="1000"/>


## Reference
- [GARBAGE CLASSIFICATION 3 Dataset v2](https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2)
- https://www.learnpytorch.io/
