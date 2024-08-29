from torch import nn
import torchvision
import torchvision.transforms as transforms


def customized_efficinet_b0(num_classes: int = 6) -> nn.Module:
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the required input size of the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad=False
    
    model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                           nn.Linear(in_features=1280, out_features=512),
                           nn.ReLU(),
                           nn.BatchNorm1d(512),
                           nn.Dropout(p=0.5),
                           nn.Linear(512, num_classes))

    return model, transform