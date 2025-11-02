# Long-Term ECG Noise Detection Using ResNet

This project focuses on automatic noise detection in long-term ECG recordings using a custom 1D ResNet model.  
It uses the BUTQDB dataset from [PhysioNet](https://physionet.org/content/butqdb/), processes ECG segments, applies augmentation, and classifies them as clean or noisy.

ResNet (Residual Network) was chosen for its skip-connection design, which helps preserve signal information and enables deeper, more stable training on ECG data.

---

## Features

- Custom 1D ResNet architecture designed for ECG signal classification  
- 5-fold cross-validation for robust model evaluation  
- On-the-fly data augmentation (noise, scaling, and shifting)  
- Early stopping and cosine annealing learning rate scheduling  
- Achieved high validation accuracy on PhysioNet BUTQDB dataset  

---

## Dataset

- **Dataset Name:** [BUTQDB - Brno University of Technology ECG Quality Database](https://physionet.org/content/butqdb/)  
- **Source:** PhysioNet  
- **Description:** Contains annotated ECG segments labeled as clean or noisy  

You must download the dataset manually from PhysioNet and place it in a folder named `dataset/` within this repository.

---

## Model Architecture

The model is based on a 1D ResNet with three residual blocks:

```python
class ResNetECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1)
        )
        self.layer1 = ResidualBlock(64)
        self.layer2 = ResidualBlock(64)
        self.layer3 = ResidualBlock(64)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 2)
