# Neural_Networks_and_CNNs


A PyTorch-based deep learning project implementing feed-forward NNs, CNNs, VGG-13, and ResNet-34—and exploring optimization, regularization, and interpretability without using pre-built architectures.

## 🔍 Project Overview

This assignment is organized into four core parts plus a bonus:

- **Basic NN** – Basic Fully Connected NN on tabular data  
- **Optimizing NN** – Hyperparameter Tuning & Optimization Techniques  
- **Convolutional NN** – Convolutional Neural Network on a 36-class image dataset  
- **VGG-13** – Custom VGG-13 (“Version B”) for 36-class classification  
- **REsnet-34 and Interpretability**   – ResNet-34 from scratch & feature/activation/weight visualization  

Each part is delivered as a self-contained Jupyter notebook with:
- Preprocessing code  
- Model definition  
- Training & evaluation loops  
- Plots (accuracy, loss, ROC, confusion matrices)  
- Saved weights (.pt files)  

## 🎯 Why This Matters

Deep learning proficiency requires you to:
- Design and implement custom architectures (NN, CNN, VGG, ResNet)  
- Master PyTorch basics: nn.Module, optimizers, loss functions  
- Apply regularization (dropout, batch norm), learning-rate schedules, early stopping  
- Tune hyperparameters methodically and interpret metrics (accuracy, precision/recall/F1, AUC)  
- Visualize model predictions and gain insights via ROC curves and confusion matrices  
- Build reproducible experiments with notebooks, pickled weights, and clear reporting  

## 🗂 Table of Contents

1. [Quick Start](#quick-start)  
2. [Notebooks Reference](#notebooks-reference)  
3. [Architecture & Design](#architecture--design)  
4. [Key Features](#key-features)  
5. [Project Structure](#project-structure)  
6. [Usage Examples](#usage-examples)  
7. [Tech Stack](#tech-stack)  
8. [Future Enhancements](#future-enhancements)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [Authors & Contact](#authors--contact)  

---

## 🔧 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/YourOrg/cse574-assignment2-ml.git
cd cse574-assignment2-ml

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Run notebooks in sequence:

- BasicNN.ipynb
- OptimizingNN.ipynb
- ConvolutionalNN.ipynb
- VGG-13.ipynb
- Resnet-34.ipynb

## 📜 Notebooks Reference

| Notebook                          | Description                                        | Output Artifacts                                  |
|-----------------------------------|----------------------------------------------------|---------------------------------------------------|
| Basic_NN.ipynb                    | Basic NN: data loading, preprocessing, model build | BasicNN_best_model.pt, metrics & plots            |
| Optimizing_NN.ipynb               | NN optimization: dropout, activations, schedulers  | OptimizingNN_best_model.pt, tuning tables & plots |
| COnvolutional_NN.ipynb            | Convolutional_NN: conv layers, batch norm, pooling | Convolutional_NN_weights.pt, metrics &plots       |
| VGG-13.ipynb                      | VGG-13: deep conv blocks + FC layers               | vgg13_weights.pt, metrics & plots                 |
| Resnet-34.ipynb                   | ResNet-34 from scratch + interpretability hooks    | resnet34_weights.pt, saliency maps                |

## 🏗 Architecture & Design

**Data Preprocessing**  
– Tabular: non-numeric cleanup, imputation, scaling, oversampling  
– Images: torchvision.transforms for resizing, normalization  

**Model Classes**  
– FeedForwardNet – Modular MLP with configurable layers  
– ConvNet – 3-layer CNN with BatchNorm + Dropout  
– VGG13 – Custom Version B architecture adapted for grayscale  
– ResNet34 – Residual blocks with identity shortcuts  

**Training Pipeline**  
- Load & preprocess data  
- Instantiate model & move to device  
- Define loss (BCE or Cross-Entropy) & optimizer (SGD/Adam)  
- Training loop with optional LR scheduler & early stopping  
- Evaluate on validation & test sets  
- Save best weights with torch.save()  

**Evaluation**  
- Accuracy, Precision/Recall/F1, ROC AUC  
- Confusion matrix heatmaps & per-epoch loss/accuracy plots  

```
┌──────────┐   preprocess    ┌──────────┐   train    ┌──────────┐
│ raw data ├────────────────▶│ cleaned  ├──────────▶│ models   │
└──────────┘                 └──────────┘           └──────────┘
```

## 🔑 Key Features

**Part I – Basic NN**
- Clean non-numeric entries, impute means, scale & oversample
- 7→64→64→1 network; ReLU + Dropout; Sigmoid output

**Part II – Optimization**
- Dropout tuning & deeper architectures
- Activation experiments (Leaky ReLU, ELU)
- BatchNorm, EarlyStopping, LR Schedulers, K-Fold CV

**Part III – CNN**
- 3×(Conv→BatchNorm→ReLU→MaxPool) + FC layers
- Achieved ∼91% accuracy on 36-class 28×28 dataset

**Part IV – VGG-13**
- Five conv blocks (2×3×3 conv + pool) + three FC layers
- Adapted for 1-channel input & 36-way output; ∼91.5% accuracy

**ResNet-34 & Interpretability**
- Built residual blocks with skip connections
- Visualized feature maps, activations, and convolutional kernels

## 📁 Project Structure

```
cse574-assignment2-ml/
├── data/                       # Raw & preprocessed datasets
│   ├── tabular/                # dataset.csv
│   ├── cnn_dataset/            # 36-class images
│   └── flower_data/                  # Flower images for ResNet task
├── notebooks/                  # Jupyter notebooks for each part
│   ├── Basic_NN.ipynb
│   ├── Optimizing_NN.ipynb
│   ├── Convolutional_NN.ipynb
│   ├── VGG-13.ipynb
│   └── Resnet-34.ipynb
├── requirements.txt            # Python dependencies
└── README.md                   # This file
## 🚀 Usage Examples

```python
# Part I: Train basic NN
from feedforward import FeedForwardNet
model = FeedForwardNet(input_size=7, hidden_sizes=[64,64], dropout=0.5)
model.to(device)
# … load data into DataLoader …
train(model, train_loader, val_loader,
      criterion='BCE', optimizer='Adam', lr=1e-3,
      epochs=10, save_path='models/part1_best_model.pt')
```

```python
# Part III: CNN evaluation
from cnn import ConvNet
cnn = ConvNet(num_classes=36).to(device)
cnn.load_state_dict(torch.load('models/part3_cnn_weights.pt'))
test_acc, test_metrics = evaluate(cnn, test_loader)
print(f"Test Accuracy: {test_acc:.2%}")
```

## 🛠 Tech Stack

- **Language**: Python 3.8+  
- **Framework**: PyTorch  
- **Data & Preprocessing**: pandas, NumPy, scikit-learn  
- **Visualization**: Matplotlib, Seaborn, Torchinfo  
- **Experiment Tracking**: Pickle, torch.save/load  
- **Notebook**: Jupyter  

## 🌱 Future Enhancements

- Automated hyperparameter search (Optuna, Ray Tune)  
- Mini-batch & distributed training for large datasets  
- Grad-CAM visualization for CNN interpretability  
- Ensemble of CNN + VGG + ResNet for marginal gains  
- CI/CD pipeline (GitHub Actions) for reproducible training  

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch  
```bash
git checkout -b feature/awesome-enhancement
```
3. Commit your changes  
```bash
git commit -m "Add awesome feature"
```
4. Push & open a Pull Request  

Feedback and improvements are always welcome!

## 📜 License

This project is licensed under the MIT License. See LICENSE for details.

## 👥 Authors & Contact

**Vinay Trinadh Naraharisetty** 
[GitHub](https://github.com/VinayTrinadh7755)  
[LinkedIn](www.linkedin.com/in/vinay-trinadh-naraharisetty)

Thank you for exploring our deep learning implementations!