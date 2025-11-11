# Detecting Data Poisoning and Adversarial Attacks in CNN-Based Image Classifiers

**Authors:** Muhammad Osama 
**Course:** INFO-6149 – Machine Learning Security  
**Institution:** Fanshawe College  
**Submission:** July 2025  
**LinkedIn:** [muhammad-osama-872328202](https://www.linkedin.com/in/muhammad-osama-872328202)  
**GitHub:** [MuhammadOsama380](https://github.com/MuhammadOsama380)

---

## 🧠 Project Overview
This project explores how **Convolutional Neural Networks (CNNs)** can be compromised through both **data poisoning** (training-time) and **adversarial** (inference-time) attacks, and implements defenses to restore robustness.  
We use the **CIFAR-10 dataset** to demonstrate real-world vulnerabilities and mitigation techniques in deep learning image classifiers.

---

## ⚔️ Objectives
- Simulate **label-flipping data poisoning** on CNN training data.  
- Launch **white-box PGD** (Projected Gradient Descent) adversarial attacks using the **SECML** framework.  
- Implement **three defense techniques**:  
  1. Label sanitization  
  2. Dropout regularization  
  3. Randomized smoothing  
- Evaluate and visualize model robustness via **security curves**, **confusion matrices**, and **quantitative metrics**.

---

## 📚 Dataset
- **Name:** CIFAR-10  
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
- **Training samples:** 50 000  
- **Testing samples:** 10 000  
- **Normalization:** Mean = (0.4914, 0.4822, 0.4465); Std = (0.247, 0.243, 0.261)  
- **Poisoning fractions tested:** 0 %, 5 %, 10 %, 15 %

---

## 🧩 Phase 1 – Baseline Model
### CNN Architecture
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
```
- Optimizer: **Adam (lr = 0.001)**  
- Loss: **Cross-Entropy**  
- Early-stopping used for stability  

### Baseline Accuracy  
- Clean Test Accuracy ≈ **62.7 %**

---

## ☠️ Phase 2 – Training-Time Attack (Data Poisoning)
### Label-Flipping Method
- 15 % of labels randomly flipped among different classes  
- Retrained model on poisoned set  

### Observations
- Accuracy dropped sharply as poisoned samples increased  
- At 15 % poisoning → ~20 % loss in test accuracy  

---

## 💥 Phase 3 – Inference-Time Attack (Adversarial)
### White-Box PGD Attack
| Parameter | Value |
|------------|--------|
| Norm | L2 |
| dmax | 0.1 |
| η (step size) | 0.025 |
| max iter | 50 |
| ε | 1e-4 |
| Attack Type | Untargeted |

Generated adversarial samples fooled the CNN with imperceptible noise.  
A clear **confidence drop** appeared between clean and adversarial predictions.

---

## 🛡️ Phase 4 – Defense Implementation
### 1️⃣ Label Sanitization  
- Used **Isolation Forest / Nearest-Neighbor** detection  
- Removed outliers before retraining  

### 2️⃣ Dropout Regularization  
- Added **Dropout (p = 0.5)** after `fc1` layer  
- Reduced overfitting and improved generalization  

### 3️⃣ Randomized Smoothing  
- Added **Gaussian noise (σ = 0.1)** to inputs  
- Averaged predictions across 50 noisy copies  
- Stabilized predictions under perturbation  

---

## 📊 Phase 5 – Evaluation & Results

### Quantitative Metrics
| Metric | Base Model | Defended Model |
|---------|-------------|----------------|
| Accuracy (Clean) | 62.75 % | 87.13 % |
| Accuracy (Adversarial) | 80.00 % | 90.00 % |
| Inference Time (Clean) | 6.44 s | 3.28 s |
| Inference Time (Adv) | 0.01 s | 0.00 s |
| Attack Success Reduction | — | 10 % |
| Clean Accuracy Preservation | — | 24.38 % |

### Visual Analyses
- **Security Curves:**  
  - Accuracy ↓ with higher poison fraction  
  - Accuracy ↓ with increasing PGD perturbation (dmax)  
- **Confidence Histogram:**  
  - Adversarial samples show lower softmax confidence  
- **Per-Class Vulnerability:**  
  - *Cat* and *dog* most susceptible  
  - *Ship* and *airplane* most robust  
- **Failure Modes:**  
  - Misclassifications concentrated near weak boundaries (e.g., cat–dog)  

---

## 🧮 Phase 6 – Defense Evaluation (Advanced)
- **Gradient Sensitivity Analysis:** Defended models show lower input-gradient norms.  
- **Decision Boundary Visualization (PCA):** Dropout model exhibited smoother, wider margins.  
- **Adversarial Training:** Further reduced attack transferability from surrogate models.  

---

## 🧾 Key Findings
- CNNs are highly vulnerable to both training-time and inference-time attacks.  
- Dropout and Randomized Smoothing significantly improve robustness.  
- Label sanitization helps remove poison outliers effectively.  
- Combining these defenses reduces attack success rate by ~10 % and boosts clean accuracy by ~24 %.  

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/MuhammadOsama380/Detecting-Data-Poisoning-and-Adversarial-Attacks.git
cd Detecting-Data-Poisoning-and-Adversarial-Attacks

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook "MLS Code.ipynb"
```

---

## 🧰 Tools & Frameworks
- Python 3.x  
- PyTorch / TorchVision  
- SECML (v0.15)  
- NumPy / Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  

---

## 📂 Repository Structure
```
Detecting-Data-Poisoning-and-Adversarial-Attacks/
│
├── data/
│   └── CIFAR10/
│
├── notebooks/
│   └── MLS Code.ipynb
│
├── reports/
│   └── MLS Project Report.pdf
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 🪪 License
This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements
Developed by **Muhammad Osama (1288056)**, **Shah Fahad**, and **Zaid Khan**  
for the course **INFO-6149 Machine Learning Security** at **Fanshawe College (2025)**.  
If you found this project helpful, please ⭐ the repository or connect on [LinkedIn](https://www.linkedin.com/in/muhammad-osama-872328202).
