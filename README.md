Flow Matching: v-Prediction vs x-Prediction

A modular, efficient PyTorch implementation and analysis comparing:
v-prediction (velocity prediction)
x-prediction (endpoint prediction)

Features
Modular Architecture: Clean separation of data, models, and sampling logic.

Linear Interpolation Flow Matching: Simple, interpretable transport paths.

Euler ODE Sampling: Step-by-step integration from source to target.

Lightweight U-Net

MNIST Trajectory Transport: Visualizing how the model "learns to move" pixels.

This repository explores the geometric and dynamical differences between:


$$v_\theta(x_t, t)$$


and


$$x_{1,\theta}(x_t, t)$$

Repository Structure

The project is organized into modular components to allow for easy experimentation with different architectures or datasets:
.
├── data.py           # MNIST loading, filtering, and preprocessing
├── models.py         # U-Net architecture and Time-Injection blocks
├── sample.py         # ODE integration logic and matplotlib visualization
├── train.py          # Training loop, loss functions, and entry point
├── requirements.txt  # Project dependencies
├── .gitignore        # Prevents tracking data/ and __pycache__/
└── README.md


Theoretical Background

Interpolation Path
We define a linear interpolation between source image $x_0$ and target image $x_1$:

$$x_t = (1 - t)x_0 + tx_1$$

where $t \in [0,1]$.

v-Prediction

The true velocity of the interpolation path is $\frac{dx_t}{dt} = x_1 - x_0$. 

The model learns to predict this velocity directly:$$\mathcal{L}_v = \|v_\theta(x_t,t) - (x_1-x_0)\|^2$$Sampling: $x_{t+\Delta t} = x_t + v_\theta(x_t,t)\Delta t$. This is generally stable as no singularity exists near $t=1$.x-PredictionThe model predicts the clean endpoint $x_1$ directly:

$$\mathcal{L}_x = \|x_{1,\theta}(x_t,t)-x_1\|^2$$

Velocity Derivation: To sample, we must derive velocity from the predicted endpoint:
$$v_\theta(x_t,t) = \frac{x_{1,\theta}(x_t,t)-x_t}{1-t}$$

Observations: The "Elbow" Effect

In x-prediction, as $t \to 1$, the denominator $(1-t)$ approaches zero. Even tiny endpoint prediction errors become amplified, causing:

Trajectory Bending: Sharp late-stage corrections (elbow-shaped paths).

Instability: High sensitivity to noise near the target.

Clamping: We use torch.clamp(1 - t, min=1e-3) to prevent NaNs, though this introduces a slight bias in the final transport steps.

Why U-Net?Unlike shallow CNNs, the U-Net architecture significantly outperforms in flow matching due to:

Hierarchical Features: Capturing both global digit structure and local pixel details.

Skip Connections: Directly passing high-resolution information to the decoder, crucial for maintaining digit identity during transport.

Time Injection: Efficiently conditioning the model on the scalar $t$ through Linear-SiLU layers.

Installation & Usage

1. Clone and SetupSince this project is optimized for GitHub Codespaces, you can get started immediately in the cloud
   :Bash

# Clone the repository
gh repo clone Uljibuh/velocity-vs-endpoint-flow-matching
cd velocity-vs-endpoint-flow-matching

# Install dependencies
pip install -r requirements.txt
2. Run TrainingThe 
train.py script will train both models sequentially and generate comparative plots:
Bash
python3 train.py


References & Inspiration
Paper: Back to Basics: Let Denoising Generative Models Denoise

Code Inspiration: JiT (Just-in-Time) Flow Matching
