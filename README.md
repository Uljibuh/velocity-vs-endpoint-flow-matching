# velocity-vs-endpoint-flow-matching


# Flow Matching: v-Prediction vs x-Prediction

A minimal PyTorch implementation and analysis of:

- v-prediction (velocity prediction)
- x-prediction (endpoint prediction)

using:
- linear interpolation flow matching
- Euler ODE sampling
- MNIST trajectory transport
- U-Net architectures

This repository explores the geometric and dynamical differences between:

\[
v_\theta(x_t, t)
\]

and

\[
x_{1,\theta}(x_t, t)
\]

including:
- trajectory behavior
- endpoint bending
- instability near \(t \to 1\)
- effects of clamping
- why U-Net significantly outperforms shallow CNNs

---

# Motivation

Flow matching and diffusion models can parameterize the transport process in different ways.

Two common formulations are:

## v-prediction

Predict the instantaneous velocity field directly.

## x-prediction

Predict the clean endpoint and derive velocity afterward.

Although mathematically related, they produce very different trajectory dynamics during sampling.

This repository provides:
- a clean implementation
- derivations
- visual trajectory comparisons
- practical observations

---

# Interpolation Path

We define a linear interpolation between source image \(x_0\) and target image \(x_1\):

\[
x_t = (1 - t)x_0 + tx_1
\]

where:

- \(x_0\): source image
- \(x_1\): target image
- \(t \in [0,1]\)

---

# v-Prediction

## Objective

The true velocity of the interpolation path is:

\[
\frac{dx_t}{dt} = x_1 - x_0
\]

Thus the model learns:

\[
v_\theta(x_t, t) \approx x_1 - x_0
\]

Loss:

\[
\mathcal{L}_v =
\|v_\theta(x_t,t) - (x_1-x_0)\|^2
\]

---

# v-Mode Sampling

Sampling directly integrates the learned velocity field:

\[
x_{t+\Delta t}
=
x_t + v_\theta(x_t,t)\Delta t
\]

using Euler integration.

This formulation is stable because:
- no singularity exists near \(t=1\)
- velocity is learned directly
- trajectories are smoother

---

# x-Prediction

## Objective

Instead of predicting velocity, the model predicts the endpoint:

\[
x_{1,\theta}(x_t,t) \approx x_1
\]

Loss:

\[
\mathcal{L}_x =
\|x_{1,\theta}(x_t,t)-x_1\|^2
\]

---

# Deriving Velocity from x-Prediction

From the interpolation equation:

\[
x_t = (1-t)x_0 + tx_1
\]

we can derive:

\[
x_1 - x_t
=
(1-t)(x_1-x_0)
\]

thus:

\[
x_1 - x_0
=
\frac{x_1-x_t}{1-t}
\]

Since:

\[
\frac{dx_t}{dt} = x_1-x_0
\]

we obtain:

\[
v(x_t,t)
=
\frac{x_1-x_t}{1-t}
\]

Therefore during sampling:

\[
v_\theta(x_t,t)
=
\frac{x_{1,\theta}(x_t,t)-x_t}{1-t}
\]

---

# x-Mode Sampling

Sampling proceeds as:

1. predict endpoint

\[
\hat{x}_1 = x_{1,\theta}(x_t,t)
\]

2. derive velocity

\[
v =
\frac{\hat{x}_1 - x_t}{1-t}
\]

3. Euler update

\[
x_{t+\Delta t}
=
x_t + v\Delta t
\]

---

# Why Trajectories Bend Near \(t=1\)

In x-prediction:

\[
v =
\frac{\hat{x}_1-x_t}{1-t}
\]

As:

\[
t \to 1
\]

the denominator approaches zero.

Even tiny endpoint prediction errors become amplified into very large velocities.

This causes:
- sharp late-stage corrections
- elbow-shaped trajectories
- instability near the endpoint
- trajectory bending

This phenomenon remains visible even when clamping is applied.

---

# Why Clamping Is Necessary

Without clamping:

\[
\frac{1}{1-t}
\]

can explode numerically near \(t=1\).

We therefore use:

```python
denom = torch.clamp(1 - t, min=1e-3)
```

This prevents:
- infinite velocities
- NaNs
- exploding trajectories
- unstable sampling

However, clamping also changes the true dynamics of the ODE and introduces bias near the endpoint.

Thus:
- clamping stabilizes sampling
- but does not completely remove trajectory bending

---

# Why U-Net Works Better Than Simple CNNs

A simple CNN struggles because:
- flow matching requires multi-scale spatial reasoning
- digit transport requires preserving global structure
- endpoint prediction requires understanding long-range spatial relationships

U-Net improves performance through:
- hierarchical feature extraction
- encoder-decoder structure
- skip connections
- better spatial reconstruction
- global + local feature integration

Empirically:
- shallow CNNs produce blurry or unstable trajectories
- U-Nets generate smoother and more coherent transport paths

---

# Observations

## v-prediction
- smoother trajectories
- more stable dynamics
- straighter transport paths
- physically meaningful vector fields

## x-prediction
- stronger endpoint supervision
- sharper endpoint reconstruction
- trajectory bending near \(t=1\)
- instability caused by velocity reconstruction

---

# Repository Structure

```bash
.
├── train.py
├── models.py
├── sample.py
├── README.md
└── assets/
```

---

# Installation

```bash
git clone <repo_url>
cd <repo_name>

pip install torch torchvision matplotlib
```

---

# Run Training

```bash
python train.py
```

---

# References

## JiT

- GitHub:
  https://github.com/LTH14/JiT

## Paper

### Back to Basics: Let Denoising Generative Models Denoise

https://arxiv.org/abs/2511.13720
