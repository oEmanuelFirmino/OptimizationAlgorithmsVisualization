# Machine Learning Optimization Visualization

This project implements and visualizes different optimization algorithms commonly used in machine learning, showing the trajectory of weights during model training.

## 📚 Mathematical Background

### Loss Functions

1. **Quadratic Loss**
   
   Simple quadratic function useful for testing optimization algorithms:
   ```math
   L(w) = w_1^2 + 2w_2^2
   ```

2. **Rastrigin Function**
   
   Non-convex function with many local minima:
   ```math
   L(w) = 10n + \sum_{i=1}^n [w_i^2 - 10\cos(2\pi w_i)]
   ```

3. **Rosenbrock Function**
   
   Classic optimization problem (banana function):
   ```math
   L(w) = 100(w_2 - w_1^2)^2 + (1 - w_1)^2
   ```

### Optimization Algorithms

1. **Stochastic Gradient Descent (SGD)**
   
   Basic update rule with momentum:
   ```math
   v_t = \gamma v_{t-1} + \eta \nabla L(w_t)
   ```
   ```math
   w_{t+1} = w_t - v_t
   ```
   where:
   - $\gamma$ is the momentum coefficient
   - $\eta$ is the learning rate
   - $\nabla L(w_t)$ is the gradient

2. **Adam (Adaptive Moment Estimation)**
   
   Combines momentum and RMSprop:
   ```math
   m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L(w_t)
   ```
   ```math
   v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L(w_t))^2
   ```
   ```math
   \hat{m}_t = \frac{m_t}{1-\beta_1^t}
   ```
   ```math
   \hat{v}_t = \frac{v_t}{1-\beta_2^t}
   ```
   ```math
   w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t
   ```
   where:
   - $\beta_1, \beta_2$ are decay rates
   - $\epsilon$ is a small constant for numerical stability

## 🔧 Implementation Details

### Learning Rate Scheduling
We implement a learning rate schedule that combines warmup and decay:
```math
\eta_t = \begin{cases}
\eta_{\text{base}} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t < t_{\text{warmup}} \\
\min(\eta_{\text{base}}, \eta_{\text{max}}) \cdot \alpha^{\frac{t}{t_{\text{decay}}}} & \text{otherwise}
\end{cases}
```

### Regularization
L2 regularization is added to prevent overfitting:
```math
L_{\text{reg}}(w) = L(w) + \lambda\|w\|_2^2
```

## 🚀 Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the interactive visualization:
```bash
streamlit run notebooks/src/interface.py
```

## 📈 Features

- Interactive visualization of optimization trajectories
- Multiple loss functions to experiment with
- Configurable hyperparameters
- Real-time weight updates visualization
- Comparison between different optimizers

## 📊 Project Structure

```
├── notebooks/
│   ├── src/
│   │   ├── optimizers.py      # Optimization algorithms
│   │   ├── loss_functions.py  # Loss function implementations
│   │   ├── visualizers.py     # Visualization utilities
│   │   ├── interface.py       # Streamlit interface
│   │   └── SimpleNeuralNet.py # Neural network implementation
│   └── optimization.ipynb     # Example notebook
├── requirements.txt
└── README.md
```
