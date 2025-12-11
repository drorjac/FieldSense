# pinn_utils.py

import numpy as np
import matplotlib.pyplot as plt
import torch

def true_solution(t, h0, v0, g):
    return h0 + v0 * t - 0.5 * g * t**2

def make_collocation_points(t_min, t_max, N_phys, device, requires_grad=True):
    t = np.linspace(t_min, t_max, N_phys).reshape(-1, 1).astype(np.float32)
    return torch.tensor(t, requires_grad=requires_grad).to(device)

def plot_predictions(t_data, h_data_noisy, t_plot, h_true, h_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(t_data, h_data_noisy, color='red', label='Noisy Data')
    plt.plot(t_plot, h_true, 'k--', label=r'$h(t) = h_0 + v_0 t - \frac{1}{2}gt^2$', linewidth=2)
    plt.plot(t_plot, h_pred, 'b', label='PINN Prediction', linewidth=2)
    plt.xlabel('Time $t$')
    plt.ylabel('Height $h(t)$')
    plt.title('PINN Prediction vs True Physics')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_losses(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['total'], label='Total Loss')
    plt.plot(history['data'], label='Data Loss')
    plt.plot(history['ode'], label='ODE Loss')
    plt.plot(history['ic'], label='Initial Condition Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves During Training')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
