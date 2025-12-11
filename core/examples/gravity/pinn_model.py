# pinn_model.py

import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, n_hidden):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, t):
        return self.net(t)

def derivative(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def data_loss(model, t_data, h_data):
    h_pred = model(t_data)
    return torch.mean((h_pred - h_data)**2)

def physics_loss(model, t_phys, v0, g):
    t_phys.requires_grad_(True)
    h_pred = model(t_phys)
    dh_dt_pred = derivative(h_pred, t_phys)
    dh_dt_true = v0 - g * t_phys
    return torch.mean((dh_dt_pred - dh_dt_true)**2)

def initial_condition_loss(model, h0, device):
    t0 = torch.zeros(1, 1, dtype=torch.float32).to(device)
    h0_pred = model(t0)
    return (h0_pred - h0).pow(2).mean()
