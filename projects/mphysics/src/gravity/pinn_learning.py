import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# Configuration Parameters
# ------------------------------------------
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'g': 9.8,
    'h0': 1.0,
    'v0': 10.0,
    't_min': 0.0,
    't_max': 2.0,
    'N_data': 20,
    'N_phys': 20,
    'noise_level': 1.5,
    'n_hidden': 20,
    'lr': 0.01,
    'num_epochs': 2000,
    'lambda_data': 3.0,
    'lambda_ode': 1,
    'lambda_ic': 2.0,
    'plot_every': 200
}

device = torch.device(CONFIG['device'])
print(f"💡 Using device: {device}")

# ------------------------------------------
# Ground Truth Physics Function
# ------------------------------------------
def true_solution(t, h0, v0, g):
    return h0 + v0 * t - 0.5 * g * t**2

# ------------------------------------------
# Generate Data Samples (Noisy Measurements)
# ------------------------------------------
np.random.seed(0)
t_data = np.linspace(CONFIG['t_min'], CONFIG['t_max'], CONFIG['N_data'])
h_data = true_solution(t_data, CONFIG['h0'], CONFIG['v0'], CONFIG['g'])
h_data_noisy = h_data + CONFIG['noise_level'] * np.random.randn(CONFIG['N_data'])

t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1).to(device)
h_data_tensor = torch.tensor(h_data_noisy, dtype=torch.float32).view(-1, 1).to(device)

# ------------------------------------------
# Generate Collocation Points for Physics Loss
# ------------------------------------------
t_phys = np.linspace(CONFIG['t_min'], CONFIG['t_max'], CONFIG['N_phys']).reshape(-1, 1).astype(np.float32)
t_phys_tensor = torch.tensor(t_phys, requires_grad=True).to(device)

# ------------------------------------------
# Define the Neural Network Model
# ------------------------------------------
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

model = PINN(CONFIG['n_hidden']).to(device)

# ------------------------------------------
# Autograd Helper
# ------------------------------------------
def derivative(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

# ------------------------------------------
# Loss Functions
# ------------------------------------------
def data_loss(model, t_data, h_data):
    h_pred = model(t_data)
    return torch.mean((h_pred - h_data)**2)

def physics_loss(model, t_phys):
    t_phys.requires_grad_(True)
    h_pred = model(t_phys)
    dh_dt_pred = derivative(h_pred, t_phys)
    dh_dt_true = CONFIG['v0'] - CONFIG['g'] * t_phys
    return torch.mean((dh_dt_pred - dh_dt_true)**2)

def initial_condition_loss(model):
    t0 = torch.zeros(1, 1, dtype=torch.float32).to(device)
    h0_pred = model(t0)
    return (h0_pred - CONFIG['h0']).pow(2).mean()

# ------------------------------------------
# Training Loop
# ------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
history = {'total': [], 'data': [], 'ode': [], 'ic': []}

for epoch in range(CONFIG['num_epochs']):
    optimizer.zero_grad()
    l_data = data_loss(model, t_data_tensor, h_data_tensor)
    l_ode  = physics_loss(model, t_phys_tensor)
    l_ic   = initial_condition_loss(model)
    loss = CONFIG['lambda_data'] * l_data + CONFIG['lambda_ode'] * l_ode + CONFIG['lambda_ic'] * l_ic
    loss.backward()
    optimizer.step()

    # Logging
    history['total'].append(loss.item())
    history['data'].append(l_data.item())
    history['ode'].append(l_ode.item())
    history['ic'].append(l_ic.item())

    if (epoch + 1) % CONFIG['plot_every'] == 0:
        print(f"Epoch {epoch+1}: Total={loss.item():.4f}, Data={l_data.item():.4f}, ODE={l_ode.item():.4f}, IC={l_ic.item():.4f}")

# ------------------------------------------
# Evaluation and Plotting
# ------------------------------------------
t_plot = np.linspace(CONFIG['t_min'], CONFIG['t_max'], 200).reshape(-1, 1).astype(np.float32)
t_plot_tensor = torch.tensor(t_plot, requires_grad=True).to(device)
h_pred_plot = model(t_plot_tensor).detach().cpu().numpy()
h_true_plot = true_solution(t_plot, CONFIG['h0'], CONFIG['v0'], CONFIG['g'])

# Plot predictions
plt.figure(figsize=(10, 5))
plt.scatter(t_data, h_data_noisy, color='red', label='Noisy Data')
plt.plot(t_plot, h_true_plot, 'k--', label=r'$h(t) = h_0 + v_0 t - \frac{1}{2}gt^2$', linewidth=2)
plt.plot(t_plot, h_pred_plot, 'b', label='PINN Prediction', linewidth=2)
plt.xlabel('Time $t$')
plt.ylabel('Height $h(t)$')
plt.title('PINN Prediction vs True Physics')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss curves
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
