import torch
import numpy as np
from pinn_model import PINN, data_loss, physics_loss, initial_condition_loss
from pinn_utils import true_solution, make_collocation_points, plot_predictions, plot_losses

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
    'noise_level': 0.75,
    'n_hidden': 20,
    'lr': 0.01,
    'num_epochs': 2500,
    'lambda_data': 3.,
    'lambda_ode': 1.5,
    'lambda_ic': 1.0,
    'plot_every': 200
}

device = torch.device(CONFIG['device'])
print(f"💡 Using device: {device}")

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
# Generate Collocation Points
# ------------------------------------------
t_phys_tensor = make_collocation_points(CONFIG['t_min'], CONFIG['t_max'], CONFIG['N_phys'], device)

# ------------------------------------------
# Initialize Model
# ------------------------------------------
model = PINN(CONFIG['n_hidden']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

# ------------------------------------------
# Training Loop
# ------------------------------------------
history = {'total': [], 'data': [], 'ode': [], 'ic': []}

for epoch in range(CONFIG['num_epochs']):
    optimizer.zero_grad()
    l_data = data_loss(model, t_data_tensor, h_data_tensor)
    l_ode  = physics_loss(model, t_phys_tensor, CONFIG['v0'], CONFIG['g'])
    l_ic   = initial_condition_loss(model, CONFIG['h0'], device)

    loss = CONFIG['lambda_data'] * l_data + CONFIG['lambda_ode'] * l_ode + CONFIG['lambda_ic'] * l_ic
    loss.backward()
    optimizer.step()

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

plot_predictions(t_data, h_data_noisy, t_plot, h_true_plot, h_pred_plot)
plot_losses(history)
