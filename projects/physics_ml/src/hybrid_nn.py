"""
Hybrid Rain Retrieval Neural Network
Combines physics-based and data-driven approaches with dynamic fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rain_simulator import get_k_alpha


class PhysicsBranch(nn.Module):
    """Model-based branch using ITU-R power law inversion"""

    def __init__(self, k_init: float, alpha_init: float, learn_baseline: bool = False):
        super().__init__()
        # Parameters in log space to ensure positivity
        self.log_k = nn.Parameter(torch.log(torch.tensor(k_init, dtype=torch.float32)))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init, dtype=torch.float32)))

        # Optional learnable baseline
        self.learn_baseline = learn_baseline
        if learn_baseline:
            self.baseline = nn.Parameter(torch.zeros(1))

    def forward(self, x_atten: torch.Tensor, link_length: float = 1.0, baseline: float = 0.0):
        k = torch.exp(self.log_k)
        alpha = torch.exp(self.log_alpha)

        if self.learn_baseline:
            baseline = self.baseline.item()

        # Rain attenuation = total - baseline
        rain_atten = F.softplus(x_atten - baseline)

        # Invert power law: R = (A/(k*L))^(1/alpha)
        rain_rate = torch.pow(rain_atten / (k * link_length + 1e-8), 1.0 / alpha)

        return F.softplus(rain_rate)  # Ensure positive

    @property
    def coefficients(self):
        """Get current k, alpha values"""
        return torch.exp(self.log_k).item(), torch.exp(self.log_alpha).item()


class NeuralBranch(nn.Module):
    """Data-driven GRU branch for sequence processing"""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq: [batch, seq_len, input_size] attenuation sequence
        Returns:
            rain_rate: [batch] predicted rain rates
        """
        gru_out, _ = self.gru(x_seq)  # [batch, seq_len, hidden]

        # Use last timestep
        last_hidden = gru_out[:, -1, :]  # [batch, hidden]
        last_hidden = self.dropout(last_hidden)

        rain_rate = self.output_layer(last_hidden).squeeze(-1)  # [batch]
        return F.relu(rain_rate)  # Ensure non-negative


class FusionGate(nn.Module):
    """Dynamic merger gate implementing paper's feature vector"""

    def __init__(self, feature_dim: int = 6):
        super().__init__()
        # Single linear layer + sigmoid as per paper
        self.gate_layer = nn.Linear(feature_dim, 1)

    def _compute_features(self, x_current, r_physics, r_neural, x_seq=None, eps=1e-6):
        """Compute 6-dimensional feature vector from paper"""

        # Normalized current input (z-score)
        x_mean = x_current.mean() if x_current.numel() > 1 else x_current
        x_std = x_current.std() if x_current.numel() > 1 else torch.tensor(1.0, device=x_current.device)
        x_tilde = (x_current - x_mean) / (x_std + eps)

        # Local statistics from sequence
        if x_seq is not None and x_seq.size(1) > 1:
            # Use last 6 timesteps for local stats
            window_size = min(6, x_seq.size(1))
            recent = x_seq[:, -window_size:, 0]  # [batch, window_size]
            mu_x = recent.mean(dim=1)
            sigma_x = recent.std(dim=1) + eps
        else:
            # Fallback
            mu_x = x_current
            sigma_x = torch.ones_like(x_current) * eps

        # Log-compressed branch predictions
        log_r_physics = torch.log1p(r_physics)
        log_r_neural = torch.log1p(r_neural)

        # Disagreement signal
        r_diff = r_physics - r_neural

        # Stack features: [x_tilde, sigma_x, mu_x, log(1+r_M), log(1+r_D), r_M - r_D]
        features = torch.stack([
            x_tilde, sigma_x, mu_x,
            log_r_physics, log_r_neural, r_diff
        ], dim=1)  # [batch, 6]

        return features

    def forward(self, x_current, r_physics, r_neural, x_seq=None):
        """
        Returns:
            gate_weight: g_t ∈ (0,1)
            r_fused: Convex combination of branches
        """
        features = self._compute_features(x_current, r_physics, r_neural, x_seq)

        gate_logit = self.gate_layer(features).squeeze(-1)  # [batch]
        gate_weight = torch.sigmoid(gate_logit)  # g_t ∈ (0,1)

        # Convex combination: r = g*r_physics + (1-g)*r_neural
        r_fused = gate_weight * r_physics + (1 - gate_weight) * r_neural

        return gate_weight, r_fused


class HybridRainModel(nn.Module):
    """Complete hybrid model: Physics + Neural + Fusion"""

    def __init__(self, k_init: float, alpha_init: float, hidden_size: int = 64,
                 learn_baseline: bool = False, lambda_physics: float = 0.05,
                 lambda_neural: float = 0.05, lambda_reg: float = 1e-5):
        super().__init__()

        # Three branches
        self.physics = PhysicsBranch(k_init, alpha_init, learn_baseline)
        self.neural = NeuralBranch(hidden_size=hidden_size)
        self.fusion = FusionGate()

        # Loss weights
        self.lambda_physics = lambda_physics
        self.lambda_neural = lambda_neural
        self.lambda_reg = lambda_reg

    def forward(self, x_seq: torch.Tensor, link_length: float = 1.0, baseline: float = 0.0):
        """
        Args:
            x_seq: [batch, seq_len, 1] attenuation sequences
            link_length: Link length in km
            baseline: Baseline attenuation in dB
        """
        # Current attenuation (last timestep)
        x_current = x_seq[:, -1, 0]  # [batch]

        # Branch predictions
        r_physics = self.physics(x_current, link_length, baseline)
        r_neural = self.neural(x_seq)

        # Fusion
        gate_weight, r_fused = self.fusion(x_current, r_physics, r_neural, x_seq)

        return {
            'rain_rate': r_fused,
            'rain_physics': r_physics,
            'rain_neural': r_neural,
            'gate_weight': gate_weight
        }

    def compute_loss(self, predictions, targets):
        """Composite loss from paper"""
        r_fused = predictions['rain_rate']
        r_physics = predictions['rain_physics']
        r_neural = predictions['rain_neural']

        # Primary loss (Huber for robustness)
        loss_fused = F.huber_loss(r_fused, targets, delta=1.0)

        # Auxiliary losses (MSE)
        loss_physics = F.mse_loss(r_physics, targets)
        loss_neural = F.mse_loss(r_neural, targets)

        # Regularization
        loss_reg = torch.tensor(0.0, device=targets.device)
        for param in self.parameters():
            loss_reg += torch.norm(param, 2)

        # Composite loss
        total_loss = (loss_fused +
                     self.lambda_physics * loss_physics +
                     self.lambda_neural * loss_neural +
                     self.lambda_reg * loss_reg)

        return {
            'total': total_loss,
            'fused': loss_fused,
            'physics': loss_physics,
            'neural': loss_neural,
            'reg': loss_reg
        }


def create_hybrid_model(freq_ghz: float, pol: str = "vertical", **kwargs):
    """Factory function using ITU-R coefficients"""
    k_init, alpha_init = get_k_alpha(freq_ghz, pol)
    return HybridRainModel(k_init=k_init, alpha_init=alpha_init, **kwargs)


def prepare_sequences(R, A, seq_len=48):
    """Convert R,A pairs to sequences for neural branch"""
    n_samples = len(R)
    sequences = []
    targets = []

    for i in range(n_samples):
        # Create sequence with variations around current attenuation
        base_atten = A[i]
        seq = np.random.normal(base_atten, 0.15, seq_len)  # Small variations
        seq[-1] = base_atten  # Ensure last value matches target

        sequences.append(seq)
        targets.append(R[i])

    # Convert to tensors
    X = torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(-1)  # [n, seq_len, 1]
    y = torch.tensor(np.array(targets), dtype=torch.float32)

    return X, y


class SimpleTrainer:
    """Simple trainer for hybrid model"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = []

    def train_epoch(self, X_train, y_train, optimizer, batch_size=32):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle data
        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size].to(self.device)
            batch_y = y_train[i:i+batch_size].to(self.device)

            optimizer.zero_grad()

            outputs = self.model(batch_x, link_length=1.5, baseline=2.0)
            losses = self.model.compute_loss(outputs, batch_y)

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += losses['total'].item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, X_val, y_val, batch_size=32):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        gate_weights = []

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_x = X_val[i:i+batch_size].to(self.device)
                batch_y = y_val[i:i+batch_size].to(self.device)

                outputs = self.model(batch_x, link_length=1.5, baseline=2.0)
                losses = self.model.compute_loss(outputs, batch_y)

                total_loss += losses['total'].item()
                gate_weights.extend(outputs['gate_weight'].cpu().numpy())
                n_batches += 1

        return total_loss / n_batches, np.mean(gate_weights)

    def train(self, dataset, epochs=20, lr=1e-3, batch_size=32):
        """Train model on dataset"""

        # Prepare data
        X_train, y_train = prepare_sequences(dataset['train']['R'], dataset['train']['A'])
        X_val, y_val = prepare_sequences(dataset['val']['R'], dataset['val']['A'])

        print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_train, optimizer, batch_size)
            val_loss, gate_usage = self.validate(X_val, y_val, batch_size)

            scheduler.step(val_loss)

            # Track history
            k_current, alpha_current = self.model.physics.coefficients
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'gate_usage': gate_usage,
                'k': k_current,
                'alpha': alpha_current
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                      f"Gate={gate_usage:.3f}, k={k_current:.2e}, α={alpha_current:.3f}")

        return best_val_loss


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_hybrid_model(freq_ghz=24, pol="vertical", hidden_size=64)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Physics coefficients: k={model.physics.coefficients[0]:.2e}, α={model.physics.coefficients[1]:.3f}")

    # Test forward pass with dummy data
    batch_size, seq_len = 32, 48
    x_dummy = torch.randn(batch_size, seq_len, 1) * 2 + 10  # Dummy attenuation sequences
    y_dummy = torch.rand(batch_size) * 10 + 0.5  # Dummy rain rates (0.5-10.5 mm/h)

    with torch.no_grad():
        outputs = model(x_dummy, link_length=1.5, baseline=2.0)
        losses = model.compute_loss(outputs, y_dummy)

    print("\nForward pass successful:")
    for key, value in outputs.items():
        print(f"  {key}: shape={value.shape}, mean={value.mean():.3f}")

    print(f"\nLoss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print(f"\nGate usage: {outputs['gate_weight'].mean():.3f} (0=neural, 1=physics)")