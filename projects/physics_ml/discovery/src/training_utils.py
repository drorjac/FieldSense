"""
Training utilities for hybrid rain retrieval model
Includes GPU detection and training functions
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from hybrid_nn import create_hybrid_model, SimpleTrainer


def get_device():
    """Get the best available device (GPU > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def train_models(datasets: Dict[str, Dict[str, Any]], config) -> Dict[str, Any]:
    """Train hybrid models on all datasets with GPU support"""

    # Get device for training
    device = get_device()

    results = {}
    total_models = len(datasets)
    model_count = 0

    print(f"\nTraining hybrid models on {device}...")
    print("=" * 80)

    for dataset_key, dataset in datasets.items():
        model_count += 1
        freq = dataset['freq']
        sigma = dataset['sigma']

        print(f"\n[{model_count:2d}/{total_models}] Training model for {freq} GHz, σ={sigma} dB")
        print("-" * 60)

        # Create model
        model = create_hybrid_model(
            freq_ghz=freq,
            pol="vertical",
            hidden_size=config.hidden_size,
            learn_baseline=False
        )

        # Initialize trainer with GPU support
        trainer = SimpleTrainer(model, device=device)

        # Train model
        best_val_loss = trainer.train(
            dataset=dataset,
            epochs=config.epochs,
            lr=config.learning_rate,
            batch_size=config.batch_size
        )

        # Store results
        results[dataset_key] = {
            'model': model,
            'trainer': trainer,
            'best_val_loss': best_val_loss,
            'history': trainer.history,
            'final_coefficients': model.physics.coefficients,
            'device': str(device)
        }

        k_final, alpha_final = model.physics.coefficients
        print(f"Final: Val Loss={best_val_loss:.4f}, k={k_final:.2e}, α={alpha_final:.3f}")

    return results


def plot_training_curves(results: Dict[str, Any], save_path: str = None):
    """Plot training curves for all models"""

    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Progress Across All Models", fontsize=16, fontweight='bold')

    # Collect data from all models
    all_train_losses = []
    all_val_losses = []
    all_gate_usage = []
    all_k_evolution = []

    for dataset_key, result in results.items():
        history = result['history']
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        gate_usage = [h['gate_usage'] for h in history]
        k_values = [h['k'] for h in history]

        # Plot individual curves (light colors)
        axes[0, 0].plot(epochs, train_losses, alpha=0.3, color='blue', linewidth=0.5)
        axes[0, 1].plot(epochs, val_losses, alpha=0.3, color='red', linewidth=0.5)
        axes[1, 0].plot(epochs, gate_usage, alpha=0.3, color='green', linewidth=0.5)
        axes[1, 1].plot(epochs, k_values, alpha=0.3, color='orange', linewidth=0.5)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_gate_usage.append(gate_usage)
        all_k_evolution.append(k_values)

    # Plot mean curves (bold)
    epochs = list(range(len(all_train_losses[0])))
    mean_train = np.mean(all_train_losses, axis=0)
    mean_val = np.mean(all_val_losses, axis=0)
    mean_gate = np.mean(all_gate_usage, axis=0)
    mean_k = np.mean(all_k_evolution, axis=0)

    axes[0, 0].plot(epochs, mean_train, color='blue', linewidth=2, label=f'Mean (N={n_models})')
    axes[0, 1].plot(epochs, mean_val, color='red', linewidth=2, label=f'Mean (N={n_models})')
    axes[1, 0].plot(epochs, mean_gate, color='green', linewidth=2, label=f'Mean (N={n_models})')
    axes[1, 1].plot(epochs, mean_k, color='orange', linewidth=2, label=f'Mean (N={n_models})')

    # Formatting
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_title("Gate Usage (0=Neural, 1=Physics)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Gate Weight")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_title("Physics Parameter k Evolution")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("k coefficient")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    plt.show()


def analyze_results(results: Dict[str, Any], datasets: Dict[str, Dict[str, Any]]):
    """Analyze final results across all models"""

    print(f"\nFINAL RESULTS ANALYSIS")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Freq':<6} {'Sigma':<7} {'Val Loss':<10} {'Gate':<7} {'k_init':<10} {'k_final':<10}")
    print("-" * 80)

    val_losses = []
    gate_usages = []
    devices_used = set()

    for dataset_key, result in results.items():
        dataset = datasets[dataset_key]
        freq = dataset['freq']
        sigma = dataset['sigma']
        k_init = dataset['k']

        val_loss = result['best_val_loss']
        final_gate = result['history'][-1]['gate_usage']
        k_final, _ = result['final_coefficients']
        device = result.get('device', 'unknown')

        val_losses.append(val_loss)
        gate_usages.append(final_gate)
        devices_used.add(device)

        print(f"{dataset_key:<15} {freq:<6.0f} {sigma:<7.2f} {val_loss:<10.4f} {final_gate:<7.3f} "
              f"{k_init:<10.2e} {k_final:<10.2e}")

    print("-" * 80)
    print(f"Mean validation loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Mean gate usage: {np.mean(gate_usages):.3f} ± {np.std(gate_usages):.3f}")
    print(f"Training devices used: {', '.join(devices_used)}")
    print("=" * 80)