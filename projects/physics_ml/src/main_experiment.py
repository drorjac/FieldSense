"""
Main experiment script for hybrid rain retrieval model
Clean interface for running experiments
"""
import sys
from dataclasses import dataclass
from typing import List

from data_analysis import analyze_synthetic_data, generate_datasets, summarize_datasets
from training_utils import train_models, plot_training_curves, analyze_results, get_device
from rain_simulator import rain_params_from_itu


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters"""
    frequencies: List[float] = None  # GHz
    noise_levels: List[float] = None  # dB
    n_samples: int = 10000
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size: int = 64
    link_length_km: float = 1.5
    baseline_atten_db: float = 2.0
    test_size: float = 0.3
    val_split: float = 0.25  # Validation split from training data
    seed: int = 42

    def __post_init__(self):
        if self.frequencies is None:
            self.frequencies = [5, 24, 28, 38, 60, 70]
        if self.noise_levels is None:
            self.noise_levels = [0.01, 0.2, 0.5, 0.7, 1.0]


def main():
    """Main experiment execution with GPU support"""

    # Configuration
    config = ExperimentConfig(
        frequencies=[24, 38, 60],  # Selected frequencies for demo
        noise_levels=[0.2, 0.5, 1.0],
        n_samples=5000,  # Smaller for faster testing
        epochs=15,
        batch_size=32,
        learning_rate=1e-3
    )

    print("HYBRID RAIN RETRIEVAL EXPERIMENT")
    print("=" * 50)
    print(f"Frequencies: {config.frequencies} GHz")
    print(f"Noise levels: {config.noise_levels} dB")
    print(f"Samples per configuration: {config.n_samples}")
    print(f"Training epochs: {config.epochs}")

    # Check GPU availability
    device = get_device()
    print(f"Training device: {device}")

    # Step 1: Analyze synthetic data
    print("\nStep 1: Data Analysis")
    gen = analyze_synthetic_data(config, save_plots=True)

    # Ask user if they want to proceed with dataset generation
    print(f"\nPress Enter to continue with dataset generation and training...")
    input()

    # Step 2: Generate datasets
    print("\nStep 2: Dataset Generation")
    datasets = generate_datasets(config)
    summarize_datasets(datasets)

    # Ask user if they want to proceed with training
    print(f"\nPress Enter to continue with model training on {device}...")
    input()

    # Step 3: Train models
    print(f"\nStep 3: Model Training")
    results = train_models(datasets, config)

    # Step 4: Analyze results
    print(f"\nStep 4: Results Analysis")
    analyze_results(results, datasets)

    # Step 5: Plot training curves
    print(f"\nStep 5: Visualization")
    plot_training_curves(results, save_path="training_curves.png")

    print(f"\nExperiment completed! Generated plots:")
    print("- synthetic_data_analysis.png")
    print("- rain_process_analysis.png")
    print("- training_curves.png")

    return datasets, results, config


def analyze_data_only():
    """Just analyze and visualize synthetic data without training"""

    config = ExperimentConfig(
        frequencies=[5, 24, 38, 60, 70],  # All frequencies for analysis
        noise_levels=[0.1, 0.5, 1.0],
        n_samples=3000
    )

    print("DATA ANALYSIS MODE")
    print("=" * 30)

    gen = analyze_synthetic_data(config, save_plots=True)

    # Additional detailed analysis for single frequency
    print(f"\nDetailed analysis for 24 GHz:")
    gen.plot_noise_comparison(
        sigma_vals=[0.1, 0.3, 0.7, 1.0],
        n_samples=2000,
        rain_params=rain_params_from_itu(24, "vertical"),
        link_length_km=config.link_length_km,
        freq_ghz=24,
        rain_sampler=lambda n: gen._rain_ar1(n, phi=0.7, mu=5.0, sigma_eps=2.0),
        unified=True
    )

    return config, gen


if __name__ == "__main__":
    # Choose mode
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-only":
        # Just data analysis
        config, gen = analyze_data_only()
    else:
        # Full experiment with GPU training
        datasets, results, config = main()