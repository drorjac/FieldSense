"""
Data analysis functions for hybrid rain retrieval
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

from rain_simulator import RainAttenuationGenerator, rain_params_from_itu, get_k_alpha


def analyze_synthetic_data(config, save_plots: bool = True):
    """
    Analyze and visualize synthetic data before training
    """
    print("\n📊 ANALYZING SYNTHETIC DATA")
    print("=" * 50)

    gen = RainAttenuationGenerator(seed=config.seed)

    # Show data for selected frequencies
    analysis_freqs = config.frequencies[:4] if len(config.frequencies) > 4 else config.frequencies
    analysis_sigmas = [0.2, 0.7]  # Two representative noise levels

    print(f"Analyzing frequencies: {analysis_freqs} GHz")
    print(f"Noise levels: {analysis_sigmas} dB")

    # Plot synthetic data grid
    gen.plot_synthetic_data(
        frequencies=analysis_freqs,
        sigma_vals=analysis_sigmas,
        n_samples=2000,
        link_length_km=config.link_length_km,
        rain_sampler=lambda n: gen._rain_ar1(n, phi=0.7, mu=5.0, sigma_eps=2.0),
        save_path="synthetic_data_analysis.png" if save_plots else None
    )

    # Print ITU coefficients for all frequencies
    print(f"\n📋 ITU-R P.838-3 Coefficients:")
    print("-" * 40)
    print(f"{'Freq (GHz)':<10} {'k coeff':<12} {'α exponent':<12}")
    print("-" * 40)

    for freq in config.frequencies:
        k, alpha = get_k_alpha(freq, "vertical")
        print(f"{freq:<10.0f} {k:<12.2e} {alpha:<12.3f}")

    # Analyze AR(1) rain process
    print(f"\n🌧️ AR(1) Rain Process Analysis:")
    print("-" * 40)

    # Generate AR(1) samples for analysis
    ar1_samples = gen._rain_ar1(5000, phi=0.7, mu=5.0, sigma_eps=2.0)
    gamma_samples = gen._rain_gamma(5000, 1.5, 4.0)

    print(f"AR(1) Process (φ=0.7, μ=5.0, σ=2.0):")
    print(f"  Mean: {ar1_samples.mean():.2f} mm/h")
    print(f"  Std:  {ar1_samples.std():.2f} mm/h")
    print(f"  Range: [{ar1_samples.min():.2f}, {ar1_samples.max():.2f}] mm/h")

    print(f"\nGamma Process (shape=1.5, scale=4.0):")
    print(f"  Mean: {gamma_samples.mean():.2f} mm/h")
    print(f"  Std:  {gamma_samples.std():.2f} mm/h")
    print(f"  Range: [{gamma_samples.min():.2f}, {gamma_samples.max():.2f}] mm/h")

    # Plot rain process comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Time series
    ax1.plot(ar1_samples[:200], 'b-', alpha=0.8, label='AR(1)')
    ax1.plot(gamma_samples[:200], 'r-', alpha=0.6, label='Gamma')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Rain rate (mm/h)')
    ax1.set_title('Rain Rate Time Series (first 200 samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histograms
    ax2.hist(ar1_samples, bins=50, alpha=0.6, density=True, label='AR(1)', color='blue')
    ax2.hist(gamma_samples, bins=50, alpha=0.6, density=True, label='Gamma', color='red')
    ax2.set_xlabel('Rain rate (mm/h)')
    ax2.set_ylabel('Density')
    ax2.set_title('Rain Rate Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig("rain_process_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Rain process analysis saved to: rain_process_analysis.png")

    plt.show()

    return gen


def generate_datasets(config) -> Dict[str, Dict[str, Any]]:
    """Generate datasets for all frequency/noise combinations"""

    gen = RainAttenuationGenerator(seed=config.seed)
    datasets = {}
    total_configs = len(config.frequencies) * len(config.noise_levels)
    config_count = 0

    print("\nGenerating datasets...")
    print("=" * 80)

    for freq in config.frequencies:
        for sigma in config.noise_levels:
            config_count += 1

            # Get ITU coefficients
            rain_params = rain_params_from_itu(freq, "vertical")

            # Generate data using AR(1) process for temporal correlation
            R_tr, R_te, A_tr, A_te, meta = gen.generate_data(
                n_samples=config.n_samples,
                rain_params=rain_params,
                link_length_km=config.link_length_km,
                sigma_db=sigma,
                rain_sampler=lambda n: gen._rain_ar1(n, phi=0.7, mu=5.0, sigma_eps=2.0),
                baseline_atten_db=config.baseline_atten_db,
                test_size=config.test_size
            )

            # Further split training into train/validation
            R_train, R_val, A_train, A_val = train_test_split(
                R_tr, A_tr, test_size=config.val_split, random_state=config.seed
            )

            # Store dataset
            dataset_key = f"{freq}GHz_sigma{sigma}"
            datasets[dataset_key] = {
                'freq': freq,
                'sigma': sigma,
                'k': rain_params.a_coeff,
                'alpha': rain_params.b_exponent,
                'train': {'R': R_train, 'A': A_train},
                'val': {'R': R_val, 'A': A_val},
                'test': {'R': R_te, 'A': A_te},
                'meta': meta
            }

            print(f"[{config_count:2d}/{total_configs}] {freq:4.0f} GHz, σ={sigma:4.2f} dB: "
                  f"Train={len(R_train):4d}, Val={len(R_val):4d}, Test={len(R_te):4d}")

    return datasets


def summarize_datasets(datasets: Dict[str, Dict[str, Any]]):
    """Print summary table of all datasets"""

    print(f"\nDATASET SUMMARY")
    print("=" * 80)
    print(f"{'Freq':<8} {'Sigma':<8} {'k coeff':<12} {'α':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 80)

    total_train = total_val = total_test = 0

    for key, data in datasets.items():
        freq = data['freq']
        sigma = data['sigma']
        k = data['k']
        alpha = data['alpha']
        n_train = len(data['train']['R'])
        n_val = len(data['val']['R'])
        n_test = len(data['test']['R'])

        total_train += n_train
        total_val += n_val
        total_test += n_test

        print(f"{freq:<8.0f} {sigma:<8.2f} {k:<12.2e} {alpha:<8.3f} "
              f"{n_train:<8} {n_val:<8} {n_test:<8}")

    print("-" * 80)
    print(f"{'TOTAL':<8} {'':<8} {'':<12} {'':<8} {total_train:<8} {total_val:<8} {total_test:<8}")
    print("=" * 80)