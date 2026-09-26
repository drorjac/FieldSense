"""
Rain Attenuation Simulator with ITU-R P.838-3 Tables
Includes AR(1) process for temporal correlation
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple, List

from core import itu_p838 as _itu
from sklearn.model_selection import train_test_split


# ITU-R P.838-3 Table 5 (k, α) coefficients (March 2005)
# ITU-R P.838-3 coefficients come from core.itu_p838, which is the single
# copy in this repository. This module used to carry its own duplicate of
# Table 5 plus an interpolation that was linear in frequency; the
# recommendation prescribes log-log for k and semi-log for alpha, so the two
# disagreed between tabulated points - 0.00222 against 0.00225 at 7.5 GHz.
# The tables were identical, only the interpolation differed, and the shared
# one is the one that follows the recommendation.
ITU_P838_TABLE = _itu.ITU_P838_TABLE
get_k_alpha = _itu.get_k_alpha


class NoiseType(Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    UNIFORM = "uniform"


@dataclass
class RainParams:
    a_coeff: float
    b_exponent: float
    rain_range: Tuple[float, float] = (0.1, 30.0)


class RainAttenuationGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _noise(self, n: int, kind: NoiseType, sigma: float) -> np.ndarray:
        r = self.rng
        if kind is NoiseType.NORMAL:
            return r.normal(0, sigma, n)
        if kind is NoiseType.LOGNORMAL:
            return r.lognormal(0, sigma, n) - np.exp(sigma**2 / 2)
        if kind is NoiseType.GAMMA:
            shape = 2.0
            return r.gamma(shape, sigma / np.sqrt(shape), n) - sigma*np.sqrt(shape)
        if kind is NoiseType.UNIFORM:
            width = sigma * np.sqrt(12)
            return r.uniform(-width/2, width/2, n)
        raise ValueError("unknown noise type")

    def _rain_uniform(self, n: int, rp: RainParams) -> np.ndarray:
        lo, hi = rp.rain_range
        return self.rng.uniform(lo, hi, n)

    def _rain_gamma(self, n: int, shape=1.5, scale=4.0) -> np.ndarray:
        return np.clip(self.rng.gamma(shape, scale, n), 0.01, None)

    def _rain_ar1(self, n: int, phi: float = 0.8, mu: float = 5.0, sigma_eps: float = 2.0) -> np.ndarray:
        """Generate AR(1) rain rate process: R_t = phi * R_{t-1} + (1-phi)*mu + eps_t"""
        if not (0 <= phi < 1):
            raise ValueError("AR(1) coefficient phi must be in [0, 1)")

        rain_rates = np.zeros(n)
        rain_rates[0] = mu  # Initialize at mean

        for t in range(1, n):
            eps_t = self.rng.normal(0, sigma_eps)
            rain_rates[t] = phi * rain_rates[t-1] + (1-phi) * mu + eps_t
            rain_rates[t] = max(0.01, rain_rates[t])  # Ensure positive

        return rain_rates

    def generate_data(self, n_samples: int, rain_params: RainParams,
                     link_length_km: float = 1.0, noise_type: NoiseType = NoiseType.NORMAL,
                     sigma_db: float = 0.1, rain_sampler: Callable[[int], np.ndarray] = None,
                     test_size: float = 0.25, add_wet_antenna: bool = False,
                     wet_max_db: float = 1.5, baseline_atten_db: float = 0.0):

        # 1) Generate rain rates
        if rain_sampler is None:
            rain_rates = self._rain_uniform(n_samples, rain_params)
        else:
            rain_rates = rain_sampler(n_samples)

        # 2) Clean attenuation with baseline
        A_clean = (rain_params.a_coeff * rain_rates**rain_params.b_exponent * link_length_km) + baseline_atten_db

        if add_wet_antenna:
            wet = wet_max_db * (1 - np.exp(-rain_rates / 10))
            A_clean += wet

        # 3) Add measurement noise
        A_noisy = np.clip(A_clean + self._noise(n_samples, noise_type, sigma_db), 0.001, None)

        # 4) Split data
        if 0.0 < test_size < 1.0:
            R_tr, R_te, A_tr, A_te = train_test_split(rain_rates, A_noisy, test_size=test_size, random_state=0)
        else:
            R_tr, A_tr = rain_rates, A_noisy
            R_te, A_te = np.array([]), np.array([])

        meta = {
            'a': rain_params.a_coeff,
            'alpha': rain_params.b_exponent,
            'link_km': link_length_km,
            'sigma_db': sigma_db,
            'n': n_samples,
            'noise': noise_type.value,
            'baseline_db': baseline_atten_db
        }

        return R_tr, R_te, A_tr, A_te, meta

    def plot_noise_comparison(self, sigma_vals: List[float] = (0.01, 0.20, 1.00),
                             n_samples: int = 2000, rain_params: RainParams = None,
                             link_length_km: float = 1.0, freq_ghz: float = 5.0,
                             rain_sampler: Callable[[int], np.ndarray] = None):

        if rain_params is None:
            rain_params = RainParams(0.0367, 1.097)

        fig, axes = plt.subplots(1, len(sigma_vals), figsize=(5*len(sigma_vals), 4), sharey=True)

        header = (f"L = {link_length_km:g} km   f = {freq_ghz:g} GHz   "
                 f"k={rain_params.a_coeff:.2g}, α={rain_params.b_exponent:.2f}")
        fig.suptitle(header, fontsize=12, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a9df4", edgecolor="none"))

        # Reference curve
        R_ref = np.linspace(*rain_params.rain_range, 200)
        A_ref = rain_params.a_coeff * R_ref**rain_params.b_exponent * link_length_km

        palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_vals)))

        for ax, sigma, col in zip(np.atleast_1d(axes), sigma_vals, palette):
            R_tr, _R_te, A_tr, _A_te, _meta = self.generate_data(
                n_samples, rain_params, link_length_km=link_length_km,
                noise_type=NoiseType.NORMAL, sigma_db=sigma,
                rain_sampler=rain_sampler, test_size=0.0)

            ax.scatter(R_tr, A_tr, s=8, alpha=0.6, color=col, label="Samples")
            ax.plot(R_ref, A_ref, "--k", lw=1.3, label="$A = kR^{\\alpha}L$")

            ax.set_xlabel("Rain rate (mm h$^{-1}$)", fontsize=14)
            if ax is axes[0]:
                ax.set_ylabel("Attenuation (dB)", fontsize=14)
            ax.set_xlim(0, 31)
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=12)

            legend = ax.legend(loc="upper left", fontsize=12, title=f"σ = {sigma:.2f} dB")
            legend.get_title().set_fontsize(12)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

    def plot_synthetic_data(self, frequencies: List[float], sigma_vals: List[float],
                            n_samples: int = 2000, link_length_km: float = 1.0,
                            rain_sampler: Callable[[int], np.ndarray] = None,
                            save_path: Optional[str] = None):
        """Attenuation against rain rate, one panel per frequency x noise level.

        Each frequency uses its ITU-R P.838-3 coefficients; the dashed line is
        the noise-free power law.
        """
        fig, axes = plt.subplots(len(sigma_vals), len(frequencies), squeeze=False,
                                 figsize=(3.6 * len(frequencies), 3.0 * len(sigma_vals)))
        for j, freq in enumerate(frequencies):
            rp = rain_params_from_itu(freq)
            for i, sigma in enumerate(sigma_vals):
                ax = axes[i][j]
                R, _, A, _, _ = self.generate_data(
                    n_samples, rp, link_length_km=link_length_km, sigma_db=sigma,
                    rain_sampler=rain_sampler, test_size=0.0)
                R_ref = np.linspace(0, max(R.max(), 1.0), 200)
                ax.scatter(R, A, s=4, alpha=0.4)
                ax.plot(R_ref, rp.a_coeff * R_ref ** rp.b_exponent * link_length_km, "--k", lw=1.2)
                ax.set_title(f"{freq:g} GHz, σ = {sigma:g} dB", fontsize=10)
                ax.grid(alpha=0.3)
                if i == len(sigma_vals) - 1:
                    ax.set_xlabel("rain rate (mm/h)")
                if j == 0:
                    ax.set_ylabel("attenuation (dB)")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        plt.show()
        return fig


def rain_params_from_itu(freq_ghz: float, pol: str = "vertical") -> RainParams:
    """Return RainParams with ITU-R coefficients"""
    k, alpha = get_k_alpha(freq_ghz, pol)
    return RainParams(a_coeff=k, b_exponent=alpha)


if __name__ == "__main__":
    # Demo of ITU coefficients
    print("ITU-R P.838-3 Coefficients Demo:")
    for f in (5, 23, 60, 70):
        kH, aH = get_k_alpha(f, "horizontal")
        kV, aV = get_k_alpha(f, "vertical")
        print(f"{f} GHz  H: k={kH:.3e}, α={aH:.3f}   |   V: k={kV:.3e}, α={aV:.3f}")

    # Demo of AR(1) process
    gen = RainAttenuationGenerator(seed=42)
    ar1_rain = gen._rain_ar1(1000, phi=0.8, mu=5.0, sigma_eps=2.0)
    print(f"\nAR(1) rain process demo: mean={ar1_rain.mean():.2f}, std={ar1_rain.std():.2f}")