"""
Rain Attenuation Simulator with ITU-R P.838-3 Tables
Includes AR(1) process for temporal correlation
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple, List
from sklearn.model_selection import train_test_split


# ITU-R P.838-3 Table 5 (k, α) coefficients (March 2005)
ITU_P838_TABLE = {
    "horizontal": {
         1:  (2.59e-5, 0.9691),  1.5: (4.43e-5, 1.0185),  2:  (8.47e-5, 1.0664),
       2.5:  (1.321e-4, 1.1209),  3:  (1.390e-4, 1.2322),  3.5:(1.155e-4, 1.4189),
         4:  (1.071e-4, 1.6009),  4.5: (1.340e-4, 1.6948),  5:  (2.162e-4, 1.6969),
       5.5:  (3.909e-4, 1.6499),  6:  (7.056e-4, 1.5900),  7:  (1.915e-3, 1.4810),
         8:  (4.115e-3, 1.3905),  9:  (7.535e-3, 1.3155), 10: (1.217e-2, 1.2571),
        11:  (1.772e-2, 1.2140), 12: (2.386e-2, 1.1825), 13: (3.041e-2, 1.1586),
        14:  (3.738e-2, 1.1396), 15: (4.481e-2, 1.1233), 16: (5.282e-2, 1.1086),
        17:  (6.146e-2, 1.0949), 18: (7.078e-2, 1.0818), 19: (8.084e-2, 1.0691),
        20:  (9.164e-2, 1.0568), 21: (1.032e-1, 1.0447), 22: (1.155e-1, 1.0329),
        23:  (1.286e-1, 1.0214), 24: (1.425e-1, 1.0101), 25: (1.571e-1, 0.9991),
        26:  (1.724e-1, 0.9884), 27: (1.884e-1, 0.9780), 28: (2.051e-1, 0.9679),
        29:  (2.224e-1, 0.9580), 30: (2.403e-1, 0.9485), 31: (2.588e-1, 0.9392),
        32:  (2.778e-1, 0.9302), 33: (2.972e-1, 0.9214), 34: (3.171e-1, 0.9129),
        35:  (3.374e-1, 0.9047), 36: (3.580e-1, 0.8967), 37: (3.789e-1, 0.8890),
        38:  (4.001e-1, 0.8816), 39: (4.215e-1, 0.8743), 40: (4.431e-1, 0.8673),
        41:  (4.647e-1, 0.8605), 42: (4.865e-1, 0.8539), 43: (5.084e-1, 0.8476),
        44:  (5.302e-1, 0.8414), 45: (5.521e-1, 0.8355), 46: (5.738e-1, 0.8297),
        47:  (5.956e-1, 0.8241), 48: (6.172e-1, 0.8187), 49: (6.386e-1, 0.8134),
        50:  (6.600e-1, 0.8084), 51: (6.811e-1, 0.8034), 52: (7.020e-1, 0.7987),
        53:  (7.228e-1, 0.7941), 54: (7.433e-1, 0.7896), 55: (7.635e-1, 0.7853),
        56:  (7.835e-1, 0.7811), 57: (8.032e-1, 0.7771), 58: (8.226e-1, 0.7731),
        59:  (8.418e-1, 0.7693), 60: (8.606e-1, 0.7656), 61: (8.791e-1, 0.7621),
        62:  (8.974e-1, 0.7586), 63: (9.153e-1, 0.7552), 64: (9.328e-1, 0.7520),
        65:  (9.501e-1, 0.7488), 66: (9.670e-1, 0.7458), 67: (9.836e-1, 0.7428),
        68:  (9.999e-1, 0.7400), 69: (1.0159, 0.7372), 70: (1.0315, 0.7345),
        71: (1.0468, 0.7318), 72: (1.0618, 0.7293), 73: (1.0764, 0.7268),
        74: (1.0908, 0.7244), 75: (1.1048, 0.7221), 76: (1.1185, 0.7199),
        77: (1.1320, 0.7177), 78: (1.1451, 0.7156), 79: (1.1579, 0.7135),
        80: (1.1704, 0.7115), 81: (1.1827, 0.7096), 82: (1.1946, 0.7077),
        83: (1.2063, 0.7058), 84: (1.2177, 0.7040), 85: (1.2289, 0.7023),
        86: (1.2398, 0.7006), 87: (1.2504, 0.6990), 88: (1.2607, 0.6974),
        89: (1.2708, 0.6959), 90: (1.2807, 0.6944), 91: (1.2903, 0.6929),
        92: (1.2997, 0.6915), 93: (1.3089, 0.6901), 94: (1.3179, 0.6888),
        95: (1.3266, 0.6875), 96: (1.3351, 0.6862), 97: (1.3434, 0.6850),
        98: (1.3515, 0.6838), 99: (1.3594, 0.6826), 100:(1.3671, 0.6815),
       120:(1.4866, 0.6640), 150:(1.5823, 0.6494), 200:(1.6378, 0.6382),
       300:(1.6286, 0.6296), 400:(1.5860, 0.6262), 500:(1.5418, 0.6253),
       600:(1.5013, 0.6262), 700:(1.4654, 0.6284), 800:(1.4335, 0.6315),
       900:(1.4050, 0.6353), 1000:(1.3795, 0.6396)
    },
    "vertical": {
         1:  (3.08e-5, 0.8592),  1.5: (5.74e-5, 0.8957),  2:  (9.98e-5, 0.9490),
       2.5:  (1.464e-4, 1.0085),  3:  (1.942e-4, 1.0688),  3.5:(2.346e-4, 1.1387),
         4:  (2.461e-4, 1.2476),  4.5: (2.347e-4, 1.3987),  5:  (2.428e-4, 1.5317),
       5.5:  (3.115e-4, 1.5882),  6:  (4.878e-4, 1.5728),  7:  (1.425e-3, 1.4745),
         8:  (3.450e-3, 1.3797),  9:  (6.691e-3, 1.2895), 10: (1.129e-2, 1.2157),
        11:  (1.731e-2, 1.1617), 12: (2.455e-2, 1.1216), 13: (3.266e-2, 1.0901),
        14:  (4.126e-2, 1.0646), 15: (5.008e-2, 1.0440), 16: (5.899e-2, 1.0273),
        17:  (6.797e-2, 1.0137), 18: (7.708e-2, 1.0025), 19: (8.642e-2, 0.9930),
        20:  (9.611e-2, 0.9847), 21: (1.063e-1, 0.9771), 22: (1.170e-1, 0.9700),
        23:  (1.284e-1, 0.9630), 24: (1.404e-1, 0.9561), 25: (1.533e-1, 0.9491),
        26:  (1.669e-1, 0.9421), 27: (1.813e-1, 0.9349), 28: (1.964e-1, 0.9277),
        29:  (2.124e-1, 0.9203), 30: (2.291e-1, 0.9129), 31: (2.465e-1, 0.9055),
        32:  (2.646e-1, 0.8981), 33: (2.833e-1, 0.8907), 34: (3.026e-1, 0.8834),
        35:  (3.224e-1, 0.8761), 36: (3.427e-1, 0.8690), 37: (3.633e-1, 0.8621),
        38:  (3.844e-1, 0.8552), 39: (4.058e-1, 0.8486), 40: (4.274e-1, 0.8421),
        41:  (4.492e-1, 0.8357), 42: (4.712e-1, 0.8296), 43: (4.932e-1, 0.8236),
        44:  (5.153e-1, 0.8179), 45: (5.375e-1, 0.8123), 46: (5.596e-1, 0.8069),
        47:  (5.817e-1, 0.8017), 48: (6.037e-1, 0.7967), 49: (6.255e-1, 0.7918),
        50:  (6.472e-1, 0.7871), 51: (6.687e-1, 0.7826), 52: (6.901e-1, 0.7783),
        53:  (7.112e-1, 0.7741), 54: (7.321e-1, 0.7700), 55: (7.527e-1, 0.7661),
        56:  (7.730e-1, 0.7623), 57: (7.931e-1, 0.7587), 58: (8.129e-1, 0.7552),
        59:  (8.324e-1, 0.7518), 60: (8.515e-1, 0.7486), 61: (8.704e-1, 0.7454),
        62:  (8.889e-1, 0.7424), 63: (9.071e-1, 0.7395), 64: (9.250e-1, 0.7366),
        65:  (9.425e-1, 0.7339), 66: (9.598e-1, 0.7313), 67: (9.767e-1, 0.7287),
        68:  (9.932e-1, 0.7262), 69: (1.0094, 0.7238), 70: (1.0253, 0.7215),
        71: (1.0409, 0.7193), 72: (1.0561, 0.7171), 73: (1.0711, 0.7150),
        74: (1.0857, 0.7130), 75: (1.1000, 0.7110), 76: (1.1139, 0.7091),
        77: (1.1276, 0.7073), 78: (1.1410, 0.7055), 79: (1.1541, 0.7038),
        80: (1.1668, 0.7021), 81: (1.1793, 0.7004), 82: (1.1915, 0.6988),
        83: (1.2034, 0.6973), 84: (1.2151, 0.6958), 85: (1.2265, 0.6943),
        86: (1.2376, 0.6930), 87: (1.2484, 0.6915), 88: (1.2590, 0.6902),
        89: (1.2694, 0.6889), 90: (1.2795, 0.6876), 91: (1.2893, 0.6864),
        92: (1.2989, 0.6852), 93: (1.3083, 0.6840), 94: (1.3175, 0.6828)
    }
}


def get_k_alpha(freq_ghz: float, pol: str = "vertical") -> Tuple[float, float]:
    """Get ITU-R coefficients with linear interpolation"""
    if pol not in ITU_P838_TABLE:
        raise ValueError(f"Polarization must be 'horizontal' or 'vertical', got '{pol}'")

    table = ITU_P838_TABLE[pol]
    freqs = sorted(table.keys())

    # Handle boundary cases
    if freq_ghz <= freqs[0]:
        return table[freqs[0]]
    if freq_ghz >= freqs[-1]:
        return table[freqs[-1]]

    # Linear interpolation in log space for k
    for i in range(len(freqs) - 1):
        f1, f2 = freqs[i], freqs[i + 1]
        if f1 <= freq_ghz <= f2:
            k1, alpha1 = table[f1]
            k2, alpha2 = table[f2]

            log_k = np.interp(freq_ghz, [f1, f2], [np.log(k1), np.log(k2)])
            alpha = np.interp(freq_ghz, [f1, f2], [alpha1, alpha2])

            return np.exp(log_k), alpha

    raise ValueError(f"Frequency {freq_ghz} GHz not found in table")


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