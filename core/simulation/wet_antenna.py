"""
Wet-antenna attenuation: the static models of the literature, and one with memory.

Water on the radome of a link's antennas attenuates the signal without any rain
on the path. The term does not grow with path length, so on a short link it can
be as large as the rain signal itself (Schleiss et al. 2013; Ostrometzky et al.
2018). The literature models fix its form in advance:

``waa_constant``        a fixed attenuation whenever the link is wet
                        (the simplest correction, e.g. Overeem et al. 2011)
``waa_schleiss_2013``   time-dependent: builds up during a wet spell towards a
                        maximum, with a time constant (Schleiss, Rieckermann &
                        Berne 2013); the discrete form follows pycomlink
``waa_pastorek_2021``   rain-dependent: saturates with rain rate,
                        ``A_max (1 - exp(-d R**zeta))`` - the "KR-alt" model of
                        Pastorek et al. (2021), after Kharadly & Ross (2001)

None of these has memory of the rain that stopped: they drop to zero as soon
as the link is called dry or the rain rate is zero. A real water film dries
over tens of minutes, which leaves a tail of attenuation after every event.
:class:`DynamicWetAntenna` is a one-state model of that film,

    d delta / dt = (delta_eq(R) - delta) / tau,

relaxing towards the rain-dependent equilibrium ``delta_eq(R)`` (the Pastorek
form) with one time constant while it wets (``delta < delta_eq``) and another
while it dries. It is the known truth of the simulator, the thing a method that
learns ``d delta / dt = F(delta, R)`` from data should recover.

Units: attenuation in dB, rain rate in mm/h, time in minutes.

References
----------
Kharadly, M. M. Z., and Ross, R. (2001). Effect of wet antenna attenuation on
    propagation data statistics. IEEE Trans. Antennas Propag., 49(8), 1183-1191.
Ostrometzky, J., Raich, R., Bao, L., Hansryd, J., and Messer, H. (2018). The
    wet-antenna effect - a factor to be considered in future communication
    networks. IEEE Trans. Antennas Propag., 66(1), 315-322.
Overeem, A., Leijnse, H., and Uijlenhoet, R. (2011). Measuring urban rainfall
    using microwave links from commercial cellular communication networks.
    Water Resour. Res., 47, W12505.
Pastorek, J., Fencl, M., Rieckermann, J., and Bares, V. (2021). Precipitation
    estimates from commercial microwave links: practical approaches to
    wet-antenna correction. IEEE Trans. Geosci. Remote Sens., 60, 1-9.
Schleiss, M., Rieckermann, J., and Berne, A. (2013). Quantification and
    modeling of wet-antenna attenuation for commercial microwave links. IEEE
    Geosci. Remote Sens. Lett., 10(5), 1195-1199.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# static literature models
# --------------------------------------------------------------------------
def waa_constant(wet: np.ndarray, a_const_db: float) -> np.ndarray:
    """``a_const_db`` wherever ``wet``, zero elsewhere."""
    return np.where(np.asarray(wet, dtype=bool), float(a_const_db), 0.0)


def waa_schleiss_2013(wet: np.ndarray, waa_max_db: float, tau_min: float, dt_min: float,
                      a_obs: Optional[np.ndarray] = None) -> np.ndarray:
    """Time-dependent wet antenna of Schleiss et al. (2013), along the LAST axis.

    During a wet spell the attenuation grows towards ``waa_max_db``,
    ``w[i] = w[i-1] + (waa_max - w[i-1]) * 3 dt / tau``, so it reaches about 95 %
    of the maximum ``tau_min`` after the spell starts. When dry it is zero.

    With ``a_obs`` (the observed excess attenuation) the model is pycomlink's
    ``waa_schleiss_2013`` exactly: the wet antenna can never exceed ``a_obs``,
    and in dry samples it is ``min(a_obs, waa_max)`` - whatever is left over
    the baseline is attributed to the antenna, not to rain.
    """
    wet = np.asarray(wet, dtype=bool)
    out = np.zeros(wet.shape, dtype=float)
    a = None if a_obs is None else np.asarray(a_obs, dtype=float)
    gain = min(1.0, 3.0 * dt_min / tau_min)
    for i in range(1, wet.shape[-1]):
        prev = out[..., i - 1]
        grown = prev + (waa_max_db - prev) * gain
        if a is None:
            out[..., i] = np.where(wet[..., i], np.minimum(grown, waa_max_db), 0.0)
        else:
            cap = np.minimum(a[..., i], waa_max_db)
            out[..., i] = np.where(wet[..., i], np.minimum(grown, cap), cap)
    return out


def waa_pastorek_2021(rain_mm_h: np.ndarray, a_max_db: float = 14.0, d: float = 0.1,
                      zeta: float = 0.55) -> np.ndarray:
    """Rain-dependent wet antenna, ``A_max (1 - exp(-d R**zeta))`` (Pastorek et al. 2021).

    The defaults are pycomlink's (the "KR-alt" fit of the paper). ``zeta = 1``
    gives the saturating form ``A_max (1 - exp(-d R))`` that
    ``cml_network.wet_antenna_db`` uses. Exactly zero for ``R <= 0``.
    """
    r = np.clip(np.asarray(rain_mm_h, dtype=float), 0.0, None)
    return np.where(r > 0, a_max_db * (1.0 - np.exp(-d * r ** zeta)), 0.0)


# --------------------------------------------------------------------------
# a wet antenna with memory
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DynamicWetAntenna:
    """A water film that builds up with rain and dries after it.

    ``d delta / dt = (delta_eq(R) - delta) / tau`` with
    ``delta_eq(R) = a_max_db (1 - exp(-gamma R**zeta))`` and
    ``tau = tau_wet_min`` while ``delta < delta_eq`` (wetting), ``tau_dry_min``
    otherwise (drying). After the rain stops, ``delta`` decays as
    ``exp(-t / tau_dry_min)``: the drying tail.

    Defaults are plausible, not fitted: a 2 dB film that saturates near
    10 mm/h, wets within minutes and dries in about half an hour.
    """

    a_max_db: float = 2.0
    gamma_per_mm_h: float = 0.3
    zeta: float = 1.0
    tau_wet_min: float = 5.0
    tau_dry_min: float = 30.0

    def __post_init__(self):
        if self.tau_wet_min <= 0 or self.tau_dry_min <= 0:
            raise ValueError("time constants must be positive")

    def equilibrium(self, rain_mm_h) -> np.ndarray:
        """``delta_eq(R)``: the attenuation the film settles at under constant rain."""
        return waa_pastorek_2021(rain_mm_h, self.a_max_db, self.gamma_per_mm_h, self.zeta)

    def rate(self, delta, rain_mm_h) -> np.ndarray:
        """``d delta / dt`` in dB per minute: the right-hand side to be learned."""
        delta = np.asarray(delta, dtype=float)
        eq = self.equilibrium(rain_mm_h)
        tau = np.where(delta < eq, self.tau_wet_min, self.tau_dry_min)
        return (eq - delta) / tau

    def step(self, delta, rain_mm_h, dt_min: float) -> np.ndarray:
        """Advance ``delta`` by ``dt_min`` with the rain rate held constant over the step.

        The exact solution of the linear relaxation, so it is stable and has no
        time-step error for any ``dt_min``. Which time constant applies cannot
        change within a step: ``delta`` approaches ``delta_eq`` without crossing it.
        """
        delta = np.asarray(delta, dtype=float)
        eq = self.equilibrium(rain_mm_h)
        tau = np.where(delta < eq, self.tau_wet_min, self.tau_dry_min)
        return eq + (delta - eq) * np.exp(-dt_min / tau)

    def simulate(self, rain_mm_h: np.ndarray, dt_min: float, delta0=0.0) -> np.ndarray:
        """``delta`` along the LAST axis of ``rain_mm_h`` (rain held over each step).

        ``out[..., i]`` is the film at the end of step ``i``, having seen
        ``rain[..., i]`` for ``dt_min``; ``delta0`` is the state before step 0.
        """
        rain = np.asarray(rain_mm_h, dtype=float)
        out = np.empty(rain.shape, dtype=float)
        d = np.broadcast_to(np.asarray(delta0, dtype=float), rain.shape[:-1]).copy()
        for i in range(rain.shape[-1]):
            d = self.step(d, rain[..., i], dt_min)
            out[..., i] = d
        return out
