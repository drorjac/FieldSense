"""Audit of the k-R power law: ITU-R P.838-3 coefficients and their inversion.

The reference is the recommendation itself. P.838-3 defines ``k`` and
``alpha`` by closed-form equations (its equations 2 and 3, with the
coefficients of Tables 1-4); Table 5 is those equations evaluated and rounded.
The coefficients are reproduced below so every tabulated value in the
repository can be checked against the definition, at every frequency and
both polarizations, rather than against a second copy of the same table.
"""

import numpy as np
import pytest

from core.cml.power_law import (ITU_2003, ITU_2005, attenuation_from_rain, itu_ab,
                                rain_from_attenuation)
from core.itu_p838 import ITU_P838_TABLE, get_k_alpha, specific_attenuation

# ITU-R P.838-3, Tables 1-4: (a_j, b_j, c_j, m, c) for log10(k) and alpha.
P838_EQ = {
    ("k", "horizontal"): ([-5.33980, -0.35351, -0.23789, -0.94158],
                          [-0.10008, 1.26970, 0.86036, 0.64552],
                          [1.13098, 0.45400, 0.15354, 0.16817], -0.18961, 0.71147),
    ("k", "vertical"): ([-3.80595, -3.44965, -0.39902, 0.50167],
                        [0.56934, -0.22911, 0.73042, 1.07319],
                        [0.81061, 0.51059, 0.11899, 0.27195], -0.16398, 0.63297),
    ("alpha", "horizontal"): ([-0.14318, 0.29591, 0.32177, -5.37610, 16.1721],
                              [1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
                              [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990], 0.67849, -1.95537),
    ("alpha", "vertical"): ([-0.07771, 0.56727, -0.20238, -48.2991, 48.5833],
                            [2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
                            [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479], -0.053739, 0.83433),
}


def p838(f_ghz, pol):
    """``(k, alpha)`` from the P.838-3 equations."""
    x = np.log10(np.asarray(f_ghz, dtype=float))
    out = []
    for name in ("k", "alpha"):
        a, b, c, m, cc = P838_EQ[(name, pol)]
        s = sum(aj * np.exp(-((x - bj) / cj) ** 2) for aj, bj, cj in zip(a, b, c)) + m * x + cc
        out.append(10.0 ** s if name == "k" else s)
    return out[0], out[1]


@pytest.mark.parametrize("pol", ["horizontal", "vertical"])
def test_itu_p838_table_is_the_recommendation(pol):
    """Every Table 5 entry in core.itu_p838 equals the equations to the table's rounding."""
    for f, (k, alpha) in ITU_P838_TABLE[pol].items():
        k_eq, a_eq = p838(f, pol)
        assert k == pytest.approx(k_eq, rel=5e-3), f
        assert alpha == pytest.approx(a_eq, abs=1.5e-3), f


@pytest.mark.parametrize("row,pol,name", [(1, "horizontal", "k"), (2, "vertical", "k"),
                                          (3, "horizontal", "alpha"), (4, "vertical", "alpha")])
def test_power_law_2005_table_is_the_recommendation(row, pol, name):
    for f, v in zip(ITU_2005[0], ITU_2005[row]):
        ref = p838(f, pol)[0 if name == "k" else 1]
        if name == "k":
            assert v == pytest.approx(ref, rel=5e-3), f
        else:
            assert v == pytest.approx(ref, abs=1.5e-3), f


def test_the_two_copies_of_table_5_agree():
    """core.itu_p838 and core.cml.power_law carry the same Table 5 to the last digit.

    They used to differ at alpha_V(10 GHz) (1.2157 vs 1.2156) and alpha_V(86 GHz)
    (0.6930 vs 0.6929); the recommendation has 1.2156 and 0.6929.
    """
    for row, pol, i in [(1, "horizontal", 0), (2, "vertical", 0), (3, "horizontal", 1), (4, "vertical", 1)]:
        for f, v in zip(ITU_2005[0], ITU_2005[row]):
            if f in ITU_P838_TABLE[pol]:
                assert ITU_P838_TABLE[pol][f][i] == v, (pol, f, i)


@pytest.mark.parametrize("f,pol,k,alpha", [
    (10, "horizontal", 0.01217, 1.2571), (10, "vertical", 0.01129, 1.2156),
    (23, "vertical", 0.1284, 0.9630), (38, "vertical", 0.3844, 0.8552),
    (80, "horizontal", 1.1704, 0.7115), (7, "vertical", 0.001425, 1.4745),
])
def test_spot_values_from_table_5(f, pol, k, alpha):
    assert get_k_alpha(f, pol) == pytest.approx((k, alpha), rel=1e-9)
    a, b = itu_ab(f, pol[0], "ITU_2005")
    assert (a[0], b[0]) == pytest.approx((k, alpha), rel=1e-9)


def test_off_grid_interpolation_against_the_equations():
    """Between tabulated frequencies both interpolations stay close to the equations.

    power_law's cubic spline is within 0.3 % in k; itu_p838's log-log linear
    interpolation is within 3.5 % in k and 0.011 in alpha (worst at 5-7 GHz, where
    the table steps by 1 GHz and log k is most curved; under 2 % above 7 GHz).
    That gap is the RISK documented in docs/algorithm_audit.md: the simulators
    (get_k_alpha) and the core.cml retrievals (itu_ab) do not share one k-R relation
    off the 1-GHz grid.
    """
    f = np.arange(5.25, 99.0, 0.5)
    for pol in ("horizontal", "vertical"):
        k_eq, a_eq = p838(f, pol)
        a, b = itu_ab(f, pol[0], "ITU_2005")
        assert np.max(np.abs(a / k_eq - 1)) < 3e-3
        assert np.max(np.abs(b - a_eq)) < 2e-3
        ka = np.array([get_k_alpha(x, pol) for x in f])
        if pol == "vertical":
            keep = f <= 94
            ka, k_eq, a_eq = ka[keep], k_eq[keep], a_eq[keep]
        assert np.max(np.abs(ka[:, 0] / k_eq - 1)) < 0.035
        assert np.max(np.abs(ka[:, 1] - a_eq)) < 0.011


def test_simulator_and_retrieval_power_laws_agree_within_two_percent():
    """Forward with core.itu_p838, invert with core.cml.power_law: R comes back within 2 %."""
    R = np.array([1.0, 5.0, 20.0])
    for f in (7.5, 15.2, 18.7, 26.5, 38.3, 73.5):
        for pol in ("horizontal", "vertical"):
            A = specific_attenuation(R, f, pol) * 3.0
            back = rain_from_attenuation(A, 3.0, f, pol[0], table="ITU_2005", r_min=0)
            assert np.allclose(back, R, rtol=0.02), (f, pol)


def test_get_k_alpha_clamps_outside_the_table():
    """RISK: out-of-table frequencies are clamped silently (power_law raises instead).

    Vertical polarization is tabulated only to 94 GHz here, so a 100 GHz vertical
    link silently gets the 94 GHz coefficients (k 4 % low).
    """
    assert get_k_alpha(0.5, "horizontal") == get_k_alpha(1.0, "horizontal")
    assert get_k_alpha(100.0, "vertical") == get_k_alpha(94.0, "vertical")
    assert p838(100.0, "vertical")[0] / get_k_alpha(100.0, "vertical")[0] == pytest.approx(1.038, abs=0.005)
    with pytest.raises(ValueError):
        itu_ab(100.5, "v")


def test_specific_attenuation_units():
    """dB/km from mm/h; path attenuation scales with length in km."""
    gamma = specific_attenuation(10.0, 23.0, "vertical")
    assert gamma == pytest.approx(0.1284 * 10 ** 0.9630, rel=1e-12)       # ~1.18 dB/km
    assert attenuation_from_rain(10.0, 2.0, 23.0, "v")[0] == pytest.approx(2 * gamma, rel=1e-12)
    assert specific_attenuation(-3.0, 23.0) == 0.0


def test_horizontal_attenuates_more_than_vertical_in_rain():
    """Oblate drops: gamma_H > gamma_V at 10-50 mm/h for every frequency 10-94 GHz."""
    for f in np.arange(10, 95, 1.0):
        for r in (10.0, 50.0):
            assert specific_attenuation(r, f, "horizontal") > specific_attenuation(r, f, "vertical"), f


def test_k_r_inversion_by_hand():
    """R = (A / (a L))**(1/b); A <= 0 -> 0; NaN stays NaN; below r_min -> 0."""
    a, b = itu_ab(38.0, "v", "ITU_2005")
    A = np.array([3.0, 0.0, -1.0, np.nan, 1e-3])
    R = rain_from_attenuation(A, 2.5, 38.0, "v", r_min=0.1)
    assert R[0] == pytest.approx((3.0 / (a[0] * 2.5)) ** (1 / b[0]), rel=1e-12)
    assert R[1] == 0 and R[2] == 0 and np.isnan(R[3]) and R[4] == 0


def test_2003_and_2005_tables_against_pycomlink():
    kr = pytest.importorskip("pycomlink.processing.k_R_relation")
    assert np.array_equal(ITU_2005, kr.ITU_table_2005)
    assert np.array_equal(ITU_2003, kr.ITU_table_2003)
    f = np.array([7.3, 13.0, 18.6, 23.1, 26.2, 38.4, 57.5, 73.2, 83.0])
    for table in ("ITU_2003", "ITU_2005"):
        for pol in ("v", "h"):
            a, b = itu_ab(f, pol, table)
            pa, pb = kr.a_b(f, pol, approx_type=table)
            assert np.allclose(a, pa, rtol=1e-12) and np.allclose(b, pb, rtol=1e-12)


def test_inversion_matches_pycomlink():
    kr = pytest.importorskip("pycomlink.processing.k_R_relation")
    rng = np.random.default_rng(0)
    A = rng.normal(2.0, 3.0, size=(3, 200))
    A[1, 5:9] = np.nan
    L = np.array([0.8, 2.3, 6.0])
    f = np.array([18.1, 23.0, 38.5])
    pol = np.array(["v", "h", "v"])
    for table in ("ITU_2003", "ITU_2005"):
        ours = rain_from_attenuation(A, L, f, pol, table=table, r_min=0.1)
        for i in range(3):
            ref = kr.calc_R_from_A(A[i], L[i], f_GHz=f[i], pol=pol[i], a_b_approximation=table, R_min=0.1)
            assert np.allclose(ours[i], ref, equal_nan=True, rtol=1e-12)
