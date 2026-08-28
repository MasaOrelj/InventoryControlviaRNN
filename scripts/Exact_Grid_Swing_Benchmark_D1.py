"""Exact, deterministic grid-based backward induction for the 1-D swing
option (see conversation) -- an independent, non-Monte-Carlo, non-regression
ground-truth benchmark for the LSM/RNN/Laguerre prices computed elsewhere in
this project. d=1 only: a full (Z,Y) grid is exactly the curse of
dimensionality LSM exists to avoid, so this doesn't generalize past d=1.

Exploits Z _|_ Y (independent OU diffusion and jump process, per
core/Electricity_Market_Model.py): the joint transition kernel factors as
P((Z',Y')|(Z,Y)) = P_Z(Z'|Z) * P_Y(Y'|Y), so the 2-D continuation expectation
reduces to the two-sided matrix product P_Z @ V @ P_Y^T, in
O(n_z^2*n_y + n_z*n_y^2) instead of O((n_z*n_y)^2) for a full joint kernel.

Same market/contract parameters as every other experiment in this project
(core.Electricity_Market_Model.HHKParams defaults, core.Swing.SwingContract
K=100, q_min=0, q_max=50, q_tilde=25, L=10, N_STEPS=50, T=1, r=0.02) for
direct comparability.

Fixes applied from the conversation's review:
  - xi_pmf (the Y innovation law) is built WITHOUT a pre-seeded atom at zero
    -- (k_up=0, k_dn=0) is an ordinary term of the (k_up,k_dn) double loop,
    avoiding the double-counting bug that would inflate P(no jump) from
    ~85.2% to ~92.6% (a uniform 0.5x under-weighting of every jump
    component -- verified exactly, not the ~0.54 first guessed).
  - The empty convolution (k_up=k_dn=0) is explicitly the identity (delta at
    0), not an all-zero array.
  - Final xi_pmf mass is ASSERTED close to 1 (tolerance looser than the
    1e-12 Poisson truncation, since the quadrature for the single-jump
    distribution and the FFT convolutions each add their own float noise),
    not silently renormalized -- a real deficit here would otherwise mask a
    too-small k_max or quadrature mass leaking off the grid.

Bug found DURING implementation (not in the original review): V_0 was
extracted via nearest-grid-point lookup at (z=0,y=0). Since V is highly
sensitive to y (S depends on y exponentially), whether 0 happens to land
exactly on a grid point is an arbitrary function of n_z/n_y, and this was
injecting resolution-dependent noise of the same order as genuine solver
error -- a convergence sweep over n_y showed non-monotonic swings of
~100+ units (0.3-0.4% of price) that tracked the nearest-point offset from
true zero, not overall resolution. Fixed with bilinear interpolation in
(z,y) at exactly (0,0); residual noise across resolutions dropped to ~6
units (~0.02% of price).

NOT implemented (unlike an earlier draft of this docstring claimed): the
exercise-region array. Only V_0 is returned; the backward induction discards
V[l] once Vnew is computed. Recovering the exercise boundary would need
storing V or Vnew's argmax(C[l], exercise) per (n,l) explicitly.

Grid range/resolution validated by convergence study (see conversation):
Y_LO,Y_HI=[-4,8] (the original suggestion) truncates real, payoff-relevant
mass in Y's heavy right tail -- V_0 kept rising as the range widened until
~[-8,16], where it stabilizes (clip mass ~1e-13). Current defaults
(N_Z=301, N_Y=2001, range=[-10,20]) give V_0 in the 30705-30709 band
across repeated resolution checks at that range; not pushed narrower or
wider only for runtime (already ~20-40s).

Run: python -m scripts.Exact_Grid_Swing_Benchmark_D1
"""
import numpy as np
from scipy.stats import norm

from core.Electricity_Market_Model import HHKParams, seasonality
from core.Swing import SwingContract

# ---------------------------------------------------------------- market/contract
PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
DELTA = MATURITY / N_STEPS
ALPHA = np.exp(-R * DELTA)

# ---------------------------------------------------------------- grid sizes
# Validated by convergence study (see conversation): [-4,8] truncated real,
# payoff-relevant mass in Y's heavy right tail (S~exp(Y) amplifies even small
# tail probabilities) -- V_0 kept rising as the range widened until ~[-8,16],
# where it stabilizes (clip mass ~1e-13). n_y=2001 at this range gives
# V_0=30705-30709 across repeated resolution checks, a ~0.02% residual band;
# not pushed further only for runtime (this call takes ~20-40s already).
N_Z = 301
N_Y = 2001
Y_LO, Y_HI = -10.0, 20.0   # asymmetric: up-jumps more frequent & larger than down
N_V_QUAD = 200            # Gauss-Legendre nodes for the V-integral
K_MAX_TAIL = 1e-12        # Poisson truncation tolerance
MASS_TOL = 1e-8           # xi_pmf total-mass assertion tolerance (looser than K_MAX_TAIL)

a_Z = np.exp(-PARAMS.kappa * DELTA)
s_Z = PARAMS.sigma * np.sqrt((1.0 - np.exp(-2.0 * PARAMS.kappa * DELTA)) / (2.0 * PARAMS.kappa))
a_Y = np.exp(-PARAMS.beta * DELTA)
sd_Z = s_Z / np.sqrt(1.0 - a_Z**2)


def make_edges(lo: float, hi: float, n: int) -> np.ndarray:
    """n grid points -> n+1 cell edges at the midpoints, with the two outer
    edges pushed to +-inf so the tail cells absorb everything beyond the grid
    (row-stochastic transition matrices by construction)."""
    centers = np.linspace(lo, hi, n)
    mids = (centers[:-1] + centers[1:]) / 2.0
    edges = np.concatenate([[-np.inf], mids, [np.inf]])
    return centers, edges


def build_P_Z() -> tuple[np.ndarray, np.ndarray]:
    z_grid, z_edges = make_edges(-6 * sd_Z, 6 * sd_Z, N_Z)
    m = a_Z * z_grid[:, None]                     # (N_Z,1) conditional means
    cdf_hi = norm.cdf((z_edges[None, 1:] - m) / s_Z)
    cdf_lo = norm.cdf((z_edges[None, :-1] - m) / s_Z)
    P_Z = cdf_hi - cdf_lo
    row_sums = P_Z.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), f"P_Z not row-stochastic: max dev {np.abs(row_sums-1).max():.2e}"
    return z_grid, P_Z


def gauss_legendre_on(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    xt = 0.5 * (x + 1) * (b - a) + a
    wt = w * 0.5 * (b - a)
    return xt, wt


def single_jump_cell_probs(mu: float, edges: np.ndarray) -> np.ndarray:
    """P(V*J in each cell of `edges`), V=exp(-beta*U), U~Unif(0,Delta), J~Exp(mean mu),
    via Gauss-Legendre quadrature over v in [a_Y, 1] of the exact conditional
    Exponential(mean mu*v) cell probabilities."""
    v_nodes, v_weights = gauss_legendre_on(a_Y, 1.0, N_V_QUAD)
    f_V = 1.0 / (PARAMS.beta * DELTA * v_nodes)     # (N_V_QUAD,)
    lo, hi = edges[:-1], edges[1:]
    lo_pos = np.maximum(lo, 0.0)
    hi_pos = np.where(np.isposinf(hi), np.inf, np.maximum(hi, 0.0))

    def F(x_pos):
        # 1 - exp(-x/(mu*v)) for x finite, 1.0 for x = +inf
        out = np.empty((len(x_pos), N_V_QUAD))
        finite = np.isfinite(x_pos)
        out[finite] = 1.0 - np.exp(-x_pos[finite, None] / (mu * v_nodes[None, :]))
        out[~finite] = 1.0
        return out

    F_hi = F(hi_pos)
    F_lo = F(lo_pos)
    F_lo[lo <= 0] = 0.0
    probs = (v_weights[None, :] * f_V[None, :] * (F_hi - F_lo)).sum(axis=1)
    return probs


def poisson_k_max(rate: float, tol: float) -> int:
    from scipy.stats import poisson
    k = 0
    while poisson.sf(k, rate) >= tol:
        k += 1
    return k + 1


def build_xi_pmf(dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Innovation law of xi = (sum of up-jump V*J) - (sum of down-jump V*J),
    on a fixed regular grid of spacing dy. Returns (values, pmf)."""
    # Single-jump distributions on a wide, dy-spaced, one-sided grid; tail
    # beyond the grid is folded into the last cell (mass-conserving, mirrors
    # build_P_Z's tail handling).
    g_max = 30.0
    n_g = int(round(g_max / dy)) + 1
    g_centers = np.arange(n_g) * dy
    g_edges = np.concatenate([[0.0], g_centers[:-1] + dy / 2.0, [np.inf]])
    # (n_g+1 edges for n_g cells: [0, c0+dy/2), [c0+dy/2, c1+dy/2), ..., [.., inf))
    # rebuild edges to have exactly n_g cells:
    g_edges = np.concatenate([[0.0], (g_centers[:-1] + g_centers[1:]) / 2.0, [np.inf]])

    g_up = single_jump_cell_probs(PARAMS.mu_up, g_edges)
    g_down = single_jump_cell_probs(PARAMS.mu_down, g_edges)
    assert abs(g_up.sum() - 1.0) < 1e-8, f"g_up mass deficit {abs(g_up.sum()-1.0):.2e}"
    assert abs(g_down.sum() - 1.0) < 1e-8, f"g_down mass deficit {abs(g_down.sum()-1.0):.2e}"

    k_max_up = poisson_k_max(PARAMS.lam_up * DELTA, K_MAX_TAIL)
    k_max_down = poisson_k_max(PARAMS.lam_down * DELTA, K_MAX_TAIL)

    from scipy.stats import poisson as poisson_dist

    # Accumulate on a symmetric wide grid indexed by integer offsets from 0
    # (in units of dy), wide enough for k_max_up-fold(+) and k_max_down-fold(-) sums.
    half_width = n_g * max(k_max_up, k_max_down) + 1
    total_len = 2 * half_width + 1
    zero_idx = half_width
    xi_pmf = np.zeros(total_len)

    identity = np.zeros(1)
    identity[0] = 1.0   # delta at 0 -- explicit identity for the empty convolution

    def self_convolve(g: np.ndarray, k: int) -> np.ndarray:
        if k == 0:
            return identity.copy()
        out = g.copy()
        for _ in range(k - 1):
            out = np.convolve(out, g)
        return out

    for k_up in range(k_max_up + 1):
        w_up = poisson_dist.pmf(k_up, PARAMS.lam_up * DELTA)
        if w_up < 1e-14:
            continue
        h_up = self_convolve(g_up, k_up)   # supported on [0, k_up*g_max], index 0 == value 0
        for k_dn in range(k_max_down + 1):
            w = w_up * poisson_dist.pmf(k_dn, PARAMS.lam_down * DELTA)
            if w < 1e-14:
                continue
            h_dn = self_convolve(g_down, k_dn)   # same, will be reflected (negated)
            h = np.convolve(h_up, h_dn[::-1])    # convolving with the reflected kernel
            # h's index 0 corresponds to value: 0 (from h_up) - (len(h_dn)-1)*dy (from reflect)
            offset = -(len(h_dn) - 1)
            start = zero_idx + offset
            xi_pmf[start:start + len(h)] += w * h

    total_mass = xi_pmf.sum()
    assert abs(total_mass - 1.0) < MASS_TOL, f"xi_pmf mass deficit {abs(total_mass-1.0):.2e} (widen k_max/grid)"
    xi_pmf /= total_mass   # correct only the tiny quadrature-level residual, already asserted small

    values = (np.arange(total_len) - zero_idx) * dy
    return values, xi_pmf


def build_P_Y(y_grid: np.ndarray, xi_values: np.ndarray, xi_pmf: np.ndarray) -> np.ndarray:
    dy = y_grid[1] - y_grid[0]
    n_y = len(y_grid)
    P_Y = np.zeros((n_y, n_y))
    y_lo, y_hi = y_grid[0], y_grid[-1]

    nz = np.nonzero(xi_pmf > 1e-16)[0]
    xi_v = xi_values[nz]
    xi_p = xi_pmf[nz]

    for i in range(n_y):
        target = a_Y * y_grid[i] + xi_v
        target = np.clip(target, y_lo, y_hi)
        pos = (target - y_lo) / dy
        j = np.floor(pos).astype(int)
        j = np.clip(j, 0, n_y - 2)
        w = pos - j
        np.add.at(P_Y[i], j, xi_p * (1 - w))
        np.add.at(P_Y[i], j + 1, xi_p * w)

    row_sums = P_Y.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"P_Y not row-stochastic: max dev {np.abs(row_sums-1).max():.2e}"
    P_Y /= row_sums[:, None]   # renormalize away the tiny clipping-at-boundary residual
    return P_Y


def backward_induction(z_grid: np.ndarray, y_grid: np.ndarray, P_Z: np.ndarray, P_Y: np.ndarray) -> float:
    n_z, n_y = len(z_grid), len(y_grid)
    L = CONTRACT.L
    N = N_STEPS

    t = np.arange(N + 1) * DELTA
    f_t = seasonality(t, PARAMS)
    S = np.exp(f_t[:, None, None] + z_grid[None, :, None] + y_grid[None, None, :])   # (N+1, n_z, n_y)
    Qpay = (CONTRACT.q_max - CONTRACT.q_tilde) * np.maximum(S - CONTRACT.K, 0.0) + \
           (CONTRACT.q_tilde - CONTRACT.q_min) * np.maximum(CONTRACT.K - S, 0.0)

    def levels(n: int) -> range:
        return range(max(0, L - n), L + 1)

    V = {l: np.zeros((n_z, n_y)) for l in range(L + 1)}
    for l in levels(N):
        V[l] = Qpay[N] if l >= 1 else np.zeros((n_z, n_y))

    P_Y_T = P_Y.T
    for n in range(N - 1, -1, -1):
        C = {}
        for l in levels(n + 1):
            C[l] = ALPHA * (P_Z @ V[l] @ P_Y_T)
        Vnew = {}
        for l in levels(n):
            if l == 0:
                Vnew[0] = C[0]
            else:
                exercise = Qpay[n] + C[l - 1]
                Vnew[l] = np.maximum(C[l], exercise)
        V = Vnew

    # V_0 at (z=0, y=0, L): nearest grid point to z=0,y=0 (grids are centered so this is exact/near-exact)
    # Bilinear interpolation at EXACTLY (z=0, y=0) -- nearest-grid-point
    # lookup was verified to introduce resolution-dependent noise of the
    # same order as the effect under study, since V is highly sensitive to y
    # (exponential price dependence): whether z=0/y=0 happens to land exactly
    # on a grid point depends on the arbitrary choice of n_z/n_y, and
    # nearest-point lookup picked up that arbitrariness as if it were solver
    # error (see conversation).
    iz1 = int(np.searchsorted(z_grid, 0.0))
    iz0 = max(iz1 - 1, 0)
    iz1 = min(iz1, len(z_grid) - 1)
    if iz0 == iz1:
        wz = 0.0
    else:
        wz = (0.0 - z_grid[iz0]) / (z_grid[iz1] - z_grid[iz0])

    iy1 = int(np.searchsorted(y_grid, 0.0))
    iy0 = max(iy1 - 1, 0)
    iy1 = min(iy1, len(y_grid) - 1)
    if iy0 == iy1:
        wy = 0.0
    else:
        wy = (0.0 - y_grid[iy0]) / (y_grid[iy1] - y_grid[iy0])

    v00, v01 = V[L][iz0, iy0], V[L][iz0, iy1]
    v10, v11 = V[L][iz1, iy0], V[L][iz1, iy1]
    v0_interp = (1 - wz) * ((1 - wy) * v00 + wy * v01) + wz * ((1 - wy) * v10 + wy * v11)
    return v0_interp, 0.0, 0.0


if __name__ == "__main__":
    print(f"Delta={DELTA}, a_Z={a_Z:.4f}, s_Z={s_Z:.4f}, a_Y={a_Y:.4f}")

    z_grid, P_Z = build_P_Z()
    print(f"P_Z built: {P_Z.shape}, row-stochastic OK")

    y_grid, y_edges = make_edges(Y_LO, Y_HI, N_Y)
    dy = y_grid[1] - y_grid[0]
    print(f"y_grid: [{Y_LO},{Y_HI}], dy={dy:.5f}")

    xi_values, xi_pmf = build_xi_pmf(dy)
    p_no_jump_true = np.exp(-(PARAMS.lam_up + PARAMS.lam_down) * DELTA)
    mass_at_zero = xi_pmf[np.argmin(np.abs(xi_values))]
    print(f"xi_pmf built: {len(xi_pmf)} points, total mass={xi_pmf.sum():.10f}")
    print(f"  P(no jump), true={p_no_jump_true:.6f} vs xi_pmf mass at y=0 cell={mass_at_zero:.6f} (not exactly equal -- cell width matters)")

    P_Y = build_P_Y(y_grid, xi_values, xi_pmf)
    print(f"P_Y built: {P_Y.shape}, row-stochastic OK")

    v0, z_used, y_used = backward_induction(z_grid, y_grid, P_Z, P_Y)
    print(f"\nV_0(z={z_used:.4f}, y={y_used:.4f}, L={CONTRACT.L}) = {v0:.4f}")
