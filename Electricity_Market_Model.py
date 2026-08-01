"""HHK electricity spot model.

    S_t  = exp( f(t) + Z_t + Y_t )
    dZ_t = -kappa Z_t dt + sigma dW_t                     (OU: fundamental diffusion)
    dY_t = -beta Y_{t-} dt + J_t dN_t - J'_t dN'_t         (mean-reverting spikes)

`N` (up-jumps) and `N'` (down-jumps) are independent Poisson processes with
intensities `lam_up` / `lam_down`; jump sizes are `J ~ Exp(mean mu_up)` and
`J' ~ Exp(mean mu_down)` (both positive-valued, the down-jump is subtracted).
Setting `lam_down = 0` recovers the single-sided (positive-jump-only) model.

`Z`, `Y`, `S` are simulated on an equidistant grid via their exact discretizations,
vectorized across paths and (independent) market dimensions; the only Python loop
is forward over time steps.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HHKParams:
    """HHK model parameters. Mean jump size is mu directly (mu is passed straight through
    as numpy's exponential `scale` parameter, i.e. E[J] = mu, not 1/mu)."""

    kappa: float          # Z mean-reversion speed (how fast the OU factor pulls back to 0)
    sigma: float           # Z volatility (size of the everyday diffusive wiggles)
    beta: float             # Y mean-reversion speed (how fast a price spike decays away)
    lam_up: float           # Poisson arrival rate of upward price spikes (jumps/unit time)
    mu_up: float            # mean size of an up-jump (E[J] = mu_up)
    lam_down: float = 0.0   # arrival rate of downward spikes; 0.0 = single-sided model (default)
    mu_down: float = 1.0    # mean size of a down-jump (irrelevant while lam_down=0)
    f_level: float = 100.0  # seasonality "base price" level, enters as ln(f_level)
    f_amp: float = 0.5      # seasonality oscillation amplitude
    f_period: float = 1.0   # seasonality period (e.g. 1.0 = one cycle per year if t is in years)


# Reference parameters from the thesis / HHK (single-sided jumps).
DEFAULT_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=200.0, lam_up=5.0, mu_up=0.4,
)


@dataclass(frozen=True)
class HHKPaths:
    """Simulated HHK paths. Arrays have shape (n_steps+1, n_paths, n_dims).

    Axis 0 = time step, axis 1 = Monte Carlo path, axis 2 = market dimension
    (e.g. independent delivery hours/locations sharing the same f(t))."""

    t: np.ndarray       # (n_steps+1,) the time grid itself, e.g. [0, dt, 2*dt, ..., maturity]
    Z: np.ndarray        # OU factor path
    Y: np.ndarray        # spike factor path
    S: np.ndarray        # spot price path = exp(f(t) + Z + Y)


def seasonality(t: np.ndarray, params: HHKParams) -> np.ndarray:
    """f(t) = ln(f_level) + f_amp * cos(2*pi*t / f_period)."""
    return np.log(params.f_level) + params.f_amp * np.cos(2.0 * np.pi * t / params.f_period)


def simulate_hhk(
    rng: np.random.Generator,
    params: HHKParams,
    n_paths: int,
    n_steps: int,
    maturity: float,
    n_dims: int = 1,
    z0: float = 0.0,
    y0: float = 0.0,
) -> HHKPaths:
    """Simulate `n_dims` independent (Z, Y) -> S components sharing one f(t)."""
    dt = maturity / n_steps
    t_grid = np.linspace(0.0, maturity, n_steps + 1)

    # Pre-allocate the full path arrays and set the (deterministic) starting point.
    Z = np.empty((n_steps + 1, n_paths, n_dims))
    Y = np.empty((n_steps + 1, n_paths, n_dims))
    Z[0] = z0
    Y[0] = y0

    # Constants for the *exact* discretization of the OU process Z (CLAUDE.md formula):
    #   Z_{n+1} = ou_decay * Z_n + ou_std * eps,   eps ~ N(0,1)
    # These don't depend on n, so compute them once outside the time loop.
    ou_decay = np.exp(-params.kappa * dt)
    ou_std = params.sigma * np.sqrt((1.0 - np.exp(-2.0 * params.kappa * dt)) / (2.0 * params.kappa))
    # Decay factor for the spike factor Y between two grid points.
    spike_decay = np.exp(-params.beta * dt)

    # Only loop over time (forward, one step at a time); every path and every
    # market dimension is updated together as a vectorized array operation.
    for n in range(n_steps):
        eps = rng.standard_normal(size=(n_paths, n_dims))
        Z[n + 1] = ou_decay * Z[n] + ou_std * eps

        # Compound-Poisson jump contribution arriving inside (t_n, t_n+1]; up and
        # down spikes are independent Poisson processes, so just sum both (down
        # jumps are subtracted). See _jump_increment for how this is vectorized.
        jump_up = _jump_increment(rng, params.lam_up, params.mu_up, params.beta, dt, n_paths, n_dims)
        jump_down = _jump_increment(rng, params.lam_down, params.mu_down, params.beta, dt, n_paths, n_dims)
        Y[n + 1] = spike_decay * Y[n] + jump_up - jump_down

    # Spot price S_t = exp(f(t) + Z_t + Y_t). f depends only on time, so broadcast
    # it across the (path, dim) axes with [:, None, None].
    f = seasonality(t_grid, params)
    S = np.exp(f[:, None, None] + Z + Y)

    return HHKPaths(t=t_grid, Z=Z, Y=Y, S=S)


def _jump_increment(
    rng: np.random.Generator,
    lam: float,
    mu: float,
    beta: float,
    dt: float,
    n_paths: int,
    n_dims: int,
) -> np.ndarray:
    """Sum_{i=1..N} exp(-beta*(dt - U_i)) * J_i for N ~ Poisson(lam*dt), U_i ~ Unif(0,dt),
    J_i ~ Exp(mean mu). Vectorized across paths/dims via a per-step jump-count cap.

    The tricky part: each (path, dim) can have a *different* number of jumps N in
    this step, so we can't just make one array of jump sizes. Instead we pick
    max_count = the largest N seen this step, build arrays of that width for
    every (path, dim), and mask off the "slots" beyond each cell's own N so they
    contribute 0. This avoids a Python-level loop over individual paths."""
    # How many jumps land in (t, t+dt] for every (path, dim) cell.
    counts = rng.poisson(lam * dt, size=(n_paths, n_dims))
    max_count = int(counts.max(initial=0))
    if max_count == 0:
        # Fast path: nothing jumped anywhere this step (very common when lam*dt is
        # small, e.g. lam_down=0 for the single-sided model) -- skip the rest.
        return np.zeros((n_paths, n_dims))

    # slots = [0, 1, ..., max_count-1]; mask[p, d, k] is True iff cell (p, d) has
    # at least k+1 jumps, i.e. "slot k" is a real jump for that cell and not padding.
    slots = np.arange(max_count)
    mask = slots[None, None, :] < counts[:, :, None]

    # Draw a jump arrival time and size for *every* slot of every cell (including
    # padding slots -- wasteful but simple, and cheap since max_count is tiny).
    arrival = rng.uniform(0.0, dt, size=(n_paths, n_dims, max_count))
    size = rng.exponential(scale=mu, size=(n_paths, n_dims, max_count))
    # Each jump decays by exp(-beta * (time remaining until t+dt)) before we reach t+dt.
    decayed = np.exp(-beta * (dt - arrival)) * size
    # Zero out padding slots, then sum the real jumps per cell.
    return np.where(mask, decayed, 0.0).sum(axis=-1)
