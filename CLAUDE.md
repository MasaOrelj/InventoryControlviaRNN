# CLAUDE.md — Project Brief for Claude Code

Standing brief for this repository. Read it in full before writing or changing code. It is the
single source of truth for scope, the model, and the working conventions. Update it as decisions
actually settle — this is a living document.

## Project

Master's thesis, MSc Stochastics and Financial Mathematics, University of Amsterdam
(Korteweg–de Vries Institute). Author: Maša Orelj. Supervisor: dr. Sven Karbach.

The thesis compares two ways of choosing the set of admissible approximation functions H in the
**Least-Square Monte Carlo (LSM)** method for a stochastic inventory-control problem:

1. **Preselection of basis functions** (classical Longstaff–Schwartz): a fixed, chosen family of
   basis functions (e.g. polynomials).
2. **Randomized neural network (RNN)**: a single-hidden-layer feedforward network whose hidden
   weights are drawn randomly and then fixed, so only the output weights are fitted by least
   squares.

Both are solved with **ridge (Tikhonov) regression**. The application is the valuation and optimal
exercise of an **electricity-market swing option** under a spot model with price spikes.

Research question, one line: *does the RNN approach give a better accuracy/cost trade-off than
preselected polynomial bases for LSM on electricity swing options, and how does that comparison
change with the state dimension?*

## Scope right now

**Swing option only.** Do NOT build the forward-curve or battery/BESS model — that is a later,
separate application. (An earlier version of this file was written from the wrong draft and
described a battery/forward-curve setup; disregard any such references.)

We already have working code. The current task is to **simplify and restructure it, file by file**
— prefer simplifying existing code to rewriting from scratch.

## Design principle (the main simplification)

Build **one LSM engine** that is agnostic to the three choices below; inject them rather than
branching on them. Getting this right collapses the whole comparison into swapping small parts.

### The three configurable knobs

1. **State input for the regression.**
   - `(Z_n, Y_n, I_n)` — the full Markov state (theoretical, F_n-measurable case), or
   - `(S_n, I_n)` — spot price and inventory only (the practically executable case).

   `S_n = exp(f(t_n) + Z_n + Y_n)` reveals only `Z_n + Y_n`, so it is *not* Markov on its own. The
   `(S_n, I_n)` regression optimizes over price-measurable policies only and yields a valid but
   possibly looser lower bound for `V_0`. Both must be runnable, because reducing to `S` changes
   the preselection-vs-RNN comparison (it lowers the preselection cost and removes the richer
   structure the random features exploit).

2. **Inventory handling in the regression.**
   - **Joint:** regress on functions of `(s, i)` (inventory enters as a regressor), or
   - **Per-level:** fix each inventory grid point `i` and run a separate regression on functions of
     `s` alone (the Boogert–de Jong storage convention).

3. **Approximation architecture / feature map.**
   - **Preselection:** a polynomial family, plus at least one other family, for comparison, or
   - **RNN:** one hidden layer, `K−1` random-but-fixed nodes, constant feature appended.

Plus: **ridge** is always on with a tunable `λ`, and the market **dimension `d`** is configurable
(1-D is the main real-world case; multi-D drives the scalability comparison). For `d > 1`, the
per-dimension swing gains are combined by an **injectable aggregation function** (sum, max, ... —
see "Swing option specification" and `Payoff_Aggregation.py`), same inject-don't-branch principle.

## Repository layout (swing-only)

```
Electricity_Market_Model.py  # HHK electricity model: simulate (Z,Y) factors -> spot S; 1-D and multi-D
Swing.py              # swing payoff Q, reward, terminal, admissible sets D_n, inventory map
Payoff_Aggregation.py # multi-D reward aggregation (sum, max, ...), injected into Swing.py's reward
Basis_Functions.py    # feature maps: polynomial family (+ one other), RNN random features, constant
Regression.py         # backward-induction LSM engine + ridge solver + policy/value
experiments.py        # run configs: dimension d, (Z,Y)|S input, joint|per-level inventory, basis
```

Files may merge or split later; this is a starting shape, not a commitment.

## Model specification — electricity market (Hambly–Howison–Kluge)

Continuous-time spot model, discretized to an equidistant grid `0 = t_0 < ... < t_N = T`,
`Δ = T/N`:

```
S_t  = exp( f(t) + Z_t + Y_t )
dZ_t = -κ Z_t dt + σ dW_t                 (OU: fundamental diffusion)
dY_t = -β Y_{t-} dt + J_t dN_t            (mean-reverting spikes)
```

`N_t` is Poisson with intensity `λ`; jump sizes `J ~ Exp(mean μ)` (positive jumps only).

Exact discretizations (both factors have closed forms):

```
Z_{n+1} = e^{-κΔ} Z_n + σ · sqrt( (1 - e^{-2κΔ}) / (2κ) ) · ε_{n+1},   ε ~ N(0,1) i.i.d.
Y_{n+1} = e^{-βΔ} Y_n + Σ_{i=1..N^Δ} e^{-β(t_{n+1} - τ_i)} J_{τ_i},     N^Δ ~ Poisson(λΔ)
```

`τ_i` = jump times in `(t_n, t_{n+1}]`.

**Multi-dimensional market:** a vector of independent `(OU, spike)` pairs sharing the *same*
seasonality `f(t)` (interpret as a portfolio over delivery hours/locations/maturities). The
regression state dimension grows with the number of such components.

Reference parameters (from the thesis / HHK):

```
κ = 7.0,  σ = 1.4,  β = 200.0,  λ = 5.0,  μ = 0.4,  Z_0 = Y_0 = 0
T = 2.0,  N = 400,  M = 20000 paths,  f(t) = ln(100) + 0.5 · cos(2π t)
```

## Swing option specification

- Inventory `I_n` = remaining swing rights, `I_n ∈ {0,1,...,L}`, `I_0 = L`.
- Action space `A = {0,1}` (1 = use a swing right, 0 = buy base load).
- Admissible sets: `D_n(i) = {0,1}` if `i ≥ 1`, else `{0}`.
- Inventory transition: `i_{n+1} = i_n − a` — always lands on the grid, so **no interpolation for
  swing** (interpolation is a continuous-inventory / BESS concern only).
- **Per-step inventory grid.** At most one swing per date, so not every level in `{0,...,L}` is
  reachable at every `n`: `𝓘(n) = {max(0, L−n), ..., L}`, `H_n = min(n, L) + 1`. `𝓘(0) = {L}` (fixed
  start); the grid only reaches the full `{0,...,L}` once `n ≥ L`. The **joint** regression knob
  iterates over this per-step grid, not a fixed-size `{0,...,L}` every step.
- One-step gain on a swing, per market dimension `j = 1..d`:
  `Q(S_n^{(j)}) = max{ (S_n^{(j)} − K)(q_max − q̃),  (K − S_n^{(j)})(q̃ − q_min),  0 } ≥ 0`.
- **Multi-dimensional reward (`d > 1`).** One shared swing right across the whole portfolio — the
  action space, inventory, and Bellman structure below are otherwise unchanged from `d = 1`. Total
  gain is `Q_total(S_n) = aggregate( Q(S_n^{(1)}), ..., Q(S_n^{(d)}) )` for an **injectable**
  `aggregate` function. Implement both `sum` and `max` in `Payoff_Aggregation.py` (more may be added
  later); selected as a config choice in `experiments.py`, never branched on inside the engine.
  `d = 1` reduces trivially to `Q_total = Q(S_n)`.
- Reward: `r_n(x, a) = Q_total(s)` if `a = 1`, else `0`, with `s = exp(f(t_n) + z + y)` (per
  dimension; `d=1` recovers the scalar case from earlier drafts of this file).
- Terminal: `g_N(x) = 1{i ≥ 1} · Q_total(s)`.
- Bellman (for `i ≥ 1`):
  `V_n(z,y,i) = max{ α E[V_{n+1}(·,·,i) | Z_n=z, Y_n=y],  Q_total(s) + α E[V_{n+1}(·,·,i−1) | Z_n=z, Y_n=y] }`,
  with `V_n(z,y,0) = 0`. Constant strike `K` (shared or per-dimension), no global volume bounds,
  discount `α ∈ (0,1]`.
- **Zero-payoff exercise guard (swing-specific decision rule, on top of the generic argmax in "LSM
  algorithm").** In the true value functions `V_n(·,·,i) ≥ V_n(·,·,i−1)` always (extra optionality
  is never worse), so `c_n^0(x) ≥ c_n^1(x)` always holds and a bare argmax would never prefer
  swinging when `Q_total(s)=0` anyway. But `c̃_n^0` and `c̃_n^1` are two *separately fitted*
  regressions with no constraint enforcing that ordering, so under sampling noise `c̃_n^1(x) >
  c̃_n^0(x)` can happen by chance — a bare argmax would then spend a swing right for zero measurable
  benefit. To guard against this, the swing decision requires **both** `Q_total(s) > 0` **and**
  `Q_total(s) + c̃_n^1(x) > c̃_n^0(x)` before choosing `a=1`; otherwise `a=0`. (`Q_total(s)=0` ⇒
  `a=0` unconditionally, which subsumes the smallest-index tie-break rule in that case.)

## LSM algorithm

1. Simulate `M` paths of the exogenous factors `(Z, Y)` -> spot `S` (1-D or multi-D). Split into
   training and evaluation halves, `M_t = M_e = M/2`.
2. Terminal cash flows `p_N^{m,h} = g_N(x_N^{m,h})` over the inventory grid `𝓘(N)`.
3. Backward `n = N−1, ..., 1`:
   a. Fit the continuation value, on **training paths only** (`m = 1..M_t`) — never the evaluation
      paths, that's step 4's job:
      - **Joint** knob: for each admissible action `a`, one ridge regression of the discounted
        rolled-back cash flow `α · p̄_{n+1}^m( Ψ_n^I(i, a) )` on the chosen feature map of the state
        (state = `(Z_n,Y_n,I_n)` or `(S_n,I_n)`, per the state-input knob), pooling all `i ∈ 𝓘(n)`
        together (`i` enters as a regressor) → one `c̃_n^a` per action.
      - **Per-level** knob: for each admissible action `a` *and* each `i ∈ 𝓘(n)`, a separate
        regression on the price alone (`Z_n,Y_n` or `S_n`, per the state-input knob, no `i` input
        since it's fixed per regression) → `c̃_n^{a,i}` per (action, inventory level) pair.
   b. For **every** sample `x_n^{m,h}`, `m = 1..M` (training *and* evaluation together — these
      propagated cash flows feed both the next backward step and, eventually, `Ṽ_0`), pick
      `ã = argmax_a ( r_n + c̃_n^a )`, **ties broken toward the smallest action index** (i.e. prefer
      `a=0`, don't-swing, over `a=1` on an exact tie), and set
      `p_n^{m,h} = r_n(x, ã) + α · p̄_{n+1}^m( Ψ_n^I(i, ã) )`. For swing specifically this argmax is
      further refined by the zero-payoff exercise guard in "Swing option specification" — don't
      implement a bare argmax there without it.
4. At `n = 0` (fixed initial state), estimate `c̃_0^a` as the plain Monte Carlo mean over the
   **evaluation paths only**, and report `Ṽ_0 = max_a ( r_0 + c̃_0^a )`.

`Ṽ_0` is a lower bound for the true `V_0`. The point of the study is comparing `Ṽ_0` (accuracy) and
runtime across the knob settings above.

## Ridge regression

With `Φ_n ∈ R^{K × (M_t·H_n)}` (columns are `Φ(x_n^{m,h})`) and target vector `y_n`:

```
β̃_n^a = argmin_β  (1 / (M_t·H_n)) · || Φ_n^T β − y_n ||²  +  λ · || I_0 β ||²
β̃_n^a = ( Φ_n Φ_n^T + λ·(M_t·H_n)·I_0 )^{-1} Φ_n y_n
```

- The `1/(M_t·H_n)` normalization makes `λ` comparable across time steps with different grid sizes
  `H_n` — keep it.
- `I_0` is the identity **except a 0 on the constant-feature position** (penalize every coefficient
  except the constant). The constant is a level term, never the source of ill-conditioning, so it
  stays unpenalized.
- Solve via the symmetric positive-definite normal equations (Cholesky), not an explicit inverse.
- **Settled convention:** every feature map in `Basis_Functions.py` puts the constant feature in the
  **last** column of `Phi` (matching the RNN formula below). `Regression.py`'s `ridge_regression`
  relies on this to build `I_0` — it always zeroes the last diagonal entry, nothing else.

## Feature maps

**Preselection (Approach 1).** Univariate family `(φ_j)`, `φ_0 ≡ 1` (e.g. monomials `φ_j = x^j`, or
an orthogonal family — Laguerre / Hermite / Legendre / Chebyshev). Multivariate features are tensor
products with a total-degree cap `Σ λ_i ≤ l` to limit the count `C(l+q, q)`; this count grows fast
with dimension `q`, which is the weakness the RNN avoids. Implement at least the polynomial family
plus one other, for comparison.

**RNN (Approach 2).** `K−1` hidden units. Draw `Θ ∈ R^{(K−1)×q}` and `b ∈ R^{K−1}` once from a
chosen distribution and **fix them for the entire LSM run** (do not resample per step). Feature map,
with constant appended:

```
Φ(x) = ( σ(θ_1·x + b_1), ..., σ(θ_{K−1}·x + b_{K−1}), 1 ) ∈ R^K
```

`σ` a component-wise activation (e.g. tanh / ReLU). Only the output weights `β` are fitted (by the
ridge step above), per time step and per action. Thesis default `K−1 = 20`. `Φ_n` is assembled once
per time step (it does not depend on the action); the regression is refit per admissible action.

## Correctness gates and gotchas

- **Compare on identical paths.** Every knob setting within one experiment must run on the *same*
  simulated paths. Simulate factors from an explicitly passed, seeded RNG, decoupled from the
  regression method, so preselection vs RNN (and `(Z,Y)` vs `S`) differ only in the method, not the
  randomness.
- **Fix the RNN weights once** (see above). Resampling per step is a bug.
- **Ridge constant handling:** `I_0` has a 0 on the constant position; a wrong `λ` scaling or
  penalizing the constant will quietly change results.
- **Out-of-sample always:** fit continuation weights on training paths, report `Ṽ_0` on the held-out
  evaluation paths. Never report an in-sample value.
- **Fit vs. propagate are different path sets, on purpose:** step 3a's regression *fit* uses training
  paths only; step 3b's action selection and cash-flow propagation runs over *all* `M` paths (both
  halves), because evaluation paths still need `p_n^{m,h}` carried backward to produce `Ṽ_0` at
  `n=0`. Don't "simplify" this into fitting on all paths — that reintroduces in-sample leakage.
- **On-grid inventory for swing:** `i − a` stays in `{0,...,L}`, so no interpolation. Keep the engine
  open to interpolation later for continuous inventory, but do not add it for swing.
- **No ITM-only filtering.** Considered and rejected: `Q_total(s)=0` only at the measure-zero event
  `S_n=K` exactly (continuous price process), so essentially every path is "in the money" for swing
  — unlike a one-sided American put/call, filtering wouldn't meaningfully shrink the training set.
  Fit `c̃_n^a` on the *full* training sample, matching the thesis's `Eq. LSM_Numerical` literally.
- `Ṽ_0` is a lower bound; a better policy gives a higher `Ṽ_0`. Rankings between architectures are the
  deliverable, not the absolute price.

## Conventions

- Python 3.11+. Core stack: `numpy`, `scipy`, `matplotlib`, `pytest`.
- **Vectorize across Monte Carlo paths;** the natural explicit loop is backward over time steps `n`
  (and over admissible actions `a`), with paths and grid as array axes.
- **Reproducibility is non-negotiable** (the comparison depends on it): use a
  `numpy.random.Generator` that is seeded and passed in — no global `np.random`, no hidden seeds.
- Keep every feature map behind one common interface (e.g. `build_features(state) -> Φ`), so
  polynomial / other-family / RNN are drop-in swappable; likewise for the state-input and
  inventory-mode choices.
- Type hints and short docstrings that state array shapes and axis meaning (time, path, factor,
  action, inventory).
- Small, focused modules matching the layout above; add a `pytest` test alongside each numerical
  component.

## Working style

- Keep it simple; this is research code a supervisor and examiner must be able to read and trust.
  Prefer clarity over cleverness.
- We proceed **file by file**, shaping and simplifying together. When a modelling choice is
  ambiguous, state the assumption in a comment and flag it rather than silently deciding.
- Update this file whenever a convention or design decision settles, so it stays the single source
  of truth. If it grows past ~200 lines, split details into `.claude/rules/`.

## Notation map (math <-> code)

```
S_n            spot price at step n
Z_n, Y_n       OU factor, spike factor  (the Markov pair)
f(t)           deterministic seasonality
I_n, L         inventory (remaining swing rights), initial rights
𝓘(n), H_n      per-step reachable inventory grid, its size (= min(n,L)+1)
a, D_n(i)      action in {0,1}, admissible action set
d              number of market dimensions (portfolio components); d=1 is the base case
Q(S_n)         per-swing gain, single dimension
Q_total        multi-D gain = aggregate(Q(S_n^(1)),...,Q(S_n^(d))); sum or max, d=1 reduces to Q(S_n)
K  (strike)    option strike price          -- overloaded, distinct from ↓
K  (features)  number of basis functions / (hidden nodes + 1)
Φ, β           feature map, fitted coefficients
c̃_n^a          approximated continuation value
Ṽ_0            approximated time-0 value (a lower bound)
M, M_t, M_e    total / training / evaluation paths
λ  (ridge)     ridge penalty                -- overloaded, distinct from ↓
λ  (jumps)     Poisson jump intensity
α              one-step discount factor
q              state-space dimension seen by the regression (d + k)
```
