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
per-dimension swing gains are combined by an **injectable aggregation function** — settled:
`max_aggregation` is the thesis's actual specified payoff (`sum_aggregation` stays implemented and
injectable, but isn't the portfolio's actual payoff; don't default to it — see "Swing option
specification" and `Payoff_Aggregation.py`), same inject-don't-branch principle otherwise.

## Repository layout (swing-only)

```
core/
  Electricity_Market_Model.py  # HHK electricity model: simulate (Z,Y) factors -> spot S; 1-D and multi-D
  Swing.py              # swing payoff Q, reward, terminal, admissible sets D_n, inventory map
  Payoff_Aggregation.py # multi-D reward aggregation (sum, max, ...), injected into Swing.py's reward
  Basis_Functions.py    # feature maps: polynomial family (+ one other), RNN random features, constant
  Regression.py         # backward-induction LSM engine + ridge solver + policy/value
scripts/               # experiment run configs + Experiment_Log.py (CSV logging shared by them)
tests/                 # pytest suite, one file per core module
R_Tables/              # R table-generation scripts and their .tex output
```

The 5 files in `core/` are the actual mechanism -- the market model, the option's payoff/exercise
rule, the feature maps being compared, and the LSM engine that fits and evaluates policies with
them. Everything in `scripts/` builds experiments *on top of* that mechanism (market/contract
calibration, which dimensions/bases to compare, the repeated-seed methodology) without adding new
algorithmic content of its own. Imports elsewhere in the repo therefore go through `core.X` (e.g.
`from core.Regression import fit_policy`) and, for the shared CSV logger, `scripts.Experiment_Log`.

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

Reference parameters, single-sided (from the thesis / HHK) -- used only by
`DEFAULT_PARAMS` / the `"single_jump"` diagnostic in `Electricity_Market_Plots.py` and by
`tests/Electricity_Market_Test.py`'s mechanics checks, **not** by any pricing experiment:

```
κ = 7.0,  σ = 1.4,  β = 200.0,  λ = 5.0,  μ = 0.4,  Z_0 = Y_0 = 0
T = 2.0,  N = 400,  M = 20000 paths,  f(t) = ln(100) + 0.5 · cos(2π t)
```

**Settled double-sided calibration, used by every actual pricing/comparison experiment**
(`MARKET_PARAMS` in every `scripts/*_Experiment.py` / `Evaluation_*.py`): asymmetric in both `λ`
and `μ` -- down-jumps are both less frequent and smaller on average than up-jumps, a deliberate
choice (electricity spikes are up-dominated), not a nuisance parameter:

```
κ = 7.0,  σ = 1.4,  β = 40.0
λ_up = 5.0,  μ_up = 0.6,  λ_down = 3.0,  μ_down = 0.4
```

Was briefly `λ_up=0.5, λ_down=0.3` (a 10x scaling bug — caught 2026-08-15 when a mentor comment on
reproduced prices flagged the mismatch): jump *frequency* (`λ`, jumps/year) and mean jump *size*
(`μ`, `E[J]=μ` directly per this project's `scale`-parameter convention) are independent controls,
and at `λ_up=0.5` spikes were an almost negligible ~0.5 jumps/year, defeating the point of a
jump-diffusion model for spiky electricity prices. Every experiment run before that date used the
wrong (10x too small) `λ`s and needs rerunning against these corrected values.

`β` was also `200.0` until 2026-08-16, changed to `40.0`: at the swing experiments' weekly grid
(`N=50`, `T=1` ⟹ `Δ≈7.3` days), `β=200`'s ~1.8-day e-folding time meant a spike firing right after
one monitoring date had decayed to just ~1.8% of its size by the next one -- individual jumps were
genuinely large (see `μ` discussion above) but almost entirely invisible to the discretely-exercised
contract that's supposed to price them. `β=40` (e-folding ≈9.1 days) leaves ~45% of a spike's size
intact in that same worst-case timing, while still decaying ~5.7x faster than `κ`'s own
mean-reversion (so `Z` and `Y` stay qualitatively distinct: slow smooth fundamental vs. faster
spikes) and leaving ~5x the average inter-jump gap (45.6 days) as slack before the next jump
typically arrives (no systematic spike pile-up). The alternative fix -- keep `β=200`, raise `N` to
resolve it properly (e.g. daily, `N=365`) -- was rejected on cost: `N=365` is ~7.3x more backward-induction
steps per fit, too slow given how many experiments in this project already rerun repeatedly.
Every experiment run before this date used `β=200` and needs rerunning against `β=40` too.

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
  `aggregate` function, both `sum` and `max` implemented in `Payoff_Aggregation.py`. **Settled:
  `max_aggregation` is the thesis's actual specified payoff** (`r_n(x,a) = max_j Q(S_n^{(j)})` for
  `a=1` — "Payoff Structure": the shared right's immediate reward is the *maximal* individual-swing
  gain across dimensions, not the sum). `sum_aggregation` stays implemented and injectable, but isn't
  the portfolio's actual payoff; don't default to it. `d = 1` reduces trivially to `Q_total = Q(S_n)`
  for either.
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
   training and evaluation halves, `M_t = M_e = M/2`. (This is `price_swing`'s convention. The main
   reported SD/CI experiments instead simulate training and evaluation samples independently — see
   "Reported standard deviation and confidence intervals".)
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

**RNN (Approach 2).** `K−1` hidden units. Draw, once, and **fix them for the entire LSM run** (do
not resample per step):

```
θ_jℓ ~ N(0, 1/q),   b_j ~ N(0, 1),   j = 1,...,K−1,  ℓ = 1,...,q
```

(`q` = state dimension seen by the regression). `Θ`'s variance shrinks with `1/q` so `Θ·x`'s
variance stays roughly independent of `q` (a sum of `q` terms, each shrunk to compensate); `b` has
no such fan-in sum to compensate for, so it stays unscaled. `make_random_features_basis`'s
`theta_scale=None` default implements exactly this; pass `theta_scale=1.0` to reproduce the older
unscaled-`N(0,1)` convention (needed only to match a specific external paper's stated methodology,
e.g. the max-call validation — see `Validate_Max_Call_Benchmark.py`). Feature map, with constant
appended:

```
Φ(x) = ( σ(θ_1·x + b_1), ..., σ(θ_{K−1}·x + b_{K−1}), 1 ) ∈ R^K
```

`σ` a component-wise activation (e.g. tanh / ReLU). Only the output weights `β` are fitted (by the
ridge step above), per time step and per action. Thesis default `K−1 = 20`. `Φ_n` is assembled once
per time step (it does not depend on the action); the regression is refit per admissible action.

## Standardization

State values are large (spot price `~10^2`), which can make a preselection family under-representative
or push RNN's activation into a regime dominated by input scale rather than shape. Every feature map
exposes `standardize(state, mean, std) -> state`, called on the raw state **before** `build_features`,
every backward-induction step (the state's own distribution generally shifts with `n`). `mean`/`std`
are computed from **that step's training rows only** — same discipline as the regression fit itself,
so the fitted policy stays independent of the evaluation sample. Two different formulas, not one:

- **Polynomial, RNN:** full z-score, `x̂_ℓ = (x_ℓ − μ_ℓ)/s_ℓ`. An *unregularized* polynomial fit
  doesn't actually need this — `{1,x,...,x^d}` and `{1,x̂,...,x̂^d}` span the same functions for any
  affine `x̂`, so `least_squares` predictions are unchanged either way (verified: see
  `test_interior_step_continuation_beats_exercise_hand_traced`, unaffected by adding this). It's kept
  because `ridge_regression`'s penalty acts on raw coefficient magnitude, which is **not**
  affine-invariant — without a common scale across features, one shared `λ` penalizes them
  inconsistently (this is exactly what caused the `(Z,Y)+poly+ridge` anomaly discussed earlier).
- **Weighted Laguerre:** scale-only, `x̂_ℓ = x_ℓ/s_ℓ` — **no centering**. Its weight `exp(-x/(2K))`
  overflows for very negative `x`; centering could turn a positive price negative and blow it up, so
  this rescales magnitude while preserving sign.

## Reported standard deviation and confidence intervals

A single `Ṽ_0` estimate has two distinct sources of randomness, by the law of total variance:

```
Var(Ṽ_0) = Var_r[ V(π_r) ]           <- component A: policy-fit variability
          + E_r[ σ²(π_r) / M_e ]     <- component B: evaluation Monte Carlo noise, for a fixed policy
```

`π_r` is the exercise policy fit from training-sample seed `r`; `V(π_r)` is that policy's true value;
`σ²(π_r)/M_e` is the Monte Carlo variance of estimating `V(π_r)` from `M_e` evaluation paths. These
answer different questions and must be reported separately, never blended into one number:

- **Component A (the main reported SD):** repeat the whole experiment 5-10 times, changing **only**
  the training-sample seed each repetition, while the **evaluation sample stays fixed** across every
  repetition (same seed, same paths, every rep). The resulting spread in `Ṽ_0` is a clean estimate of
  `Var_r[V(π_r)]`, uncontaminated by evaluation noise — that contamination is exactly why
  `price_swing`'s single-seed-for-everything convention (train and eval as two halves of one
  `simulate_hhk` call, redrawn together each rep) is *not* what the main reported SD uses; that
  convention still ties the two components together.
- **Component B (a separate CLT confidence interval):** for one fixed, already-fitted policy, the
  evaluation mean `Ṽ_0 = mean(cashflows)` over `M_e` i.i.d. evaluation paths is itself a Monte Carlo
  estimator, so by the CLT it's approximately `N(V(π_r), σ²(π_r)/M_e)`. Report this as a confidence
  interval `Ṽ_0 ± z · s/√M_e`, `s` the sample std of `evaluate_policy`'s `cashflows` — never as
  another contribution to the between-repetition SD above.
- **Consistency check:** evaluate **one** fixed policy on several *different* evaluation samples
  (different seeds). The empirical spread of `Ṽ_0` across those runs should match the CLT interval
  computed from any single one of them — this confirms the CLT approximation is behaving as expected,
  and gives a direct read on how big component B is relative to component A (if B is negligible next
  to A, a single evaluation sample's CLT interval is a fair stand-in for repeating the evaluation).

**Regression.py API.** `fit_policy(S_train, regression_state_train, contract, aggregate, basis, fit,
alpha, train_itm_only=False) -> Policy` runs the backward induction on a training sample alone and
returns the fitted per-step coefficients (a `Policy` — regression coefficients + standardization
stats, per step — not propagated cash flows). `evaluate_policy(policy, S_eval, regression_state_eval,
contract, aggregate, basis, alpha) -> {"v0", "p", "cashflows"}` applies a `Policy`'s fixed coefficients
to a (possibly separately-simulated, possibly differently-sized) evaluation sample, never refitting.
`clt_confidence_interval(cashflows, confidence=0.95)` implements the CLT interval above, from
`evaluate_policy`'s `"cashflows"`. `price_swing` is unchanged and still the right tool for a single
quick price (one array, split train/eval in half internally) — it's now a thin wrapper composing
`fit_policy` + `evaluate_policy` under the hood (see "Correctness gates and gotchas" for the
equivalence), and is what the validation benchmark and most unit tests still use. The main reported
SD/CI experiments should call `fit_policy`/`evaluate_policy` directly instead.

**Open assumption, flag if wrong:** `clt_confidence_interval` defaults to a 95% two-sided normal
interval (`z ≈ 1.96`) — the standard convention, but the thesis's own `\eqref{Eq:Confidence_Interval}`
wasn't available when this was implemented (only its label, referenced from the pasted "Reported
Standard Deviation" paragraph). Override `confidence=` (or the formula itself, if it differs in kind,
not just level) if the thesis pins something else.

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
  `n=0`. Don't "simplify" this into fitting on all paths — that reintroduces in-sample leakage. (This
  describes `price_swing`'s single-array convention specifically; `fit_policy`/`evaluate_policy`
  generalize the same fit/propagate split to independently-simulated training/evaluation samples —
  see "Reported standard deviation and confidence intervals". `price_swing` is provably a special case:
  `test_fit_policy_then_evaluate_policy_matches_price_swing_split` checks the two agree exactly.)
- **Standardization is training-rows-only too**, same reason: `mean`/`std` computed from eval paths
  (even partially) would leak evaluation-sample information into the fitted policy. Applies to the
  *whole* state array (train+eval) using train-only statistics — never recomputed per split.
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
Q_total        multi-D gain = aggregate(Q(S_n^(1)),...,Q(S_n^(d))); settled: max, d=1 reduces to Q(S_n)
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
