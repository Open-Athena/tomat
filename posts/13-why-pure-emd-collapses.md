# Why pure-EMD loss collapses (the actual math)

**Status**: living analysis — update as new ablations land.

## TL;DR

You'd think EMD beats cross-entropy for predicting a float: it understands
near-miss bins, CE doesn't. In practice, our three EMD-only training runs
all **collapsed within ~200 steps** to predicting a single constant $\rho$
everywhere and stayed flat for 3000+ more.

Three ingredients compose badly:

1. **Softmax** parameterizes the bin probabilities — and softmax's gradient
   multiplies *every* logit's update by that logit's current probability.
2. **A Dirac target** (one true bin, one-hot $\delta_t$) — because each
   density token has *one* true value.
3. **EMD as the loss** — because EMD is "expected distance under $P$,"
   which inherits the softmax-$P$-factor in its gradient.

At any concentrated $P$ (all mass at one bin $b^*$), the softmax factor
zeros out the gradient everywhere. Loss correctly reports "you're
wrong" — gradient says "nothing to push." Model is stuck.

CE escapes because its gradient has an extra $-\mathbf{1}[v=t]$ term
that *doesn't* go through the softmax factor — it always says "push the
true bin up by 1," regardless of current $P$. That single term breaks
the degeneracy.

The right fix has two parts:

1. **Drop Dirac targets.** The truth $\delta_t$ being a point mass
   is what makes EMD's gradient self-referential — there's no
   target-side $Q(v)$ term to anchor the gradient anywhere except
   "where I already put mass." Replace $\delta_t$ with a soft
   target $Q$ (e.g. Gaussian $\tau(v) \propto e^{-(\rho_v - \rho_t)^2/2\sigma^2}$),
   and now the gradient has a $-Q(v)$ term that **pulls toward
   where mass should be**, independent of $P$.
2. **Pick a loss that uses both $P$ and $Q$**. Several defensible
   choices (see §5–6); the canonical default is **KL divergence**
   (a.k.a. cross-entropy with soft targets), which gives a clean
   single-loss formulation in nats, no $\lambda$, no unit mixing.
   It's what C51 (distributional RL) and label smoothing (NLP) use.

The legacy `ce_emd` hybrid (CE + $\lambda$·EMD) is the *cheapest*
A/B — one config flag, breaks the degeneracy via CE's
$-\mathbf{1}[v=t]$ — but mixes units (nats + $\rho$-units), needs
a $\lambda$ sweep, and is conceptually two-ideas-glued. **Lead with
KL-Gaussian; keep `ce_emd` as a smoke comparator.**

For runs that demand the "physical-distance" structure of EMD even
after softening the target, **Sinkhorn-regularized $W_1$**
(entropy-regularized OT) is the principled answer. **CRPS**
(Continuous Ranked Probability Score) is the distributional-
forecasting standard from a different angle (acts on CDFs, not bin
probabilities). Both worth ablating; see §5.

---

## 1. Setup

Each density token's vocabulary is a finite set of $V$ codebook bins
(LMQ-v2 → $V \approx 16{,}384$). The codec gives each bin a
**reconstruction value** $\rho_v \in \mathbb{R}_{\ge 0}$ — the float we
decode when we sample bin $v$.

The model outputs logits $z \in \mathbb{R}^V$ per density position, with
$P(v) = \mathrm{softmax}(z)_v = e^{z_v} / \sum_u e^{z_u}$.

### Why a Dirac target?

Each training example has *one* true $\rho_t$, which gets quantized
into *one* true bin $t$. So the target distribution over bins is "all
mass on $t$, zero elsewhere" — a Dirac delta $\delta_t$.

You can soften this (label smoothing, Gaussian-around-$t$) and that's
one of the standard escape routes — see §6 below. The Dirac is the
default supervised setup, not a "natural law."

### 1.1 Cross-entropy

$$
\mathcal{L}_{\mathrm{CE}} = -\log P(t)
$$

"the probability you assigned to the true bin."

### 1.2 EMD ($W_1$, the Wasserstein-1 distance)

For a Dirac target $\delta_t$, $W_1$ collapses to the **expected
$\rho$-distance** under $P$:

$$
\mathcal{L}_{\mathrm{EMD}}
   = W_1(P,\ \delta_t)
   = \sum_v P(v)\,\bigl|\rho_v - \rho_t\bigr|
$$

"the average real-line cost of moving the mass at each bin to where
the truth sits." Near-miss bins (small $|\rho_v - \rho_t|$) barely
contribute; CE would charge them $\ln V \approx 9.7$.

---

## 2. The gradients

Let $c_v \equiv |\rho_v - \rho_t|$ (per-bin cost).

### 2.1 CE gradient

$$
\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial z_v}
   = P(v) - \mathbf{1}[v = t]
$$

The second term is the rescue: even when $P(t) = 0$, the gradient at
$z_t$ is $0 - 1 = -1$, pushing the model to *increase* $z_t$. The true
bin always has a nonzero gradient pull, no matter what $P$ looks like.

### 2.2 EMD gradient (and the softmax factor)

Chain rule:

$$
\frac{\partial \mathcal{L}_{\mathrm{EMD}}}{\partial z_v}
   = \sum_u c_u \cdot \underbrace{P(u)(\delta_{uv} - P(v))}_{\partial P(u)/\partial z_v}
$$

Working it out (sub $u = v$, then $u \neq v$):

$$
   = P(v)\,c_v\ -\ P(v)\!\sum_u P(u)\,c_u
   = P(v)\,\bigl(c_v - \mathcal{L}_{\mathrm{EMD}}\bigr)
$$

So:

$$
\boxed{\ \frac{\partial \mathcal{L}_{\mathrm{EMD}}}{\partial z_v}
        = P(v) \cdot \bigl(c_v - \mathcal{L}_{\mathrm{EMD}}\bigr)\ }
$$

**In words**: each bin's gradient is "this bin's probability"
× "how-much-worse-than-this-model's-own-average this bin is."

The deeper observation: this gradient is **entirely
self-referential**. The truth $t$ only enters via the per-bin cost
$c_v = |\rho_v - \rho_t|$, and that cost is multiplied by $P(v)$.
There is no term that says "push the true bin $t$ up regardless of
what you predicted" — every contribution gets filtered through the
model's own current distribution.

Contrast CE's $-\mathbf{1}[v=t]$ term: a non-self-referential signal
pointing directly at the truth, independent of $P$. *That* is what
EMD is missing.

### 2.3 Loss vs gradient: a clean distinction

The loss *value* correctly reports distance: predicting bin $b^*$ at
$\rho \approx 25$ when the truth is $\rho_t = 80$ gives
$\mathcal{L} = 55$. The loss measurement is fine.

The *gradient* is what we use to update weights. And the gradient has
a $P(v)$ factor that kills the update on every bin where current
probability is near zero. The model isn't told "push toward truth"
— it's told "nothing to push, since you've put no mass anywhere we
could improve."

This is the gap your intuition is sniffing at: "the loss penalizes
wrong predictions, so why doesn't the gradient fix them?" The
gradient *does* try, but the chain rule through softmax multiplies
every push by the current $P$ at that bin. Zero $P$ → zero push.

---

## 3. The trivial attractor

Suppose the model concentrates all mass on a single bin $b^*$:
$P(b^*) = 1$, $P(v) = 0\ \forall v \neq b^*$. Then:

- For every $v \neq b^*$: gradient $= 0 \cdot (\cdot) = 0$. Cannot
  start spreading mass.
- For $v = b^*$: gradient $= 1 \cdot (c_{b^*} - \mathcal{L})$, and
  $\mathcal{L} = c_{b^*}$ (only $b^*$ has mass). So gradient on $b^*$
  is also zero.

**Every logit has zero gradient.** Concentrated $P$ is a fixed point.

### 3.1 Why is the fixed point *attractive*?

You asked "how is this an attractor?" — fair question. The wrong bin
$b^*$ doesn't have zero loss; it has nonzero loss. Why does the model
end up there?

Because $b^*$ minimizes the *average* loss across training rows under
the constraint "predict one bin everywhere." For tomat:
$\rho_{b^*} \approx \mathrm{median}_t(\rho_t)$ — the L1-median of the
true density distribution. Predicting any *other* single bin makes the
loss higher on average.

So gradient descent flows toward $b^*$ during the first ~200 steps:
- Far from $b^*$: gradient has nonzero $P(v)$, the model spreads mass
  out broadly, finds that mass-near-$b^*$ has lowest cost, concentrates
- *At* $b^*$: gradient vanishes, model locks in

It's attractive in the sense that nearby starting points get sucked
toward $b^*$. Stable in the sense that once there, no infinitesimal
perturbation gets the model out — every other gradient component is
zero.

### 3.2 What we actually see

| Run                 | Step at collapse | Predicted $\rho$ | NMAE     |
|---------------------|------------------|------------------|----------|
| `train-mg-3-cos-emd` | ~200             | const 25.21      | 58-68%   |
| `train-ar-emd-real` | ~200             | const ~25        | 50.6%    |

TL dropped from "thousands" to ~100 in the first 200 steps, then
**flat for 3000+ more** with near-zero variance. Not a noisy plateau
— actually-zero gradient signal.

---

## 4. Counter-intuitions (your questions)

### "But $P(v)$ is never exactly zero in softmax."

True — it's $e^{-(z_{b^*} - z_v)}$, which for trained $z_{b^*} \gg z_v$
gives values like $10^{-20}$. The gradient *exists* but is too small
to overcome any reasonable LR. Empirically: TL flat for thousands of
steps, lower LR doesn't help, higher LR oscillates around the same
$b^*$.

### "The model has context — surely it can use that to differentiate."

It *would* if it could escape the attractor. But escape requires
nonzero gradient on logits other than $b^*$, and at the attractor
those are all zero. Context can't propagate backward through a zero
gradient.

By contrast, CE's $-\mathbf{1}[v=t]$ term is *always* nonzero on $z_t$
regardless of current $P$, so the model immediately starts attending
to context for finding $t$.

### "But a neuron $\theta$ affects *all* 16k logits — surely some of its effect leaks through to a real gradient?"

This is the sharpest version of the puzzle. The intuition: parameter
$\theta$ has nonzero $\partial z_v / \partial \theta$ for many $v$,
and the loss depends on all logits. So the chain-rule sum
$\partial L/\partial \theta = \sum_v (\partial L/\partial z_v) \cdot
(\partial z_v/\partial \theta)$ should be nonzero generically.

What goes wrong: at the attractor, **every** $\partial L/\partial z_v$
is zero, so no $\partial z_v/\partial \theta$ values save us:

- For $v \neq b^*$: $P(v) = 0$, so $\partial L/\partial z_v = 0 \cdot (\cdot) = 0$.
- For $v = b^*$: $c_{b^*} - \mathcal{L}_{\mathrm{EMD}} = 0$ (loss equals
  the only contributing cost), so $\partial L/\partial z_v = 1 \cdot 0 = 0$.

The chain-rule sum is $\sum_v 0 \cdot g_v = 0$ for *every* parameter $\theta$.

The geometric way to see it: the softmax map $z \mapsto P$ has a
**singular Jacobian** at concentrated points. Perturb $\theta$ by
$\delta$. The induced change in $P(v)$ to first order is

$$
\Delta P(v) = P(v)\bigl(g_v - \textstyle\sum_u P(u)\,g_u\bigr)\,\delta
\quad\text{where}\quad g_v \equiv \tfrac{\partial z_v}{\partial \theta}
$$

At the attractor: $\Delta P(b^*) = 1 \cdot (g_{b^*} - g_{b^*})\delta = 0$,
and $\Delta P(v) = 0 \cdot (\cdot) = 0$ for $v \neq b^*$. **$P$ doesn't
change at all to first order.** Logits move (the neuron's effect on
$z$ is real), but they all move in the *saturated*-softmax regime
where probabilities are insensitive to logit shifts: $P(b^*)$ stays
near 1, the others stay near 0.

To escape the attractor, $P(b^*)$ has to drop from 1 — but softmax
needs a *finite* shift in logit gaps to do that, not an infinitesimal
one. Gradient descent only makes infinitesimal moves. Stuck.

This is the same vanishing-gradient pathology that plagues sigmoid
layers in saturation — softmax has a higher-dimensional version of
it, and the trivial-attractor analysis tells us this saturation point
is the *natural endpoint* of EMD's loss landscape (not transient, the
way it would be from a random init).

### "We'd get a similar weight-grad update if model predicted 100% at $t{+}1$ vs 100% at $t{+}2$ — shouldn't we scale to absolute max distance, not the local average?"

Spot-on observation, with an even more degenerate twist than you
phrased: **both updates aren't just "similar" — they're both exactly
zero**. The trivial-attractor argument fires for *any* concentrated
$P = \delta_{b^*}$, regardless of how far $b^*$ sits from $t$.

So:
- Loss values are *different* ($c_{t+1}$ vs $c_{t+2}$ — the loss does
  rank "wrong by 1" as better than "wrong by 2").
- Gradient signals are *the same* (both zero).

Your "scale to absolute" intuition is sniffing at: "the loss knows
which attractor is worse, why can't the gradient transmit that?"

#### Does scaling to absolute fix it?

Suppose we replace the reference $\mathcal{L}_{\mathrm{EMD}} = \sum_u P(u) c_u$
in the gradient with a global constant like $c_{\max}$ (the
worst-possible cost, $\rho_{\max} - \rho_{\min}$):

$$
\frac{\partial L'}{\partial z_v} \stackrel{?}{=} P(v)\,(c_v - c_{\max})
$$

At $P = \delta_{b^*}$: $\partial L'/\partial z_v = 0$ for $v \neq b^*$
(still! — the $P(v)$ factor still kills it), and $\partial L'/\partial z_{b^*} = c_{b^*} - c_{\max} < 0$
(now nonzero — push $z_{b^*}$ down). The fixed point is gone for $b^*$
but the gradient on the *other* 16k logits is still zero.

In softmax, lowering $z_{b^*}$ does relatively raise the other logits
(only gaps matter). But uniformly — the gradient doesn't tell the
model *which* bin is correct, just that $b^*$ is wrong. The model
would need to spread mass back out and *re-discover* $t$ from scratch.

So absolute-scaling makes the attractor unstable but doesn't actually
target $t$. The bigger issue lurks beneath: the multiplicative $P(v)$
factor in *all* the per-logit gradients. Any loss of the form
"$\sum_v P(v) \cdot \text{something}(v)$" — including any reweighting
of the cost — produces gradients proportional to $P(v)$ on bin $v$,
which means **zero at any non-$v$ bin where $P(v) = 0$**.

The fixes that actually work all introduce a term that **doesn't go
through the $P(v)$ factor**:
- CE's $-\mathbf{1}[v=t]$ — pure index-target term, no $P$.
- Label smoothing / Gaussian target's $-\tau(v)$ — soft index-target
  term, no $P$.
- Entropy regularization $-H(P)$ — has $\log P(v) + 1$ structure, only
  partially mitigates but pushes against concentration.

This is why the cheap fix is `ce_emd = CE + λ·EMD`: CE's
$-\mathbf{1}[v=t]$ term reanimates the gradient at the attractor
*specifically pointing toward $t$*, then EMD takes over once $P$ has
spread out a bit and its near-miss-tolerance kicks in.

#### Side note: yes, your LR observation is real (separately)

EMD's per-step gradient magnitude scales with $c_v - L$, which is in
"$\rho$-units" (a few × $\rho_{\max} - \rho_{\min}$). CE's per-step
gradient magnitude is in "probability units" (0 to 1). Same nominal
LR therefore corresponds to different effective step sizes. We've
been firing EMD runs with CE-tuned LR, which on top of the
collapse-to-attractor issue probably also undershoots EMD's natural
LR scale. Worth a separate sweep if/when EMD-only ever does start
learning.

### "If I imagine the model as argmax instead of softmax, would EMD work?"

Argmax has *no* gradient anywhere (it's piecewise constant), so it
can't be trained at all via gradient descent. The natural smoothing
of argmax is softmax-with-temperature; as temperature → 0, softmax
approaches argmax. The pathology gets *worse* with lower temperature
(sharper concentrations → smaller $P(v)$ at non-$b^*$ bins → smaller
gradient). So no, going closer to argmax doesn't help.

### "What if we used a Gaussian target around $t$ instead of a Dirac?"

This is exactly your "weight could have been at any of the bins closer
to $t$" intuition — and yes, it works. With a target distribution
$\tau(v) \propto e^{-(\rho_v - \rho_t)^2 / 2\sigma^2}$, the EMD loss
becomes

$$
\mathcal{L}_{\mathrm{EMD}}(P, \tau) = \sum_v P(v) \sum_u \tau(u)\,|\rho_v - \rho_u|
$$

— now the cost function "stretches" smoothly around $t$, neighbor
bins of $t$ have nonzero target mass, and the gradient has cross
terms that don't all collapse at concentrated $P$. You're not training
the model to predict a Gaussian per se — the model still outputs a
categorical via softmax. You're just giving it a softer target so the
gradient stays alive.

This is the standard fix in distributional learning. See §6.

---

## 5. The loss menu — what to actually ablate

There's no single "correct" loss for "predict a categorical over
ordered bins of a continuous quantity." Several defensible choices,
roughly ordered by ease-of-adoption × principled-ness:

### 5.1 KL(Q‖P) with Gaussian $Q$ — recommended default

$$
\mathcal{L} = -\sum_v Q(v) \log P(v), \quad
Q(v) \propto \exp\!\bigl(-(\rho_v - \rho_t)^2 / 2\sigma^2\bigr)
$$

Gradient: $\partial \mathcal{L}/\partial z_v = P(v) - Q(v)$. Both
terms in nats. The $-Q(v)$ term is the **ideal-bin pull** EMD is
missing. One hyperparameter: $\sigma$. As $\sigma \to 0$ recovers
vanilla CE; wide $\sigma$ gives EMD-like smoothness.

This is **what C51 (distributional RL) and label smoothing (NLP)
use**. Most ML-mainstream of the options. Strictly better than
`ce_emd` on first principles (no $\lambda$, single unit, no
hybrid).

### 5.2 `ce_emd` (CE + $\lambda \cdot$ EMD) — cheap A/B

What's already in the code. Borrow CE's $-\mathbf{1}[v=t]$ rescue
term, keep EMD's near-miss tolerance. Mixed units, $\lambda$ sweep,
two-ideas-glued. Useful as a smoke A/B — if it works, KL-Gaussian
should work too.

### 5.3 CRPS (Continuous Ranked Probability Score) — and the L¹-of-CDFs identity

Before talking about CRPS, an identity worth stating because it
reframes everything: **EMD between $P$ and a Dirac (on $\mathbb{R}$)
is exactly the L¹ distance between their CDFs**.

$$
W_1(P, \delta_t) = \int_{-\infty}^{\infty} |F_P(x) - \mathbf{1}[x \ge \rho_t]| \, dx
   = \sum_v P(v)\,|\rho_v - \rho_t|
$$

The two forms are equal by change-of-order. Same metric, two views.

**CRPS** is the **L² version of the same idea**:

$$
\mathrm{CRPS}(F_P, t) = \int_{-\infty}^{\infty} \bigl(F_P(x) - \mathbf{1}[x \ge \rho_t]\bigr)^2 \, dx
$$

Squared CDF mismatch instead of absolute. Consequences:

- **For Dirac vs Dirac**, CRPS = EMD numerically (squared indicator
  difference = symmetric-difference indicator, which integrates to the
  gap). So at any $P = \delta_{b^*}$, both losses report the same
  $|\rho_{b^*} - \rho_t|$.
- **For diffuse $P$**, CRPS ≤ EMD pointwise. CRPS up-weights large
  CDF mismatches more aggressively than EMD.
- **CRPS is a strictly proper scoring rule** — uniquely minimized by
  predicting the *true conditional distribution*. EMD against a Dirac
  is minimized by collapsing to the L¹-median, not by predicting the
  right distribution. This is CRPS's main theoretical edge.

#### The honest correction: CRPS shares the trivial-attractor pathology

A previous version of this post claimed CRPS "sidesteps the softmax +
Dirac pathology entirely" — that's **wrong**, and the user (good
question, kept me honest) caught it. The gradient w.r.t. $z_v$ chains
through

$$
\frac{\partial F_P(\rho_k)}{\partial z_v} = P(v)\bigl(\mathbf{1}[v \le k] - F_P(\rho_k)\bigr)
$$

— same multiplicative $P(v)$ factor as EMD. At $P = \delta_{b^*}$,
this is zero for every $v \ne b^*$ (because $P(v) = 0$) and zero for
$v = b^*$ (because $\mathbf{1}[v \le k] - F_P(\rho_k) = 0$ at the step
location and elsewhere). **All gradients vanish at the trivial
attractor**, same as EMD.

The energy-score view makes this concrete: $\mathrm{CRPS}(P, t) =
\mathbb{E}_P[|x - \rho_t|] - \tfrac{1}{2}\mathbb{E}_{P \otimes P}[|x - y|]$.
First term is EMD. Second term is a self-spread regularizer that
vanishes at any Dirac. So at the attractor CRPS reduces to EMD and
inherits its gradient pathology.

#### What CRPS *is* good for

The pathology lives in **softmax + concentrated $P$**, not in the
choice of loss function. Any loss whose gradient w.r.t. $z$ factors
through $\partial P / \partial z$ has the same trivial-attractor
problem.

CRPS as a *standalone* loss therefore doesn't escape; pair it with
one of the escape mechanisms (§5.1's KL-Gauss / soft target → CRPS
against $Q$, or §5.2's `ce_emd`-style hybrid → CRPS + λ·CE) and the
escape returns. The proper-scoring-rule property is then a real
upside vs EMD.

### 5.4 Sinkhorn-regularized $W_1$ ($P$ vs Gaussian $Q$)

Entropy-regularized OT (Cuturi 2013). Preserves the
"physical-distance" structure of EMD that you wanted in the first
place, but the entropic reg breaks the multiplicative-softmax
pathology and Gaussian $Q$ supplies the ideal-bin pull.

Most expensive of the four (Sinkhorn iterations inside the forward
pass), most principled-feeling for "predict ρ, care about ρ-distance."
Two hyperparameters: $\sigma$ (Gaussian width) + $\epsilon$ (Sinkhorn
reg strength).

### 5.5 The softmax pathology is in the parameterization — try atan-normalization

The trivial-attractor analysis from §3-4 is entirely about softmax's
gradient structure, not about EMD. For any parameterization
$P(v) = f(z_v) / Z$:

$$
\frac{\partial L_{\mathrm{EMD}}}{\partial z_v} = \frac{f'(z_v)}{Z}\bigl(c_v - L_{\mathrm{EMD}}\bigr)
$$

Softmax has $f(z) = e^z$, so $f'(z_v)/Z = P(v)$, and the factor
vanishes **exponentially** at concentrated $P$.

**Atan-normalization** uses $f(z) = (\arctan(z) + \pi/2)/\pi$ (maps
$\mathbb{R} \to (0, 1)$ smoothly). Its derivative is $1/(\pi(1+z^2))$
— **polynomially** decaying. At a logit gap of 20 nats, softmax's
gradient factor is $\sim e^{-20} \approx 2 \times 10^{-9}$; atan's is
$\sim 1/400 \approx 2.5 \times 10^{-3}$. **About 7 orders of
magnitude more gradient signal** at concentrated $P$.

The gradient direction is correct too: at $P = \delta_{b^*}$, the
gradient on $z_v$ for $v \ne b^*$ is $\frac{f'(z_v)}{Z}(c_v - c_{b^*})$,
which pulls bins closer to truth (where $c_v < c_{b^*}$) up. So
atan-norm + pure EMD plausibly escapes the trivial attractor without
needing CE, soft targets, or any other escape mechanism.

Caveats:

- **Still saturates**, just polynomially. At extreme $|z_v|$
  ($\sim 100$+), $1/z_v^2$ is also small. Not a complete fix; pair
  with soft targets for robustness.
- **Probabilities never exactly 0 or 1** under atan-norm.
- **Less library support** than softmax / sparsemax. Custom layer,
  ~20 LoC in JAX.

The literature alternatives that share the "less-saturating"
property:

- **Sparsemax** (Martins & Astudillo 2016) — Euclidean projection
  onto the simplex; gives *exact* zeros outside a support set with
  full-rank Jacobian on the support.
- **Entmax / α-entmax** (Peters, Niculae, Martins 2019) — interpolates
  softmax (α=1) ↔ sparsemax (α=2).
- **Mixture-of-Gaussians head** — predict $(\mu_k, \sigma_k, \pi_k)$
  directly, skip bin discretization.
- **Direct regression on $\rho$** with NLL — simplest, loses
  multi-modality.

### When atan-norm matters most

Two cases where this is the right experiment:

1. `ce_emd` works but pure EMD collapses → confirms the issue is
   gradient signal at the attractor, not the loss formulation.
   `emd_atan` should also work; if it does, it's a cleaner story than
   `ce_emd` (no λ-tuning, no two-ideas-glued).
2. KL-Gauss wins → atan-norm is unnecessary; the soft target alone
   supplies the escape and atan-norm complicates the model.

### What I'd actually run

A four-way ablation, all from the same warm-start ckpt
(e.g. cont33k), ~2k steps each on Modal H100×8:

| Run                        | Loss                              | Cost  | Notes  |
|----------------------------|-----------------------------------|-------|--------|
| `cont-ce-baseline`         | vanilla CE                        | 1×    | reference |
| `cont-kl-gauss-s0.5`       | KL(Gauss(σ=0.5) ‖ $P$)            | 1×    | recommended default — has the escape term |
| `cont-ce-emd-l0.1`         | CE + 0.1·EMD                      | 1×    | cheap A/B for "does any escape mechanism work" |
| `cont-crps`                | pure CRPS                         | 1.1×  | likely collapses to same attractor as pure EMD — useful as a negative control + sanity-check that the attractor is really softmax-driven, not EMD-specific |

(Sinkhorn-W1 added later if any of the above show traction — it's
the most expensive and benefits from $\sigma$ + $\epsilon$ knobs
already set by KL-Gauss.)

If pure CRPS collapses as predicted, the meaningful CRPS arm is
**`cont-crps-kl-gauss`** — CRPS against a Gaussian $Q$ instead of a
Dirac. That gets the proper-scoring-rule property AND the escape
term in one loss.

A $\sigma$ sweep on the winner second (probably 3 values: 0.25,
0.5, 1.0 in σ/ρ_max units).

---

## 6. Literature pointers (this problem is well-studied)

The "EMD loss with softmax + sparse targets is degenerate" failure
mode has a small-but-real literature in the optimal-transport for
learning space:

- **Frogner et al. (2015), "Learning with a Wasserstein Loss"**.
  Direct precedent — uses $W_p$ losses for multi-class classification.
  Notes the degenerate-fixed-point issue and proposes regularization
  + entropy-relaxed Sinkhorn-style losses to mitigate.
- **Bellemare et al. (2017), "A Distributional Perspective on
  Reinforcement Learning"** (the C51 paper). Predicts categorical
  distributions over discrete return support, uses a cross-entropy
  loss with carefully crafted Bellman targets. Sidesteps the Dirac
  issue by construction.
- **Cuturi (2013), "Sinkhorn Distances"**. Adds entropy regularization
  to the OT problem; the resulting Sinkhorn distance has nicer
  gradients than vanilla $W_1$ at the cost of approximating the true
  Wasserstein distance.
- **Gneiting & Raftery (2007), "Strictly Proper Scoring Rules"**.
  CRPS is *the* go-to loss for distributional forecasting (weather,
  load, financial). Avoids the softmax + Dirac trap by acting on the
  CDF directly.
- **Müller (1997), "Integral probability metrics"**. Frames $W_1$
  alongside other distribution distances; useful for thinking about
  why $W_1$'s primal form gives this particular gradient.
- **Martins & Astudillo (2016), "From Softmax to Sparsemax"**.
  Drop-in sparsemax replacement, well-behaved with EMD/W_1 losses.

The standard advice across these:
1. Soft / smoothed targets (or distributional Bellman),
2. Entropy regularization on the predicted distribution, **or**
3. A parameterization that doesn't have softmax's multiplicative
   coupling.

Hybrid losses (CE + λ·W) are the cheapest of these.

---

## 7. Practical implications for tomat

- All "EMD" results in the project before 2026-05-25 were actually
  CE — see [the init-cls bug postmortem](../specs/done/30-levanter-init-cls-postmortem.md).
  EMD as a primary loss has only been tested in a handful of
  post-fix runs (all collapsed).
- **Lead with KL-Gaussian, not `ce_emd`.** It's one loss, one unit,
  one hyperparameter ($\sigma$), and has a principled story (target-
  side ideal-bin pull). `ce_emd` is the cheaper A/B but it's
  strictly less clean.
- **The 4-way ablation in §5** is what tells us which loss
  actually wins on MT/MV at fixed step budget. Hypothesis: at least
  one of {KL-Gauss, CRPS} beats vanilla CE meaningfully on NMAE per
  step (NMAE rewards near-miss credit, which CE doesn't supply).
  Risk: maybe they all converge to similar end-points and CE just
  takes more steps — that's still useful information.
- **No architectural changes for Phase A.** All four ablation arms
  are 30-100 LoC in the loss module + one env var or CLI flag each.
  Same model, same data, same trainer config; only the loss
  changes.

---

## 8. Why the user's frustration is reasonable

The intuition "CE is poorly suited to predicting floats, EMD should
help" is correct **for the eval metric**: mat-NMAE is essentially
post-hoc EMD on decoded $\rho$, and a model that gets near-miss bins
right scores much better on NMAE.

But that "should help" applies to the *evaluation*, not the
*training-time loss landscape*. EMD as a training loss interacts
badly with softmax + Dirac targets. The remedies above all preserve
the "near-miss tolerance" intuition while fixing the gradient
pathology.

The takeaway isn't "EMD is a bad metric." It's "you can't naively
plug a Wasserstein loss into a softmax-categorical / one-hot-target
setup and expect well-behaved gradients." The literature is clear
on this; we should adopt one of the standard fixes.

---

## 9. Forensic data on GCS

- `gs://marin-us-east5/tomat/results/train-mg-3-cos-emd/checkpoints/.../step-3195/`
- `gs://marin-us-east5/tomat/results/train-ar-emd-real/checkpoints/.../step-3738/`
- Eval JSONs at `gs://marin-eu-west4/tomat/eval/results/{run}/val_200{,-maskgit}/step-{3195,3738}.json`
- Per-mat predicted $\rho$ histograms (recovered from the eval JSONs)
  all show the singular spike at $\rho \approx 25$.

See also: [init-cls bug postmortem](../specs/done/30-levanter-init-cls-postmortem.md),
spec 17 (density-loss design), spec 20 (8k EMD-DO session log).
