# Lot Vector Space — Derivations

> Derivations for the lot feature vector used by `LotSnapshot`, written as a standalone companion to `PortfolioMath.md`.

---

## 1. Lot-to-Vector Map

For an open lot $k$ of asset $A_i$ at day $t$, define the feature map

$$
\phi:(\text{Lot}_k,\mathcal{S}_t,\mathbf{P}_t)\mapsto x_{k,t}\in\mathbb{R}^{d},\quad d\approx 15.
$$

Using the project schema, the coordinate vector is

$$
x_{k,t}=\bigl(L,H,S,B,W,K,G_t^{\mathrm{YTD}},\sigma_{\mathrm{TE}},C_t^{A_i},R_t,\Sigma\mathrm{Range},\Delta\mathrm{MA50},\Delta\mathrm{MA200},\alpha_{\mathrm{tax}},\mathrm{DaysToYE}\bigr)^\top.
$$

---

## 2. Lot-Level Coordinate Derivations

Given shares $q_k$, cost basis $p_k$, purchase day $s_k$, current price $P_t$, and portfolio value $V_t$:

$$
L=\ell_k=\frac{P_t-p_k}{p_k}
$$

$$
H=h_k=t-s_k
$$

$$
S=\mathbf{1}_{\{h_k\ge 365\}}
$$

$$
B=p_k
$$

$$
W=w_k=\frac{q_kP_t}{V_t}
$$

$$
K=\#\{\text{open lots with ticker }A_i\}
$$

So the lot subspace coordinate is

$$
x^{\mathrm{lot}}_{k,t}=(L,H,S,B,W,K)^\top\in\mathbb{R}^{6}.
$$

---

## 3. Portfolio and Asset Coordinates

Portfolio coordinates:

$$
x^{\mathrm{port}}_{k,t}=(G_t^{\mathrm{YTD}},\sigma_{\mathrm{TE}},C_t^{A_i})^\top\in\mathbb{R}^{3},\qquad
C_t^{A_i}:=\text{wash clock in days since last harvest for asset }A_i.
$$

Asset coordinates:

$$
R_t=\frac{P_t-P_{t-1}}{P_{t-1}}
$$

$$
\Sigma\mathrm{Range}=\frac{H_t-L_t}{P_{t-1}}
$$

$$
\Delta\mathrm{MA50}=\frac{P_t-\mathrm{MA}_{50}}{\mathrm{MA}_{50}},\qquad
\Delta\mathrm{MA200}=\frac{P_t-\mathrm{MA}_{200}}{\mathrm{MA}_{200}}
$$

$$
x^{\mathrm{asset}}_{k,t}=(R_t,\Sigma\mathrm{Range},\Delta\mathrm{MA50},\Delta\mathrm{MA200})^\top\in\mathbb{R}^{4}.
$$

Derived coordinates:

$$
\alpha_{\mathrm{tax}}=\tau(h_k)\cdot\lvert G_{\mathrm{lot}}\rvert\cdot\mathbf{1}_{\{G_t^{\mathrm{YTD}}>0\}}
$$

$$
x^{\mathrm{derived}}_{k,t}=(\alpha_{\mathrm{tax}},\mathrm{DaysToYE})^\top\in\mathbb{R}^{2}.
$$

---

## 4. Lot Vector Space Decomposition

The total feature space is the direct-sum style decomposition

$$
\mathcal{X}=\mathcal{X}_{\mathrm{lot}}\oplus\mathcal{X}_{\mathrm{port}}\oplus\mathcal{X}_{\mathrm{asset}}\oplus\mathcal{X}_{\mathrm{derived}},
$$

with

$$
\dim(\mathcal{X})=6+3+4+2=15\quad(\text{before one-hot encoding of Sector}).
$$

Hence each lot snapshot row is a point

$$
(x_{k,t},y_{k,t})\in\mathcal{X}\times\mathcal{Y},\qquad
\mathcal{Y}=\{0,1\}\times[0,1].
$$

---

## 5. Oracle Region in Lot Vector Space

The hard oracle label is

$$
\theta_1=\text{loss threshold},\qquad \theta_2=\text{tracking-error threshold},\qquad \theta_3=30\ \text{(wash-sale day threshold)}.
$$

$$
f^*(x)=\mathbf{1}_{\{L\le-\theta_1\}}\cdot\mathbf{1}_{\{\sigma_{\mathrm{TE}}\le\theta_2\}}\cdot\mathbf{1}_{\{G_t^{\mathrm{YTD}}>0\}}\cdot\mathbf{1}_{\{C_t^{A_i}\ge\theta_3\}}.
$$

Therefore the harvest region is

$$
\Omega=\{x\in\mathcal{X}:L\le-\theta_1,\ \sigma_{\mathrm{TE}}\le\theta_2,\ G_t^{\mathrm{YTD}}>0,\ C_t^{A_i}\ge\theta_3\}.
$$

This gives the explicit lot-vector-space derivation used by the simulator and by downstream ML training.
