# ML Model Derivations in `LotStateVector` Space

This memo extends the model math already summarized in `MLNetPipeline.md` and focuses on issue #9: explicit derivations for the implemented ML models in the `LotStateVector` feature space.

## 1) Feature map and target map

For each lot-time observation \(i\), define:
\[
v_i = (x_i, z_i, y_i^{\text{oracle}}, \tilde y_i^{\text{BT}})
\]
where \(x_i\in\mathbb{R}^{15}\) are numeric fields (`FeatureLists.NumericFeatures`) and \(z_i\) is sector.

The pipeline map is:
\[
\phi_i = \big[\mathrm{Norm}(x_i),\ \mathrm{OneHot}(\mathrm{SectorClean}(z_i))\big]\in\mathbb{R}^{d}
\]
with \(d=15+m\), where \(m\) is the sector vocabulary size learned on the training fold.

Targets:
\[
y_i=
\begin{cases}
\mathbf{1}[Y_{\text{Oracle},i}=1], & \texttt{target = "oracle"}\\[4pt]
\mathbf{1}[\tilde y^{\text{BT}}_i>0], & \texttt{target = "soft_bt"}
\end{cases}
\]
Rows with NaN\((\tilde y^{\text{BT}}_i)\) are dropped before splitting.

## 2) Train/test protocol used by all supervised trainers

Given filtered rows \(D\):

1. Stratified split \(D=D_{\text{train}}\cup D_{\text{test}}\), test fraction \(0.20\), seed \(42\).
2. Fit medians on \(D_{\text{train}}\) only, apply to train/test.
3. Compute balanced class weights on \(D_{\text{train}}\) only:
   \[
   w_c=\frac{N}{K\,n_c},\quad c\in\{0,1\}
   \]
   where \(N=|D_{\text{train}}|\), \(K\)=number of classes, \(n_c\)=count of class \(c\).
4. Run 5-fold stratified CV over a finite hyperparameter grid on training data only.
5. Refit on full \(D_{\text{train}}\) with best hyperparameters.
6. Evaluate once on \(D_{\text{test}}\).

## 3) Logistic regression (`LbfgsLogisticRegression`)

Model:
\[
p_i=\sigma(w^\top \phi_i+b)
\]

Objective:
\[
\min_{w,b}\ \sum_{i\in D_{\text{train}}} w_{y_i}\Big[-y_i\log p_i-(1-y_i)\log(1-p_i)\Big]
+\lambda\lVert w\rVert_2^2
\]
with \(\lambda=1/C\), and \(C\in\{0.01,0.1,1,10\}\) selected by CV.

## 4) Gradient-boosted trees (`FastTree`)

Additive score:
\[
F_M(\phi)=\sum_{m=1}^{M}\nu f_m(\phi)
\]
where each tree \(f_m\) is fit to pseudo-residuals of logistic loss.

Grid searched:
- `numberOfTrees` \(\in\{100,200\}\)
- `learningRate` \(\in\{0.10,0.05\}\)
- `numberOfLeaves` \(\in\{20,31\}\)

## 5) Random forest (`FastForest`)

Ensemble score:
\[
F(\phi)=\frac1T\sum_{t=1}^{T} f_t(\phi)
\]
with bagging and random feature subsampling.

Grid searched:
- `numberOfTrees` \(\in\{100,200\}\)
- `numberOfLeaves` \(\in\{20,31\}\)

## 6) Elastic net logistic (`SdcaLogisticRegression`)

Objective:
\[
\min_{w,b}\ \sum_{i\in D_{\text{train}}} w_{y_i}\,\ell_{\log}(y_i,w^\top\phi_i+b)
+\lambda_1\lVert w\rVert_1+\lambda_2\lVert w\rVert_2^2
\]

Grid searched:
- \(\lambda_1\in\{0.001,0.01,0.1\}\)
- \(\lambda_2\in\{0.001,0.01,0.1\}\)

## 7) Linear regression baseline (`Sdca`) on binary labels

Prediction:
\[
\hat y_i=w^\top\phi_i+b
\]

Objective:
\[
\min_{w,b}\ \sum_{i\in D_{\text{train}}} w_{y_i}\,(y_i-\hat y_i)^2+\lambda\lVert w\rVert_2^2
\]
with \(\lambda\in\{10^{-4},10^{-3},10^{-2}\}\) selected by CV.

This model is intentionally misspecified for binary outcomes and is kept as a poor-fit comparator.

## 8) Unsupervised derivations

### 8.1 PCA (`PcaPipeline`)

On standardized numeric matrix \(X\in\mathbb{R}^{n\times 15}\):
\[
C=\frac{1}{n-1}X^\top X,\qquad Cu_j=\lambda_j u_j
\]
Explained variance ratio:
\[
\rho_j=\frac{\lambda_j}{\sum_k \lambda_k}
\]
Keep smallest \(r\) such that:
\[
\sum_{j=1}^{r}\rho_j\ge 0.95
\]

### 8.2 K-means (`KMeansPipeline`)

For each symbol \(s\), aggregate four asset-level features across time
\((R_t,\Sigma_{\text{Range}},\Delta MA50,\Delta MA200)\),
then standardize to \(g_s\in\mathbb{R}^4\). Solve:
\[
\min_{\{\mu_c\},\{a_s\}}\sum_s \lVert g_s-\mu_{a_s}\rVert_2^2,\qquad a_s\in\{1,\dots,k\}
\]
Choose \(k\in\{5,10,15,20,25\}\) by maximum silhouette score.

## 9) Why this derivation is in lot-vector space

All supervised objectives above are functions of \(\phi(v_i)\), where \(v_i\) is a `LotStateVector` instance. This keeps the mathematical object (lot snapshot) aligned with the typed C# schema and avoids positional/CSV reinterpretation between simulation and ML training.

## Cross-reference

- `DataMemo/MLNetPipeline.md` (existing high-level pipeline + prior model equations)
- `DataMemo/MLNetLeakageAudit.md` (training-fold-only invariants)
- `src/Core/Portfolio/LotStateVector.cs` (canonical schema source)
- `src/ML/CSharp/MLNet/Models/*.cs` (trainer-level implementation)
