## Proposed Roadmap

| Version | Timeline | Focus |
|---------|----------|-------|
| v0.1 | PSTAT 231 (Spring 2026) | Supervised ML baseline + soft labels |
| v0.3 | Summer / Junior Fall | Volatility sub-model|
| v0.4 | Junior Year | RL policy layer |
| v0.5 | Senior Capstone | Full system integration |
| v1.0 | Post-graduation | Production |

---

### v0.1 — Supervised ML Baseline *(PSTAT 231, Spring 2026)*
- Hard oracle labels `y ∈ {0,1}` from wash-sale-aware threshold gate
- Soft labels `ỹ ∈ [0,1]` from 30-day forward simulation window
- ~10 features from lot state + historical price snapshot
- Four model types fit and tuned with k-fold CV
- Hard vs. soft label comparison as primary empirical finding
- **Deliverable:** course project + "does inner-domain learning exist?"

### v0.3 — Volatility Sub-Model *(Summer / Junior Year Fall)*
- Replace constant-σ GBM with GARCH/EWMA volatility estimates
- Ledoit-Wolf shrinkage for covariance matrix estimation
- Richer soft label construction using time-varying σ̂
- **Deliverable:** "does better σ̂ improve harvest urgency scoring?"

### v0.4 — RL Policy Layer *(Junior Year)*
- PPO or SAC agent trained on simulator environment
- Oracle gap quantified empirically (how much alpha is irreducible?)
- **Deliverable:** "how much alpha does the RL policy recover over supervised baseline?"

### v0.5 — Full System Integration *(Senior Capstone)*
- Live data pipeline replacing historical snapshots
- Client-parameterized policies (risk tolerance, tax bracket, horizon)
- Backtesting on real historical S&P 500 data
- **Deliverable:** working demo + capstone presentation

...

### v1.0 — Production *(Post-grad / lifetime)*
- Production deployment
- Regulatory compliance (RIA registration) & self-credibility
- HNW client onboarding
- B2C/C2C SaaS with curated intervention/custimization by yours truly (per client).
- 0.15-0.75 basis points planned annual charge per client although will depend on a very large number of factors in the future or depending on tiers offered up to family office.

### V2.0 (hypothetical):
- everything in V1.0 with financial derivative based tax alpha optimization

Very simplified steps:

Step 1: Data & Schema downloaded & modeled 



Step 2: Simulator runs (C#)
        → computes ALL features x_{k,t}
        → computes ALL labels ỹ_{k,t}    ← labels are known BEFORE any ML
        → writes both to lots.csv
        → simulator is done forever

Step 3: ML training (C#)
        → reads lots.csv
        → sees BOTH features AND labels
        → learns mapping: x → ỹ
        → produces trained model η̂

Step 4: Deployment (future)
        → only features x_{k,t} exist
        → labels don't exist yet (future prices unknown)
        → trained η̂ estimates what ỹ would be
...
<h2>**Beyond the course**</h2>

goal: deployment of a full live D.I system C2C SaaS-ready (Client to Client meaning that I myself am a client of my very own future DI system)
Priority assumptions for deploying the full RIA-style portfolio management schema:
I. credibility via 
...

try out:
I. Neural Networks
II. Reinforcement learning (no oracle boundary needed)

hypothesis: whether these methods beyond the course result in larger portfolio tax-based alpha and reduced total tracking error & efficient dimensionality reduction / swappage.

once trialed (reinforcment learning likely the main archetype)

- volatility in the feature space as a supervised sub-model (surface volatility via gradient descent, regression, ect) with its own sub feature-space returns $$σ^t​ \in \mathbb{R}$$ value instead of the main DI classification system. || time series based forecasting.

