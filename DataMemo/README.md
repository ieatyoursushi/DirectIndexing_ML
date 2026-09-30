# DataMemo — the design and mathematics, in three tiers

The tier tells you what a document is *for*, and therefore whether it has to be current.

| Tier | Contract | Contents |
|---|---|---|
| [`spec/`](spec/SymbolTable.md) | **Live.** Must match the code; machine-checked by `dotnet run --project src -- docs-check`. **The only tier you need to hold in your head.** | [`SymbolTable.md`](spec/SymbolTable.md) is the index: every math object, typed, linked to its code member and test. Then [`MLDerivations.md`](spec/MLDerivations.md), [`SimulationMath.md`](spec/SimulationMath.md), [`PortfolioMath.md`](spec/PortfolioMath.md), [`MLNetLayer.md`](spec/MLNetLayer.md), [`MLNetLeakageAudit.md`](spec/MLNetLeakageAudit.md) |
| [`decisions/`](decisions/GYTD_Redesign_Plan.md) | **Dated design records.** Frozen after merge; superseded, never edited. The *why* behind a version. | [`GYTD_Redesign_Plan.md`](decisions/GYTD_Redesign_Plan.md) (v0.25), [`ValidationHardening_v026.md`](decisions/ValidationHardening_v026.md) (v0.26); v0.3 memos land here |
| [`archive/`](archive/RetiredComponents.md) | **History.** Frozen, with a banner saying what superseded each document. | [`RetiredComponents.md`](archive/RetiredComponents.md) (everything the downsizing removed and what it taught), the theory memos, the v0.2 lifecycle walk, the course recap, the architecture thread |

**How to read the math alongside the code.**
1. Look up an object in [`SymbolTable.md`](spec/SymbolTable.md). Its row gives the type
   (with units), a one-line definition, the implementing member, and the test that pins it.
2. `grep -rn "\[math:<id>\]" src` jumps from the row to the exact code.
3. The derivation link goes to the full treatment in the spec doc.

**Standing rule 8:** a PR that changes an `[math:*]`-anchored member updates its SymbolTable row in
the same PR. `docs-check` enforces the mechanical half.
