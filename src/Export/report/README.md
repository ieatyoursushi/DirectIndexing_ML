# Frozen: the PSTAT 231 submission report (v0.2)

These files are the **historical course deliverable** — the 2-year-window report, codebook,
and executed notebook exactly as submitted. They are no longer regenerated: the report and
submission commands (`report`, `report-all`, `submission`) and their scripts were retired in
the pre-v0.3 downsizing (`DataMemo/archive/RetiredComponents.md` §7). The code that produced
them is recoverable from the tag `archive/v0.3-pre-downsize`.

Note the schema here is **v2/v3** (it predates schema v4) and the numbers are the v0.2
two-year run. For the live column dictionary, run `dotnet run --project src -- codebook`
(renders `src/ML/Python/scripts/codebook_schema.py`, the single schema source, and asserts it
against the dataset header). The economic report returns with the v0.3 metric ladder.
