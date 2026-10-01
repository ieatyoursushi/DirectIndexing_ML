using System.Diagnostics;
using DirectIndexing.Core.Simulation;
using DirectIndexing.DataCollection;
using DirectIndexing.Export;
using DirectIndexing.ML;
using DirectIndexing.ML.MLNet;
using DirectIndexing.ML.MLNet.Data;
using DirectIndexing.ML.MLNet.Splits;

var mode = args.FirstOrDefault() ?? "simulate";

// ── Oracle configuration ────────────────────────────────────────────────────
// The scalarized oracle: 3 hard gates · 𝟙[U>0], U = taxValue − λσ_TE² − c_trade.
// The v0.2 gated arm (--oracle=gated) was retired in the pre-v0.3 downsizing; fail
// loudly rather than silently running the scalarized oracle under an old command.
if (args.Any(a => a.StartsWith("--oracle=")))
{
    Console.Error.WriteLine("[ERROR] --oracle was retired with the gated oracle arm (schema v4). " +
        "Findings: DataMemo/archive/RetiredComponents.md §6; code: tag archive/v0.3-pre-downsize.");
    Environment.Exit(2);
}
var oracleCfg = DirectIndexing.Core.Oracle.OracleConfig.Default;

// --ctrade=<dollars>: override the flat per-harvest friction inside U(x)
// (e.g. --ctrade=0 for the frictionless ablation arm).
var ctradeArg = args.FirstOrDefault(a => a.StartsWith("--ctrade="));
if (ctradeArg is not null && decimal.TryParse(ctradeArg["--ctrade=".Length..],
        System.Globalization.NumberStyles.Number,
        System.Globalization.CultureInfo.InvariantCulture, out var ctradeOverride))
    oracleCfg = oracleCfg with { CTrade = ctradeOverride };

// --no-reharvest-guard: drop the stricter-than-§1091 "no second harvest of a ticker within
// 30 days of its own loss sale" term from 𝒲 (v0.3-2b ablation; §1091's two sides stay).
bool noReharvestGuard = args.Contains("--no-reharvest-guard");
if (noReharvestGuard) oracleCfg = oracleCfg with { ReharvestGuard = false };

// The oracle flags only shape simulate/simulate-mc; warn instead of silently
// no-op'ing when passed to other modes (mlnet-* modes read a lots CSV as-is).
if (mode is not ("simulate" or "simulate-mc") && (ctradeArg is not null || noReharvestGuard))
    Console.WriteLine($"[WARN] --ctrade/--no-reharvest-guard have no effect on mode '{mode}' — " +
                      "it configures the simulation only. mlnet-* modes read the lots CSV as-is.");

// ── Split policy (v0.26, validation hardening) ──────────────────────────────
// --split=temporal: chronological purged splits (embargo >= 30d label horizon)
// for every mlnet-* trainer in this run. --embargo=N and --testfrac=F tune it
// (--testfrac=0.5 => the decade walk-forward). Artifacts write to
// data/artifacts-mlnet-temporal/ so random-split baselines are never clobbered.
if (args.Contains("--split=temporal"))
    SplitPolicy.Mode = SplitMode.TemporalPurged;
var embargoArg = args.FirstOrDefault(a => a.StartsWith("--embargo="));
if (embargoArg is not null && int.TryParse(embargoArg["--embargo=".Length..], out var embargoDays))
    SplitPolicy.EmbargoDays = embargoDays;
var testfracArg = args.FirstOrDefault(a => a.StartsWith("--testfrac="));
if (testfracArg is not null && double.TryParse(testfracArg["--testfrac=".Length..],
        System.Globalization.NumberStyles.Float,
        System.Globalization.CultureInfo.InvariantCulture, out var testFrac))
    SplitPolicy.TestFractionOverride = testFrac;
if (mode.StartsWith("mlnet"))
    Console.WriteLine($"[SplitPolicy] {SplitPolicy.Describe()}");

// --target=a,b,… selects the binary targets mlnet-all trains (default soft_bt,oracle; v0.3-11 adds
// soft_bt_90 = the realized 90-day hit). A label horizon h needs an embargo ≥ h (standing rule 5),
// so a 90-day target raises the temporal embargo to 90 automatically.
var targetArg = args.FirstOrDefault(a => a.StartsWith("--target="));
var mlTargets = targetArg is null ? new[] { "soft_bt", "oracle" }
                                  : targetArg["--target=".Length..].Split(',', StringSplitOptions.RemoveEmptyEntries);
foreach (var tgt in mlTargets)
    if (tgt is not ("soft_bt" or "oracle" or "soft_bt_90"))
        throw new ArgumentException($"--target entries must be soft_bt|oracle|soft_bt_90, got '{tgt}'");
if (mlTargets.Contains("soft_bt_90") && SplitPolicy.EmbargoDays < DirectIndexing.Core.Simulation.SoftLabelBuilder.WindowLong)
{
    SplitPolicy.EmbargoDays = DirectIndexing.Core.Simulation.SoftLabelBuilder.WindowLong;
    if (mode.StartsWith("mlnet")) Console.WriteLine($"[SplitPolicy] embargo raised to {SplitPolicy.EmbargoDays}d for the 90-day target");
}

// --features=no-vol drops the σ̂ feature block (SigmaHat, SigmaMkt, ZBarrier, PBarrier) from
// every trainer — the v0.3-8 role-1 ablation arm (artifacts tagged -novol). Run it under
// --split=temporal: σ̂_m is a near-injective function of the date (VolatilityModel_v03 §5).
var featuresArg = args.FirstOrDefault(a => a.StartsWith("--features="));
if (featuresArg is not null)
    DirectIndexing.ML.MLNet.Schema.FeatureLists.ExcludeVolFeatures = featuresArg["--features=".Length..] switch
    {
        "no-vol" => true,
        "all"    => false,
        var x    => throw new ArgumentException($"--features must be all|no-vol, got '{x}'"),
    };

// ── Contribution policy (v0.3 P0 — the cost-basis-aging fix) ────────────────
// --contrib enables periodic cash inflows that mint fresh lots at current prices,
// restoring the harvestable supply that the open-once book loses to aging.
// Off by default: an unflagged run is the no-contribution baseline arm.
// --contrib-allow-harvestable drops the default skip of names holding a harvestable lot.
// Tunable: --contrib-interval=N (trading days), --contrib-rate=R (annual, as a
// fraction of the INITIAL book), --contrib-names=M (most-underweight names bought).
var contribCfg = DirectIndexing.Core.Simulation.ContributionPolicy.Off;
if (args.Contains("--contrib"))
{
    contribCfg = contribCfg with { Enabled = true };
    var ciArg = args.FirstOrDefault(a => a.StartsWith("--contrib-interval="));
    if (ciArg is not null && int.TryParse(ciArg["--contrib-interval=".Length..], out var ci))
        contribCfg = contribCfg with { IntervalDays = ci };
    var crArg = args.FirstOrDefault(a => a.StartsWith("--contrib-rate="));
    if (crArg is not null && decimal.TryParse(crArg["--contrib-rate=".Length..],
            System.Globalization.NumberStyles.Number,
            System.Globalization.CultureInfo.InvariantCulture, out var cr))
        contribCfg = contribCfg with { AnnualRate = cr };
    if (args.Contains("--contrib-allow-harvestable"))
        contribCfg = contribCfg with { SkipHarvestableNames = false };
    var cnArg = args.FirstOrDefault(a => a.StartsWith("--contrib-names="));
    if (cnArg is not null && int.TryParse(cnArg["--contrib-names=".Length..], out var cn))
        contribCfg = contribCfg with { NamesPerContribution = cn };
}

// ── Risk model (v0.3-6, F1 + Q2) ─────────────────────────────────────────────
// --cov=fullsample|pit|pit-lw selects Σ̂ (default pit-lw: point-in-time Ledoit–Wolf;
// fullsample is the legacy look-ahead arm). --te-weights=names|dollars selects δw
// (default dollars). `--cov=fullsample --te-weights=names` reproduces pre-v0.3-6 output.
var riskCfg = DirectIndexing.Core.Simulation.Covariance.RiskModel.Default;
var covArg = args.FirstOrDefault(a => a.StartsWith("--cov="));
if (covArg is not null)
    riskCfg = riskCfg with { Covariance = DirectIndexing.Core.Simulation.Covariance.CovarianceFactory.Parse(covArg["--cov=".Length..]) };
var teArg = args.FirstOrDefault(a => a.StartsWith("--te-weights="));
if (teArg is not null)
    riskCfg = riskCfg with { Weighting = teArg["--te-weights=".Length..] switch
    {
        "names"   => DirectIndexing.Core.Simulation.TeWeighting.Names,
        "dollars" => DirectIndexing.Core.Simulation.TeWeighting.Dollars,
        var x     => throw new ArgumentException($"--te-weights must be names|dollars, got '{x}'"),
    } };

// ── σ̂ label role (v0.3-9) ───────────────────────────────────────────────────
// --soft-gbm=fhs fills Y_Soft_GBM by filtered historical simulation (empirical residuals,
// EWMA σ path) instead of constant-σ GBM paths. Dataset tag _softfhs.
var softGbmArg = args.FirstOrDefault(a => a.StartsWith("--soft-gbm="));
var softGbm = softGbmArg?["--soft-gbm=".Length..] switch
{
    null or "gbm" => DirectIndexing.Core.Simulation.SoftGbmMode.Gbm,
    "fhs"         => DirectIndexing.Core.Simulation.SoftGbmMode.Fhs,
    var x         => throw new ArgumentException($"--soft-gbm must be gbm|fhs, got '{x}'"),
};
var softTag = softGbm == DirectIndexing.Core.Simulation.SoftGbmMode.Fhs ? "_softfhs" : "";

// ── Sell-winner trim (v0.3-4) ────────────────────────────────────────────────
// --trim sells gain lots of names above (1 + band) × equal weight back toward target
// and reinvests the proceeds, making realized gains endogenous so the Schedule D
// ledger has gains to net. Off by default. Tunable: --trim-interval=N, --trim-band=B.
var trimCfg = DirectIndexing.Core.Simulation.TrimPolicy.Off;
if (args.Contains("--trim"))
{
    trimCfg = trimCfg with { Enabled = true };
    var tiArg = args.FirstOrDefault(a => a.StartsWith("--trim-interval="));
    if (tiArg is not null && int.TryParse(tiArg["--trim-interval=".Length..], out var ti))
        trimCfg = trimCfg with { IntervalDays = ti };
    var tbArg = args.FirstOrDefault(a => a.StartsWith("--trim-band="));
    if (tbArg is not null && decimal.TryParse(tbArg["--trim-band=".Length..],
            System.Globalization.NumberStyles.Number,
            System.Globalization.CultureInfo.InvariantCulture, out var tb))
        trimCfg = trimCfg with { Band = tb };
}
if (mode is ("simulate" or "simulate-mc") && contribCfg.Enabled)
    Console.WriteLine($"[ContributionPolicy] {contribCfg.Describe()}");
// ── Dataset + artifact layout ───────────────────────────────────────────────
// --lots=<path> picks the dataset every mlnet-* / eda / codebook mode reads
// (default data/lots.csv). The artifact directory is derived from the dataset's
// arm tag plus the split tag, so arms never clobber each other:
//   lots.csv + random          → data/artifacts-mlnet/
//   lots_contrib.csv + temporal → data/artifacts-mlnet_contrib-temporal/
// --ctrade=<x> and --no-reharvest-guard tag the simulated dataset (lots_ctrade<x>.csv,
// lots_noreharvest.csv) the same way.
var ctradeTag      = (ctradeArg is null ? "" : $"_ctrade{oracleCfg.CTrade.ToString(System.Globalization.CultureInfo.InvariantCulture)}")
                   + (noReharvestGuard ? "_noreharvest" : "");
// All default paths are anchored at the repo root (the directory holding
// DirectIndexing.sln), so `dotnet run --project src -- …` from the root and
// `cd src && dotnet run -- …` behave identically. A user-supplied --lots is
// resolved against the shell's working directory, as any CLI path would be.
var repoRoot       = PythonRunner.LocateRepoRoot();
var dataDir        = Path.Combine(repoRoot, "data");
var exportDir      = Path.Combine(repoRoot, "src", "Export");
var lotsArg        = args.FirstOrDefault(a => a.StartsWith("--lots="));
var lotsPath       = lotsArg is null ? Path.Combine(dataDir, "lots.csv")
                                     : Path.GetFullPath(lotsArg["--lots=".Length..]);
var datasetTag     = DatasetTagOf(lotsPath);
var mlnetArtifacts = Path.Combine(dataDir, $"artifacts-mlnet{datasetTag}{SplitPolicy.ArtifactTag}{DirectIndexing.ML.MLNet.Schema.FeatureLists.ArtifactTag}") + Path.DirectorySeparatorChar;
var edaOut         = Path.Combine(exportDir, "eda-mlnet");
var modelsOut      = Path.Combine(exportDir, "models-mlnet");
if (mode.StartsWith("mlnet") || mode is "codebook")
    Console.WriteLine($"[Data] lots={lotsPath}\n[Data] artifacts={mlnetArtifacts}");

switch (mode)
{
    case "download":
    {
        var sw = Stopwatch.StartNew();
        try
        {
            var apiKey = Environment.GetEnvironmentVariable("FMP_API_KEY")
                         ?? throw new InvalidOperationException(
                             "FMP_API_KEY environment variable is not set. " +
                             "Export it before running: export FMP_API_KEY=your_key_here");

            DateOnly? startDate = null;
            DateOnly? endDate = null;
            int years = 2;

            if (args.Length >= 3 && args[1] == "--from" && args[2].Length == 10)
            {
                if (DateOnly.TryParseExact(args[2], "yyyy-MM-dd", null, System.Globalization.DateTimeStyles.None, out var sd))
                    startDate = sd;
                else
                    Console.WriteLine($"[WARN] Invalid start date format: {args[2]}. Expected yyyy-MM-dd");
            }

            if (args.Length >= 5 && args[3] == "--to" && args[4].Length == 10)
            {
                if (DateOnly.TryParseExact(args[4], "yyyy-MM-dd", null, System.Globalization.DateTimeStyles.None, out var ed))
                    endDate = ed;
                else
                    Console.WriteLine($"[WARN] Invalid end date format: {args[4]}. Expected yyyy-MM-dd");
            }

            if (startDate.HasValue != endDate.HasValue)
                throw new InvalidOperationException("Both --from and --to dates must be specified together, or neither.");

            await new MarketDataDownloader(apiKey)
                .DownloadAllHistoricalData(Path.Combine(dataDir, "raw"), years: years, startDate: startDate, endDate: endDate);

            sw.Stop();
            Console.WriteLine($"[download] Completed in {sw.Elapsed.TotalMinutes:F2} minutes ({sw.Elapsed.TotalSeconds:F0}s)");
        }
        catch (InvalidOperationException ex)
        {
            sw.Stop();
            Console.WriteLine($"[ERROR] {ex.Message}");
            if (ex.InnerException != null)
                Console.WriteLine($"[ERROR] Details: {ex.InnerException.Message}");
            Console.WriteLine($"[download] Failed after {sw.Elapsed.TotalMinutes:F2} minutes ({sw.Elapsed.TotalSeconds:F0}s)");
            Environment.ExitCode = 1;
        }
    }
    break;
    case "simulate":
    {
        var sw = Stopwatch.StartNew();
        var loader = new PriceLoader();
        loader.Load(Path.Combine(dataDir, "raw"), Path.Combine(dataDir, "constituents.json"));

        var engine    = new SimulationEngine(loader, oracleCfg, contribCfg, trimCfg, riskCfg);
        var snapshots = engine.Run(initialPortfolioValue: 10_000_000m);

        var softLabeller = new SoftLabelBuilder(loader, oracleCfg, softGbm);
        softLabeller.Label(snapshots);

        var outPath = Path.Combine(dataDir, $"lots{contribCfg.DatasetTag}{trimCfg.DatasetTag}{riskCfg.DatasetTag}{softTag}{ctradeTag}.csv");
        SimulationExporter.WriteCsv(snapshots, outPath);
        sw.Stop();
        Console.WriteLine($"[simulate] → {outPath}  " +
                          $"({sw.Elapsed.TotalMinutes:F2} minutes, {sw.Elapsed.TotalSeconds:F0}s)");
    }
    break;
    // Synthetic world: the SAME SimulationEngine over a GBM price source (the second
    // price source; the RL environment's episode generator). σ per name is calibrated
    // from the real cache's trailing 60 days, or — with --mc-standalone=<names> — a
    // uniform universe that needs no data at all (smoke tests, stress runs).
    // Flags: --mc-days=N (default 504), --mc-seed=N (default 42), --mc-sigma=S
    // (standalone only, default 0.25). Oracle/contribution flags apply as in simulate.
    case "simulate-mc":
    {
        int    mcDays  = IntFlag("--mc-days=", 504);
        int    mcSeed  = IntFlag("--mc-seed=", 42);
        int    mcNames = IntFlag("--mc-standalone=", 0);
        float  mcSigma = (float)DoubleFlag("--mc-sigma=", 0.25);

        List<(string Symbol, string Sector, float AnnualSigma)> universe;
        if (mcNames > 0)
            universe = PriceLoader.UniformGbmUniverse(mcNames, mcSigma);
        else
        {
            var real = new PriceLoader();
            real.Load(Path.Combine(dataDir, "raw"), Path.Combine(dataDir, "constituents.json"));
            universe = PriceLoader.CalibrateGbmUniverse(real);
        }

        // --world=fhs (v0.3-10): a filtered-historical-simulation world — clustered σ, fat tails,
        // the source's cross-sectional correlation. Source = the real cache, or (standalone) a
        // GARCH-factor panel. Otherwise the GBM world.
        string worldArg = args.FirstOrDefault(a => a.StartsWith("--world="))?["--world=".Length..] ?? "gbm";
        PriceLoader synthetic;
        string worldTag;
        if (worldArg == "fhs")
        {
            PriceLoader source;
            if (mcNames > 0) source = PriceLoader.GarchFactorPanel(mcNames, 3000, mcSeed + 1);
            else
            {
                source = new PriceLoader();
                source.Load(Path.Combine(dataDir, "raw"), Path.Combine(dataDir, "constituents.json"));
            }
            synthetic = PriceLoader.FromFhs(source, mcDays, mcSeed);
            worldTag  = "-fhs";
        }
        else if (worldArg == "gbm") { synthetic = PriceLoader.FromGbm(universe, mcDays, mcSeed); worldTag = ""; }
        else throw new ArgumentException($"--world must be gbm|fhs, got '{worldArg}'");
        var snapshots = new SimulationEngine(synthetic, oracleCfg, contribCfg, trimCfg, riskCfg).Run(10_000_000m);
        new SoftLabelBuilder(synthetic, oracleCfg, softGbm).Label(snapshots);
        SimulationExporter.WriteCsv(snapshots,
            Path.Combine(dataDir, $"lots-mc{worldTag}{contribCfg.DatasetTag}{trimCfg.DatasetTag}{riskCfg.DatasetTag}{softTag}{ctradeTag}.csv"));
    }
    break;
    // σ̂ forecast evaluation (v0.3-7): QLIKE per estimator × horizon × market-vol tercile.
    // World: the real cache by default; --mc-standalone=N [--mc-days --mc-seed --mc-sigma]
    // for a GBM world (the control: constant σ, so the constant estimator should win).
    // → data/artifacts-vol/qlike{-mc}.json
    case "vol-eval":
    {
        int mcNames = IntFlag("--mc-standalone=", 0);
        PriceLoader world;
        string tag;
        if (mcNames > 0)
        {
            world = PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(mcNames, (float)DoubleFlag("--mc-sigma=", 0.25)),
                                        IntFlag("--mc-days=", 504), IntFlag("--mc-seed=", 42));
            tag = "-mc";
        }
        else
        {
            world = new PriceLoader();
            world.Load(Path.Combine(dataDir, "raw"), Path.Combine(dataDir, "constituents.json"));
            tag = "";
        }
        var result = DirectIndexing.Core.Simulation.Volatility.VolEval.Run(world);
        var outPath = Path.Combine(dataDir, "artifacts-vol", $"qlike{tag}.json");
        DirectIndexing.Core.Simulation.Volatility.VolEval.Write(result, outPath, tag == "" ? "real" : "gbm");
        foreach (var (h, cells) in result)
        {
            Console.WriteLine($"[vol-eval] h={h,2}  " + string.Join("  ", cells.OrderBy(c => c.Value.Mean)
                .Select(c => $"{c.Key}={c.Value.Mean:F4} (L {c.Value.Low:F3} M {c.Value.Mid:F3} H {c.Value.High:F3})")));
        }
        Console.WriteLine($"[vol-eval] → {outPath}");
    }
    break;
    // The economic ladder (v0.3-11): the same world under rung 1 (never harvest), rung 2 (naive
    // loss threshold) and rung 3 (the oracle) — engine only, no soft labels. World flags as in
    // simulate-mc (--mc-standalone/--mc-days/--mc-seed, --world=fhs) or the real cache by default;
    // every simulation flag (--contrib, --trim, --cov, …) applies to all rungs.
    // → data/runs/ladder{tags}/<policy>.json and a printed table.
    // [math:ladder] — DataMemo/spec/SymbolTable.md
    case "ladder":
    {
        int mcNames = IntFlag("--mc-standalone=", 0);
        int mcDays  = IntFlag("--mc-days=", 1260);
        int mcSeed  = IntFlag("--mc-seed=", 42);
        int seeds   = Math.Max(1, IntFlag("--seeds=", 1));
        bool fhsWorld = args.Contains("--world=fhs");
        PriceLoader? real = null;
        if (mcNames == 0)
        {
            real = new PriceLoader();
            real.Load(Path.Combine(dataDir, "raw"), Path.Combine(dataDir, "constituents.json"));
            if (!fhsWorld) seeds = 1;   // one real history
        }
        string worldTag = (mcNames > 0 ? "-mc" : "") + (fhsWorld ? "-fhs" : "");
        string runDir = Path.Combine(dataDir, "runs",
            $"ladder{worldTag}{contribCfg.DatasetTag}{trimCfg.DatasetTag}{riskCfg.DatasetTag}{ctradeTag}");
        var rungs = new DirectIndexing.Core.Policy.IHarvestPolicy[]
        {
            new DirectIndexing.Core.Policy.NeverHarvestPolicy(),
            new DirectIndexing.Core.Policy.ThresholdHarvestPolicy(),
            new DirectIndexing.Core.Policy.OraclePolicy(),
        };
        // results[rung][seed]
        var results = rungs.Select(_ => new List<DirectIndexing.Core.Policy.RunMetrics>()).ToArray();
        for (int k = 0; k < seeds; k++)
        {
            int seed = mcSeed + k;
            PriceLoader world = mcNames > 0
                ? (fhsWorld ? PriceLoader.FromFhs(PriceLoader.GarchFactorPanel(mcNames, 3000, seed + 1000), mcDays, seed)
                            : PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(mcNames, (float)DoubleFlag("--mc-sigma=", 0.25)), mcDays, seed))
                : (fhsWorld ? PriceLoader.FromFhs(real!, mcDays, seed) : real!);
            for (int r = 0; r < rungs.Length; r++)
            {
                var engine = new SimulationEngine(world, oracleCfg, contribCfg, trimCfg, riskCfg, rungs[r]);
                engine.Run(10_000_000m);
                var m = engine.Metrics();
                m.Write(Path.Combine(runDir, $"seed{seed}", $"{rungs[r].Name}.json"));
                results[r].Add(m);
            }
        }
        // per-seed paired differences vs rung 1 — the paths are shared within a seed, so the
        // pairing removes the market path's common noise; s.e. across seeds
        static (double Mean, double Se) Stat(IReadOnlyList<double> x)
        {
            double m = x.Average();
            double se = x.Count > 1 ? Math.Sqrt(x.Sum(v => (v - m) * (v - m)) / (x.Count - 1) / x.Count) : double.NaN;
            return (m, se);
        }
        Console.WriteLine();
        Console.WriteLine($"[ladder] {seeds} seed(s); differences vs rung 1 (never), mean ± s.e. across seeds");
        Console.WriteLine($"{"rung",-10} {"Δ after-tax",22} {"Δ liquidation",22} {"Δ pre-tax wealth",22} {"Δ W_tax",20} {"loss sales",11} {"TE realized",12} {"turnover",9} {"wash",5}");
        for (int r = 0; r < rungs.Length; r++)
        {
            var at  = Stat(results[r].Select((m, k) => (double)(m.AfterTaxWealth - results[0][k].AfterTaxWealth)).ToList());
            var liq = Stat(results[r].Select((m, k) => (double)(m.LiquidationValue - results[0][k].LiquidationValue)).ToList());
            var pre = Stat(results[r].Select((m, k) => (double)((m.TerminalHoldings + m.PendingReopenCash + m.Cash) - (results[0][k].TerminalHoldings + results[0][k].PendingReopenCash + results[0][k].Cash))).ToList());
            var wt  = Stat(results[r].Select((m, k) => (double)(m.TaxPosition - results[0][k].TaxPosition)).ToList());
            Console.WriteLine($"{rungs[r].Name,-10} {at.Mean,11:N0} ± {at.Se,8:N0} {liq.Mean,11:N0} ± {liq.Se,8:N0} {pre.Mean,11:N0} ± {pre.Se,8:N0} {wt.Mean,10:N0} ± {wt.Se,7:N0} " +
                              $"{results[r].Average(m => m.LossSales),11:N0} {results[r].Average(m => m.RealizedTe),12:P2} {results[r].Average(m => m.AnnualTurnover),9:P1} {results[r].Sum(m => m.WashViolations),5}");
        }
        Console.WriteLine($"[ladder] → {runDir}");
    }
    break;
    // ── ML.NET layer — typed, in-process supervised pipeline (GBT + logistic) ──
    // Each case loads the --lots dataset into List<LotStateVector>, then hands it
    // straight to LoadFromEnumerable. No CSV inside ML.NET, no [LoadColumn]
    // round-trip — the typed schema flows all the way through.
    case "mlnet-eda":
    {
        var rc = PythonRunner.Run("scripts.eda", "--in", lotsPath, "--out", edaOut);
        Environment.ExitCode = rc;
    }
    break;
    // ── Individual model cases (full CV + test eval for a single model) ──────
    case "mlnet-gbt":
    {
        var data = LotStateVectorCsvReader.Read(lotsPath);
        MLnetPipeline.RunSupervisedModel("gbt", data, "soft_bt", mlnetArtifacts);
        MLnetPipeline.RunSupervisedModel("gbt", data, "oracle",  mlnetArtifacts);
    }
    break;
    case "mlnet-logistic":
    {
        var data = LotStateVectorCsvReader.Read(lotsPath);
        MLnetPipeline.RunSupervisedModel("logistic", data, "soft_bt", mlnetArtifacts);
        MLnetPipeline.RunSupervisedModel("logistic", data, "oracle",  mlnetArtifacts);
    }
    break;

    // ── Comparison run: CV GBT + logistic → leaderboard → test eval of both ───
    case "mlnet-compare":
    {
        var data = LotStateVectorCsvReader.Read(lotsPath);
        MLnetPipeline.RunAllSupervised(data, target: "soft_bt", artifactsDir: mlnetArtifacts);
        MLnetPipeline.RunAllSupervised(data, target: "oracle",  artifactsDir: mlnetArtifacts);
    }
    break;

    // Regenerate a single target's champion-selection artifacts (leaderboard + champion
    // test eval). Used to finish a target after a partial mlnet-all/compare run.
    case "mlnet-soft":
    {
        var data = LotStateVectorCsvReader.Read(lotsPath);
        MLnetPipeline.RunAllSupervised(data, target: "soft_bt", artifactsDir: mlnetArtifacts);
    }
    break;
    case "mlnet-oracle":
    {
        var data = LotStateVectorCsvReader.Read(lotsPath);
        MLnetPipeline.RunAllSupervised(data, target: "oracle", artifactsDir: mlnetArtifacts);
    }
    break;
    // Continuous regression on Y_TaxValue (v0.25, issue #17 family). Excludes the
    // TaxValue feature — the target IS TaxValue by construction, so the task is
    // recovering g(ledger, H, L) from raw features.
    case "mlnet-tax":
    {
        var data = LotStateVectorCsvReader.Read(lotsPath);
        DirectIndexing.ML.MLNet.Models.TaxValueRegressionPipeline.Run(
            data, artifactsDir: mlnetArtifacts);
    }
    break;

    case "mlnet-render":
    {
        var rc = MLnetPipeline.RunRender(lotsPath, mlnetArtifacts, edaOut, modelsOut);
        Environment.ExitCode = rc;
    }
    break;
    case "mlnet-all":
    {
        var sw = Stopwatch.StartNew();
        var data = LotStateVectorCsvReader.Read(lotsPath);
        // Comparison run: CV GBT + logistic, leaderboard names the champion, test eval both.
        foreach (var target in mlTargets)
            MLnetPipeline.RunAllSupervised(data, target: target, artifactsDir: mlnetArtifacts);
        var rc = MLnetPipeline.RunRender(lotsPath, mlnetArtifacts, edaOut, modelsOut);
        sw.Stop();
        Console.WriteLine($"[mlnet-all] Completed in {sw.Elapsed.TotalMinutes:F2} minutes ({sw.Elapsed.TotalSeconds:F0}s)");
        Environment.ExitCode = rc;
    }
    break;

    // ── Codebook — the column dictionary, with the schema-drift assert ─────────
    // Renders scripts/codebook_schema.py (the single schema source) against the
    // --lots CSV header and FAILS if they differ. (The course report/submission
    // commands were retired; their outputs are frozen in src/Export/report/.)
    case "codebook":
    {
        var rc = PythonRunner.Run("scripts.codebook",
            "--lots", lotsPath,
            "--out",  Path.Combine(exportDir, "codebook"));
        Environment.ExitCode = rc;
    }
    break;
    // ── docs-check — the math ↔ code ↔ test spine (DataMemo/spec/SymbolTable.md) ──
    // Fails (exit 1) on: a [math:<id>] tag with no SymbolTable row, a live row with no
    // tag, a named code/test member that no longer exists, a §K constant that differs
    // from its declaration, schema-order drift, or a broken relative markdown link.
    case "docs-check":
    {
        var rc = PythonRunner.Run("scripts.check_math_sync");
        Environment.ExitCode = rc;
    }
    break;

    // ── DevTools — dependency/coupling atlas of this C# layer ────────────────
    // Brute-force regex scan of src/**/*.cs (same approach as Zombtoy
    // DevTools/Diagrams) → one markdown file of mermaid diagrams + tables.
    case "deps":
    {
        var rc = PythonRunner.Run("scripts.dependencies",
            "--src", Path.Combine(repoRoot, "src"),
            "--out", Path.Combine(exportDir, "diagrams"));
        Environment.ExitCode = rc;
    }
    break;

    case "test":
    {
        // v0.1 smoke tests — simple Debug.Assert runners.
        // Move to a proper xUnit/NUnit project when the simulation layer is added.

        var portfolioTests = new PortfolioStateTests();
        portfolioTests.Test_HarvestLoss_DecreasesRealizedGains();
        portfolioTests.Test_WashSaleClock_StartsAtZeroAfterHarvest();
        portfolioTests.Test_YearEnd_BanksNetLoss_AndClocksPersist();

        var ledgerTests = new TaxLedgerTests();
        ledgerTests.Test_LedgerNet_AccumulatesSignedRealized();
        ledgerTests.Test_RollYearEnd_BanksExcessLoss();
        ledgerTests.Test_OffsetBudget_And_Capacity_DrawDown();
        ledgerTests.Test_ComputeTaxValue_CapacitySplit_And_Rates();
        ledgerTests.Test_PortfolioState_RoutesThroughLedger();
        ledgerTests.Test_IsLongTerm_CalendarEdges();

        var scalarizedTests = new OracleScalarizedTests();
        scalarizedTests.Test_Fires_WithoutRealizedGains();
        scalarizedTests.Test_Blocked_WhenUtilityNegative();
        scalarizedTests.Test_TradeOff_TaxValueVsTrackingError();
        scalarizedTests.Test_HardCeiling_BindsInPathologicalRegimes();
        scalarizedTests.Test_LossAndWashGates_StillBind();
        scalarizedTests.Test_WashGate_OpensAfterInclusiveWindow();
        scalarizedTests.Test_SnapshotOverload_MatchesScalarForm();
        scalarizedTests.Test_Utility_Arithmetic_And_CTrade();

        var teTests = new TrackingErrorProxyTests();
        teTests.Test_SigmaTE_Zero_WhenPortfolioEqualsFullBenchmark();
        teTests.Test_SigmaTE_Positive_WhenPortfolioExcludesDivergingTicker();
        teTests.Test_SigmaTE_StaysBounded_AfterStructuralLotRemoval();
        teTests.Test_SigmaTE_Positive_ForAntiCorrelatedUniverse();

        var gbmTests = new GbmSimulatorTests();
        gbmTests.Test_SimulatePaths_AllPricesPositive();
        gbmTests.Test_SimulatePaths_SeedPriceAtStepZero();
        gbmTests.Test_FractionFiring_Zero_WhenConditionNeverMet();
        gbmTests.Test_FractionFiring_One_WhenConditionAlwaysFires();
        gbmTests.Test_FractionFiring_InRange_ForRealisticPredicate();
        gbmTests.Test_NextGaussian_NearStandardNormal();

        var synthTests = new SyntheticWorldTests();
        synthTests.Test_Calendar_IsWeekdaysAndCrossesYearEnd();
        synthTests.Test_Deterministic_ForSeed();
        synthTests.Test_RealisedVol_MatchesSigma();
        synthTests.Test_CanonicalEngine_RunsOnSyntheticWorld();

        // ── ML.NET layer tests ──────────────────────────────────────────────
        new LotStateVectorCsvReaderTests().Test_RoundTrip_PreservesAllFields();
        new StratifiedSplitTests().Test_PreservesClassProportionWithin1Percent();
        new StratifiedKFoldTests().Test_FoldsPartitionDataAndContainPositives();

        var washTests = new WashSaleTests();
        washTests.Test_Audit_WindowEdges_SameLot_AndGains();
        washTests.Test_State_BeforeSide_LotLevelClock();
        washTests.Test_Engine_ZeroViolations_OnWorldsThatHadThem();
        washTests.Test_ReharvestGuard_Off_IsStillLawful();

        var covTests = new CovarianceTests();
        covTests.Test_LedoitWolf_RepairsRank_IntensityShrinksWithT();
        covTests.Test_PointInTime_IgnoresTheFuture();
        covTests.Test_DollarWeights_SeePositionSize();

        var volTests = new VolatilityTests();
        volTests.Test_Ewma_Recursion_And_QLike_Floor();
        volTests.Test_Estimators_AreCausal_FitRejectsLookAhead();
        volTests.Test_Garch_RecoversParams_AndWinsOnClusteredData();
        volTests.Test_Barrier_MatchesMonteCarlo();

        var fhsTests = new FhsTests();
        fhsTests.Test_FhsWorld_Clusters_FatTails_Correlation_Deterministic();
        fhsTests.Test_FhsLabels_Causal_And_AgreeWithGbmOnGbm();

        var policyTests = new PolicyTests();
        policyTests.Test_OraclePolicy_IsDefault_NeverHarvest_NeverSells();
        policyTests.Test_RunMetrics_Identities();
        policyTests.Test_Wealth_IsConserved_WhenReopenRunsPastTheEnd();

        var trimTests = new TrimTests();
        trimTests.Test_Trim_SellsOnlyGains_ConsumesCarryforward();

        var contribTests = new ContributionPolicyTests();
        contribTests.Test_DefaultIsDisabled();
        contribTests.Test_AmountProRatedOverInterval();
        contribTests.Test_ScheduleIsExogenous();
        contribTests.Test_EnabledTagsDataset();

        var temporalTests = new TemporalSplitTests();
        temporalTests.Test_TrainTest_BoundaryAndEmbargo();
        temporalTests.Test_TrainTest_PurgeAccounting();
        temporalTests.Test_PurgedFolds_EmbargoBothSides();
        temporalTests.Test_Deterministic();
        temporalTests.Test_DataSplit_PolicyDispatch();
        var preprocessing = new PreprocessingTests();
        preprocessing.Test_MedianImputerReplacesNaNs();
        preprocessing.Test_ClassWeightsBalanced();
        new GridSearchTests().Test_PicksLargestCForLinearlySeparableData();

        // ── Model-suite tests ───────────────────────────────────────────────
        new GbtTrainerTests().Test_GbtCv_SeparableData();
        var metricTests = new BinaryMetricsTests();
        metricTests.Test_AveragePrecision_And_Roc_HandComputed();
        metricTests.Test_PerfectRanker_IsOne_AtAnyPrevalence();
        metricTests.Test_NoSkill_Floors_RocHalf_PrEqualsPrevalence();

        var championTests = new ChampionSelectionTests();
        championTests.Test_SelectChampion_IsArgmaxOfCv();
        championTests.Test_GbtBeatsLogisticOnNonLinearTarget();

        Console.WriteLine("All tests passed.");
    }
    break;
}

// ── Flag helpers ─────────────────────────────────────────────────────────────
int IntFlag(string prefix, int fallback)
{
    var a = args.FirstOrDefault(x => x.StartsWith(prefix));
    return a is not null && int.TryParse(a[prefix.Length..], out var v) ? v : fallback;
}

double DoubleFlag(string prefix, double fallback)
{
    var a = args.FirstOrDefault(x => x.StartsWith(prefix));
    return a is not null && double.TryParse(a[prefix.Length..],
        System.Globalization.NumberStyles.Float,
        System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : fallback;
}

// "lots.csv" → "", "lots_contrib.csv" → "_contrib", "lots-mc.csv" → "-mc",
// any other file name → "_<stem>" (so a renamed dataset still gets its own dir).
static string DatasetTagOf(string lotsCsv)
{
    var stem = Path.GetFileNameWithoutExtension(lotsCsv);
    return stem.StartsWith("lots") ? stem["lots".Length..] : "_" + stem;
}
