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

// The oracle flags only shape simulate/simulate-mc; warn instead of silently
// no-op'ing when passed to other modes (mlnet-* modes read a lots CSV as-is).
if (mode is not ("simulate" or "simulate-mc") && ctradeArg is not null)
    Console.WriteLine($"[WARN] --ctrade has no effect on mode '{mode}' — " +
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

// ── Contribution policy (v0.3 P0 — the cost-basis-aging fix) ────────────────
// --contrib enables periodic cash inflows that mint fresh lots at current prices,
// restoring the harvestable supply that the open-once book loses to aging.
// Off by default so an unflagged run reproduces v0.26 byte-for-byte.
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
    var cnArg = args.FirstOrDefault(a => a.StartsWith("--contrib-names="));
    if (cnArg is not null && int.TryParse(cnArg["--contrib-names=".Length..], out var cn))
        contribCfg = contribCfg with { NamesPerContribution = cn };
}
if (mode is ("simulate" or "simulate-mc") && contribCfg.Enabled)
    Console.WriteLine($"[ContributionPolicy] {contribCfg.Describe()}");
// ── Dataset + artifact layout ───────────────────────────────────────────────
// --lots=<path> picks the dataset every mlnet-* / eda / codebook mode reads
// (default data/lots.csv). The artifact directory is derived from the dataset's
// arm tag plus the split tag, so arms never clobber each other:
//   lots.csv + random          → data/artifacts-mlnet/
//   lots_contrib.csv + temporal → data/artifacts-mlnet_contrib-temporal/
// --ctrade=<x> tags the simulated dataset (lots_ctrade<x>.csv) the same way.
var ctradeTag      = ctradeArg is null ? "" : $"_ctrade{oracleCfg.CTrade.ToString(System.Globalization.CultureInfo.InvariantCulture)}";
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
var mlnetArtifacts = Path.Combine(dataDir, $"artifacts-mlnet{datasetTag}{SplitPolicy.ArtifactTag}") + Path.DirectorySeparatorChar;
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

        var engine    = new SimulationEngine(loader, oracleCfg, contribCfg);
        var snapshots = engine.Run(initialPortfolioValue: 10_000_000m);

        var softLabeller = new SoftLabelBuilder(loader, oracleCfg);
        softLabeller.Label(snapshots);

        var outPath = Path.Combine(dataDir, $"lots{contribCfg.DatasetTag}{ctradeTag}.csv");
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

        var synthetic = PriceLoader.FromGbm(universe, mcDays, mcSeed);
        var snapshots = new SimulationEngine(synthetic, oracleCfg, contribCfg).Run(10_000_000m);
        new SoftLabelBuilder(synthetic, oracleCfg).Label(snapshots);
        SimulationExporter.WriteCsv(snapshots,
            Path.Combine(dataDir, $"lots-mc{contribCfg.DatasetTag}{ctradeTag}.csv"));
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
        MLnetPipeline.RunAllSupervised(data, target: "soft_bt", artifactsDir: mlnetArtifacts);
        MLnetPipeline.RunAllSupervised(data, target: "oracle",  artifactsDir: mlnetArtifacts);
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

        var scalarizedTests = new OracleScalarizedTests();
        scalarizedTests.Test_Fires_WithoutRealizedGains();
        scalarizedTests.Test_Blocked_WhenUtilityNegative();
        scalarizedTests.Test_TradeOff_TaxValueVsTrackingError();
        scalarizedTests.Test_HardCeiling_BindsInPathologicalRegimes();
        scalarizedTests.Test_LossAndWashGates_StillBind();
        scalarizedTests.Test_WashGate_OpensExactlyAtBoundary();
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
