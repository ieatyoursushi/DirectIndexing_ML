using System.Diagnostics;
using DirectIndexing.Core.Simulation;
using DirectIndexing.DataCollection;
using DirectIndexing.Export;
using DirectIndexing.ML;
using DirectIndexing.ML.MLNet;
using DirectIndexing.ML.MLNet.Data;

var mode = args.FirstOrDefault() ?? "simulate";

// ── Oracle ablation flag (v0.25, issue #23) ─────────────────────────────────
// --oracle=scalarized (default): 3 hard gates · 𝟙[U>0], no external-gains seed.
// --oracle=gated: v0.2-legacy 4-gate oracle incl. G_YTD>0 + seed — the ablation
// baseline. Datasets are kept apart (lots.csv vs lots_gated.csv) because the
// ACTING oracle changes the trajectory itself; they are separate runs, not
// two label columns of one run.
var oracleCfg = args.Contains("--oracle=gated")
    ? DirectIndexing.Core.Oracle.OracleConfig.Gated
    : DirectIndexing.Core.Oracle.OracleConfig.Scalarized;

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
                .DownloadAllHistoricalData("../data/raw", years: years, startDate: startDate, endDate: endDate);

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
        loader.Load("../data/raw", "../data/constituents.json");

        var engine    = new SimulationEngine(loader, oracleCfg);
        var snapshots = engine.Run(initialPortfolioValue: 10_000_000m);

        var softLabeller = new SoftLabelBuilder(loader, oracleCfg);
        softLabeller.Label(snapshots);

        var outPath = $"../data/lots{oracleCfg.DatasetTag}.csv";
        SimulationExporter.WriteCsv(snapshots, outPath);
        sw.Stop();
        Console.WriteLine($"[simulate] oracle={oracleCfg.Mode} → {outPath}  " +
                          $"({sw.Elapsed.TotalMinutes:F2} minutes, {sw.Elapsed.TotalSeconds:F0}s)");
    }
    break;
    // Alternate: Monte Carlo simulation with synthetic GBM prices.
    // Calibrates per-stock σ from the same real data then generates N days of GBM paths.
    // Produces data/lots-mc.csv alongside the backtesting data/lots.csv.
    case "simulate-mc":
    {
        var loader = new PriceLoader();
        loader.Load("../data/raw", "../data/constituents.json");

        var mcEngine  = new MonteCarloEngine(loader, annualDrift: 0f);
        var snapshots = mcEngine.Run(
            initialPortfolioValue: 10_000_000m,
            simDays:    504,
            warmupDays: 200,
            seed:       42,
            oracleConfig: oracleCfg);

        SimulationExporter.WriteCsv(snapshots, $"../data/lots-mc{oracleCfg.DatasetTag}.csv");
    }
    break;
    case "train": throw new NotImplementedException("Training not yet built — use mlnet-supervised.");

    // ── ML.NET layer — typed, in-process supervised/unsupervised pipeline ─────
    // Each case loads data/lots.csv into List<LotStateVector>, then hands it
    // straight to LoadFromEnumerable. No CSV inside ML.NET, no [LoadColumn]
    // round-trip — the typed schema flows all the way through.
    case "mlnet-eda":
    {
        var rc = PythonRunner.Run("scripts.eda",
            "--in",  "../../../data/lots.csv",
            "--out", "../../Export/eda-mlnet/");
        Environment.ExitCode = rc;
    }
    break;
    case "mlnet-unsupervised":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunUnsupervised(data, "../data/artifacts-mlnet/");
    }
    break;
    case "mlnet-supervised":   // PRIMARY: logistic on Y_Soft_BT (backward compat)
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunSupervised(data, target: "soft_bt", artifactsDir: "../data/artifacts-mlnet/");
    }
    break;
    case "mlnet-baseline":     // SANITY: logistic on Y_Oracle (backward compat)
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunSupervised(data, target: "oracle", artifactsDir: "../data/artifacts-mlnet/");
    }
    break;

    // ── Individual model cases (full CV + test eval for a single model) ──────
    case "mlnet-gbt":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunSupervisedModel("gbt", data, "soft_bt", "../data/artifacts-mlnet/");
        MLnetPipeline.RunSupervisedModel("gbt", data, "oracle",  "../data/artifacts-mlnet/");
    }
    break;
    case "mlnet-rf":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunSupervisedModel("rf", data, "soft_bt", "../data/artifacts-mlnet/");
        MLnetPipeline.RunSupervisedModel("rf", data, "oracle",  "../data/artifacts-mlnet/");
    }
    break;
    case "mlnet-elnet":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunSupervisedModel("elnet", data, "soft_bt", "../data/artifacts-mlnet/");
        MLnetPipeline.RunSupervisedModel("elnet", data, "oracle",  "../data/artifacts-mlnet/");
    }
    break;
    case "mlnet-linreg":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunSupervisedModel("linreg", data, "soft_bt", "../data/artifacts-mlnet/");
        MLnetPipeline.RunSupervisedModel("linreg", data, "oracle",  "../data/artifacts-mlnet/");
    }
    break;

    // ── Champion-selection run: CV all models → best 1-2 get test eval ───────
    case "mlnet-compare":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunAllSupervised(data, target: "soft_bt", artifactsDir: "../data/artifacts-mlnet/");
        MLnetPipeline.RunAllSupervised(data, target: "oracle",  artifactsDir: "../data/artifacts-mlnet/");
    }
    break;

    // Regenerate a single target's champion-selection artifacts (leaderboard + champion
    // test eval). Used to finish a target after a partial mlnet-all/compare run.
    case "mlnet-soft":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunAllSupervised(data, target: "soft_bt", artifactsDir: "../data/artifacts-mlnet/");
    }
    break;
    case "mlnet-oracle":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunAllSupervised(data, target: "oracle", artifactsDir: "../data/artifacts-mlnet/");
    }
    break;
    // Continuous regression on Y_TaxValue (v0.25, issue #17 family). Excludes the
    // TaxValue feature — the target IS TaxValue by construction, so the task is
    // recovering g(ledger, H, L) from raw features.
    case "mlnet-tax":
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        DirectIndexing.ML.MLNet.Models.TaxValueRegressionPipeline.Run(
            data, artifactsDir: "../data/artifacts-mlnet/");
    }
    break;

    case "mlnet-render":
    {
        var rc = MLnetPipeline.RunRender(
            lotsCsv:      "../../../data/lots.csv",
            artifactsDir: "../../../data/artifacts-mlnet/",
            edaDir:       "../../Export/eda-mlnet/",
            modelsDir:    "../../Export/models-mlnet/");
        Environment.ExitCode = rc;
    }
    break;
    case "mlnet-all":
    {
        var sw = Stopwatch.StartNew();
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunUnsupervised(data, "../data/artifacts-mlnet/");
        // Champion-selection: CV all supervised models, full eval for best 1-2 + linreg demonstration.
        MLnetPipeline.RunAllSupervised(data, target: "soft_bt", artifactsDir: "../data/artifacts-mlnet/");
        MLnetPipeline.RunAllSupervised(data, target: "oracle",  artifactsDir: "../data/artifacts-mlnet/");
        var rc = MLnetPipeline.RunRender(
            lotsCsv:      "../../../data/lots.csv",
            artifactsDir: "../../../data/artifacts-mlnet/",
            edaDir:       "../../Export/eda-mlnet/",
            modelsDir:    "../../Export/models-mlnet/");
        sw.Stop();
        Console.WriteLine($"[mlnet-all] Completed in {sw.Elapsed.TotalMinutes:F2} minutes ({sw.Elapsed.TotalSeconds:F0}s)");
        Environment.ExitCode = rc;
    }
    break;

    // ── Report layer — final-project report (notebook + HTML + codebook) ─────
    // Python renders; preflight fails fast (exit 2) if ML artifacts are missing.
    case "report":
    {
        var rc = PythonRunner.Run("scripts.report",
            "--lots",      "../../../data/lots.csv",
            "--artifacts", "../../../data/artifacts-mlnet/",
            "--notebook",  "notebooks/final_report.ipynb",
            "--out",       "../../Export/report/");
        Environment.ExitCode = rc;
    }
    break;
    case "report-all":   // mlnet-all training + report, one command
    {
        var data = LotStateVectorCsvReader.Read("../data/lots.csv");
        MLnetPipeline.RunUnsupervised(data, "../data/artifacts-mlnet/");
        MLnetPipeline.RunAllSupervised(data, target: "soft_bt", artifactsDir: "../data/artifacts-mlnet/");
        MLnetPipeline.RunAllSupervised(data, target: "oracle",  artifactsDir: "../data/artifacts-mlnet/");
        var rc = PythonRunner.Run("scripts.report",
            "--lots",      "../../../data/lots.csv",
            "--artifacts", "../../../data/artifacts-mlnet/",
            "--notebook",  "notebooks/final_report.ipynb",
            "--out",       "../../Export/report/");
        Environment.ExitCode = rc;
    }
    break;
    // ── DevTools — dependency/coupling atlas of this C# layer ────────────────
    // Brute-force regex scan of src/**/*.cs (same approach as Zombtoy
    // DevTools/Diagrams) → one markdown file of mermaid diagrams + tables.
    case "deps":
    {
        var rc = PythonRunner.Run("scripts.dependencies",
            "--src", "../../../src",
            "--out", "../../Export/diagrams/");
        Environment.ExitCode = rc;
    }
    break;

    case "submission":   // package the course submission zip at the repo root
    {
        var rc = PythonRunner.Run("scripts.package_submission",
            "--repo-root", "../../..",
            "--out",       "../../../submission.zip");
        Environment.ExitCode = rc;
    }
    break;

    case "test":
    {
        // v0.1 smoke tests — simple Debug.Assert runners.
        // Move to a proper xUnit/NUnit project when the simulation layer is added.

        var portfolioTests = new PortfolioStateTests();
        portfolioTests.Test_HarvestLoss_DecreasesGYTD();
        portfolioTests.Test_WashSaleClock_StartsAtZeroAfterHarvest();
        portfolioTests.Test_OracleBlocked_WhenGYTD_IsNegative();
        portfolioTests.Test_SeedGYTD_EnablesOracleGate();
        portfolioTests.Test_SeedGYTD_ReSeeds_AfterYearEndReset();

        var ledgerTests = new TaxLedgerTests();
        ledgerTests.Test_LedgerNet_MatchesLegacyGYTD();
        ledgerTests.Test_RollYearEnd_BanksExcessLoss();
        ledgerTests.Test_OffsetBudget_And_Capacity_DrawDown();
        ledgerTests.Test_ComputeTaxValue_CapacitySplit_And_Rates();
        ledgerTests.Test_PortfolioState_RoutesThroughLedger();

        var oracleTests = new OracleBoundaryTests();
        oracleTests.Test_Oracle_FiresWhenAllConditionsMet();
        oracleTests.Test_Oracle_Blocked_WhenLossInsufficient();
        oracleTests.Test_Oracle_Blocked_WhenTEOverBudget();
        oracleTests.Test_Oracle_Blocked_WhenGYTD_Zero();
        oracleTests.Test_Oracle_Blocked_WhenWashSaleActive();
        oracleTests.Test_Oracle_Fires_AtWashSaleBoundary();

        var scalarizedTests = new OracleScalarizedTests();
        scalarizedTests.Test_Fires_WithoutRealizedGains();
        scalarizedTests.Test_Blocked_WhenUtilityNegative();
        scalarizedTests.Test_TradeOff_TaxValueVsTrackingError();
        scalarizedTests.Test_HardCeiling_BindsInPathologicalRegimes();
        scalarizedTests.Test_LossAndWashGates_StillBind();
        scalarizedTests.Test_GatedMode_MatchesLegacyOverload();
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

        // ── ML.NET layer tests ──────────────────────────────────────────────
        new LotStateVectorCsvReaderTests().Test_RoundTrip_PreservesAllFields();
        new StratifiedSplitTests().Test_PreservesClassProportionWithin1Percent();
        new StratifiedKFoldTests().Test_FoldsPartitionDataAndContainPositives();
        new SilhouetteTests().Test_TwoBlobsHighSilhouette();
        var preprocessing = new PreprocessingTests();
        preprocessing.Test_MedianImputerReplacesNaNs();
        preprocessing.Test_ClassWeightsBalanced();
        new GridSearchTests().Test_PicksLargestCForLinearlySeparableData();

        // ── Model-suite tests ───────────────────────────────────────────────
        new GbtTrainerTests().Test_GbtCv_SeparableData();
        new RfTrainerTests().Test_RfCv_SeparableData();
        new ElasticNetTrainerTests().Test_ElnetCv_BothPenaltiesSearched();
        var linRegTests = new LinRegTrainerTests();
        linRegTests.Test_LinRegCv_ProducesLowerPrAucThanGbt();
        linRegTests.Test_LinRegRun_FractionOutsideUnit();
        new ChampionSelectionTests().Test_GbtBeatsLinregOnSeparableData();

        Console.WriteLine("All tests passed.");
    }
    break;
}
