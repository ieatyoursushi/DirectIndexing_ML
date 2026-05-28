using DirectIndexing.Core.Simulation;
using DirectIndexing.DataCollection;
using DirectIndexing.Export;
using DirectIndexing.ML;
using DirectIndexing.ML.MLNet;
using DirectIndexing.ML.MLNet.Data;

var mode = args.FirstOrDefault() ?? "simulate";
switch (mode)
{
    case "download":
    {
        var apiKey = Environment.GetEnvironmentVariable("FMP_API_KEY")
                     ?? throw new InvalidOperationException(
                         "FMP_API_KEY environment variable is not set. " +
                         "Export it before running: export FMP_API_KEY=your_key_here");

        await new MarketDataDownloader(apiKey).DownloadAllHistoricalData("data/raw", years: 2); 
    }
    break;
    case "simulate":
    {
        var loader = new PriceLoader();
        loader.Load("data/raw", "data/constituents.json");

        var engine    = new SimulationEngine(loader);
        var snapshots = engine.Run(initialPortfolioValue: 10_000_000m);

        var softLabeller = new SoftLabelBuilder(loader);
        softLabeller.Label(snapshots);

        SimulationExporter.WriteCsv(snapshots, "data/lots.csv");
    }
    break;
    // Alternate: Monte Carlo simulation with synthetic GBM prices.
    // Calibrates per-stock σ from the same real data then generates N days of GBM paths.
    // Produces data/lots-mc.csv alongside the backtesting data/lots.csv.
    case "simulate-mc":
    {
        var loader = new PriceLoader();
        loader.Load("data/raw", "data/constituents.json");

        var mcEngine  = new MonteCarloEngine(loader, annualDrift: 0f);
        var snapshots = mcEngine.Run(
            initialPortfolioValue: 10_000_000m,
            simDays:    504,
            warmupDays: 200,
            seed:       42);

        SimulationExporter.WriteCsv(snapshots, "data/lots-mc.csv");
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
        var data = LotStateVectorCsvReader.Read("data/lots.csv");
        MLnetPipeline.RunUnsupervised(data, "data/artifacts-mlnet/");
    }
    break;
    case "mlnet-supervised":   // PRIMARY: Y_Soft_BT
    {
        var data = LotStateVectorCsvReader.Read("data/lots.csv");
        MLnetPipeline.RunSupervised(data, target: "soft_bt", artifactsDir: "data/artifacts-mlnet/");
    }
    break;
    case "mlnet-baseline":     // SANITY: Y_Oracle
    {
        var data = LotStateVectorCsvReader.Read("data/lots.csv");
        MLnetPipeline.RunSupervised(data, target: "oracle", artifactsDir: "data/artifacts-mlnet/");
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
        var data = LotStateVectorCsvReader.Read("data/lots.csv");
        MLnetPipeline.RunUnsupervised(data, "data/artifacts-mlnet/");
        MLnetPipeline.RunSupervised(data, target: "soft_bt", artifactsDir: "data/artifacts-mlnet/");
        MLnetPipeline.RunSupervised(data, target: "oracle",  artifactsDir: "data/artifacts-mlnet/");
        var rc = MLnetPipeline.RunRender(
            lotsCsv:      "../../../data/lots.csv",
            artifactsDir: "../../../data/artifacts-mlnet/",
            edaDir:       "../../Export/eda-mlnet/",
            modelsDir:    "../../Export/models-mlnet/");
        Environment.ExitCode = rc;
    }
    break;

    //case "results" will likely be in python to generate the ipynb performance report
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

        var oracleTests = new OracleBoundaryTests();
        oracleTests.Test_Oracle_FiresWhenAllConditionsMet();
        oracleTests.Test_Oracle_Blocked_WhenLossInsufficient();
        oracleTests.Test_Oracle_Blocked_WhenTEOverBudget();
        oracleTests.Test_Oracle_Blocked_WhenGYTD_Zero();
        oracleTests.Test_Oracle_Blocked_WhenWashSaleActive();
        oracleTests.Test_Oracle_Fires_AtWashSaleBoundary();

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

        Console.WriteLine("All tests passed.");
    }
    break;
}
