# Dependency & Coupling Atlas — .NET layer

> Generated 2026-06-10 by `dotnet run deps` (`scripts/dependencies.py`, brute-force regex scan — no compiler). Scanned **36 .cs files**, **50 types**, **125 reference edges**. Same approach as Zombtoy `DevTools/Diagrams`, adapted for C# records. Approximation note: instance-typed receivers (`loader.Load()`) can't be resolved without a compiler, so §5's call table covers static calls and constructions; §2's *reference* graph (any use of a known type name) is the complete coupling picture.

## 1. Layer graph — namespace level

Arrows read "references"; weights are total type-name mentions.

```mermaid
flowchart LR
    NS0["(entrypoint)<br/>(1 types)"]
    NS1["Core.Oracle<br/>(1 types)"]
    NS2["Core.Portfolio<br/>(3 types)"]
    NS3["Core.Simulation<br/>(7 types)"]
    NS4["DataCollection<br/>(1 types)"]
    NS5["Export<br/>(1 types)"]
    NS6["ML<br/>(1 types)"]
    NS7["ML.MLNet<br/>(4 types)"]
    NS8["ML.MLNet.Data<br/>(1 types)"]
    NS9["ML.MLNet.Io<br/>(1 types)"]
    NS10["ML.MLNet.Metrics<br/>(5 types)"]
    NS11["ML.MLNet.Models<br/>(12 types)"]
    NS12["ML.MLNet.Preprocessing<br/>(8 types)"]
    NS13["ML.MLNet.Schema<br/>(1 types)"]
    NS14["ML.MLNet.Splits<br/>(2 types)"]
    NS15["ML.MLNet.Tuning<br/>(1 types)"]
    NS11 -->|37| NS2
    NS11 -->|32| NS12
    NS3 -->|27| NS2
    NS0 -->|21| NS7
    NS7 -->|19| NS11
    NS11 -->|19| NS13
    NS7 -->|14| NS9
    NS11 -->|14| NS10
    NS14 -->|14| NS2
    NS11 -->|12| NS14
    NS0 -->|10| NS8
    NS11 -->|8| NS15
    NS12 -->|8| NS2
    NS3 -->|7| NS1
    NS15 -->|6| NS12
    NS12 -->|5| NS13
    NS15 -->|5| NS2
    NS0 -->|5| NS3
    NS0 -->|5| NS6
    NS7 -->|4| NS2
    NS8 -->|3| NS2
    NS10 -->|3| NS7
    NS7 -->|2| NS13
    NS7 -->|2| NS6
    NS0 -->|2| NS5
    NS1 -->|1| NS2
    NS5 -->|1| NS2
    NS7 -->|1| NS10
    NS15 -->|1| NS14
    NS15 -->|1| NS10
    NS0 -->|1| NS4
```

## 2. Class dependency graph — type references, clustered by namespace

Edge weight = number of times the source type's body mentions the target type.

```mermaid
flowchart LR
    subgraph NS0["(entrypoint)"]
        Program["Program<br/><i>entrypoint</i>"]
    end
    subgraph NS1["Core.Oracle"]
        OracleBoundary["OracleBoundary"]
    end
    subgraph NS2["Core.Portfolio"]
        Lot["Lot"]
        LotStateVector["LotStateVector<br/><i>record</i>"]
        PortfolioState["PortfolioState"]
    end
    subgraph NS3["Core.Simulation"]
        GbmSimulator["GbmSimulator"]
        MonteCarloEngine["MonteCarloEngine"]
        PriceLoader["PriceLoader"]
        SigmaTeBuffer["SigmaTeBuffer"]
        SimulationEngine["SimulationEngine"]
        SoftLabelBuilder["SoftLabelBuilder"]
        TrackingErrorProxy["TrackingErrorProxy"]
    end
    subgraph NS4["DataCollection"]
        MarketDataDownloader["MarketDataDownloader"]
    end
    subgraph NS5["Export"]
        SimulationExporter["SimulationExporter"]
    end
    subgraph NS6["ML"]
        PythonRunner["PythonRunner"]
    end
    subgraph NS7["ML.MLNet"]
        BaseMetrics["BaseMetrics<br/><i>record</i>"]
        Confusion["Confusion<br/><i>record</i>"]
        CurvePointDto["CurvePointDto<br/><i>record</i>"]
        MLnetPipeline["MLnetPipeline"]
    end
    subgraph NS8["ML.MLNet.Data"]
        LotStateVectorCsvReader["LotStateVectorCsvReader"]
    end
    subgraph NS9["ML.MLNet.Io"]
        Artifacts["Artifacts"]
    end
    subgraph NS10["ML.MLNet.Metrics"]
        BinaryMetrics["BinaryMetrics"]
        BinaryMetricsResult["BinaryMetricsResult"]
        CurvePoint["CurvePoint<br/><i>record</i>"]
        ScoredRow["ScoredRow"]
        SilhouetteScore["SilhouetteScore"]
    end
    subgraph NS11["ML.MLNet.Models"]
        ClusterPrediction["ClusterPrediction"]
        ClusterRow["ClusterRow"]
        ElasticNetTrainer["ElasticNetTrainer"]
        GradientBoostedTreesTrainer["GradientBoostedTreesTrainer"]
        KMeansPipeline["KMeansPipeline"]
        LinearRegressionTrainer["LinearRegressionTrainer"]
        LogisticTrainer["LogisticTrainer"]
        PcaOutput["PcaOutput<br/><i>record</i>"]
        PcaPipeline["PcaPipeline"]
        RandomForestTrainer["RandomForestTrainer"]
        RegressionScoredRow["RegressionScoredRow"]
        SymbolAggregate["SymbolAggregate<br/><i>record</i>"]
    end
    subgraph NS12["ML.MLNet.Preprocessing"]
        ClassWeights["ClassWeights"]
        MLReadyRow["MLReadyRow<br/><i>record</i>"]
        MedianImputer["MedianImputer"]
        PreprocessingPipeline["PreprocessingPipeline"]
        SectorCleanFactory["SectorCleanFactory"]
        SectorIn["SectorIn"]
        SectorOut["SectorOut"]
        WeightedRow["WeightedRow<br/><i>record</i>"]
    end
    subgraph NS13["ML.MLNet.Schema"]
        FeatureLists["FeatureLists"]
    end
    subgraph NS14["ML.MLNet.Splits"]
        StratifiedKFold["StratifiedKFold"]
        StratifiedSplit["StratifiedSplit"]
    end
    subgraph NS15["ML.MLNet.Tuning"]
        GridSearchCV["GridSearchCV"]
    end
    BaseMetrics -->|14| Artifacts
    BaseMetrics -->|4| CurvePointDto
    BaseMetrics -->|4| LotStateVector
    BaseMetrics -->|4| LogisticTrainer
    BaseMetrics -->|3| GradientBoostedTreesTrainer
    BaseMetrics -->|3| RandomForestTrainer
    BaseMetrics -->|3| ElasticNetTrainer
    BaseMetrics -->|3| LinearRegressionTrainer
    BaseMetrics -->|2| Confusion
    BaseMetrics -->|2| FeatureLists
    BaseMetrics -->|2| KMeansPipeline
    BaseMetrics -->|2| PythonRunner
    BaseMetrics -->|1| BinaryMetricsResult
    BaseMetrics -->|1| PcaPipeline
    BinaryMetrics -->|5| CurvePoint
    BinaryMetrics -->|3| BinaryMetricsResult
    BinaryMetrics -->|3| ScoredRow
    BinaryMetrics -->|3| Confusion
    BinaryMetricsResult -->|4| CurvePoint
    ClassWeights -->|3| WeightedRow
    ClassWeights -->|2| LotStateVector
    ElasticNetTrainer -->|6| LotStateVector
    ElasticNetTrainer -->|4| FeatureLists
    ElasticNetTrainer -->|3| MedianImputer
    ElasticNetTrainer -->|2| StratifiedSplit
    ElasticNetTrainer -->|2| GridSearchCV
    ElasticNetTrainer -->|1| BinaryMetricsResult
    ElasticNetTrainer -->|1| ClassWeights
    ElasticNetTrainer -->|1| BinaryMetrics
    ElasticNetTrainer -->|1| PreprocessingPipeline
    GradientBoostedTreesTrainer -->|6| LotStateVector
    GradientBoostedTreesTrainer -->|3| MedianImputer
    GradientBoostedTreesTrainer -->|3| FeatureLists
    GradientBoostedTreesTrainer -->|2| StratifiedSplit
    GradientBoostedTreesTrainer -->|2| GridSearchCV
    GradientBoostedTreesTrainer -->|1| BinaryMetricsResult
    GradientBoostedTreesTrainer -->|1| ClassWeights
    GradientBoostedTreesTrainer -->|1| BinaryMetrics
    GradientBoostedTreesTrainer -->|1| PreprocessingPipeline
    GridSearchCV -->|5| LotStateVector
    GridSearchCV -->|3| MedianImputer
    GridSearchCV -->|1| StratifiedKFold
    GridSearchCV -->|1| ClassWeights
    GridSearchCV -->|1| BinaryMetrics
    GridSearchCV -->|1| MLReadyRow
    GridSearchCV -->|1| WeightedRow
    LogisticTrainer -->|6| LotStateVector
    LogisticTrainer -->|4| FeatureLists
    LogisticTrainer -->|3| MedianImputer
    LogisticTrainer -->|2| StratifiedSplit
    LogisticTrainer -->|2| GridSearchCV
    LogisticTrainer -->|1| BinaryMetricsResult
    LogisticTrainer -->|1| ClassWeights
    LogisticTrainer -->|1| BinaryMetrics
    LogisticTrainer -->|1| PreprocessingPipeline
    LotStateVectorCsvReader -->|3| LotStateVector
    MedianImputer -->|4| LotStateVector
    MedianImputer -->|3| MLReadyRow
    MonteCarloEngine -->|5| LotStateVector
    MonteCarloEngine -->|5| Lot
    MonteCarloEngine -->|4| PortfolioState
    MonteCarloEngine -->|3| OracleBoundary
    MonteCarloEngine -->|2| GbmSimulator
    MonteCarloEngine -->|2| SigmaTeBuffer
    MonteCarloEngine -->|1| PriceLoader
    MonteCarloEngine -->|1| TrackingErrorProxy
    OracleBoundary -->|1| LotStateVector
    PcaOutput -->|2| LotStateVector
    PcaOutput -->|2| MedianImputer
    PcaOutput -->|2| FeatureLists
    PcaOutput -->|1| StratifiedSplit
    PcaOutput -->|1| MLReadyRow
    PortfolioState -->|3| Lot
    PreprocessingPipeline -->|1| SectorCleanFactory
    Program -->|21| MLnetPipeline
    Program -->|10| LotStateVectorCsvReader
    Program -->|5| PythonRunner
    Program -->|2| PriceLoader
    Program -->|2| SimulationExporter
    Program -->|1| MarketDataDownloader
    Program -->|1| SimulationEngine
    Program -->|1| SoftLabelBuilder
    Program -->|1| MonteCarloEngine
    RandomForestTrainer -->|6| LotStateVector
    RandomForestTrainer -->|3| MedianImputer
    RandomForestTrainer -->|3| FeatureLists
    RandomForestTrainer -->|2| StratifiedSplit
    RandomForestTrainer -->|2| GridSearchCV
    RandomForestTrainer -->|1| BinaryMetricsResult
    RandomForestTrainer -->|1| ClassWeights
    RandomForestTrainer -->|1| BinaryMetrics
    RandomForestTrainer -->|1| PreprocessingPipeline
    RegressionScoredRow -->|9| LotStateVector
    RegressionScoredRow -->|6| MedianImputer
    RegressionScoredRow -->|3| FeatureLists
    RegressionScoredRow -->|2| StratifiedSplit
    RegressionScoredRow -->|2| ClassWeights
    RegressionScoredRow -->|2| ScoredRow
    RegressionScoredRow -->|2| BinaryMetrics
    RegressionScoredRow -->|1| BinaryMetricsResult
    RegressionScoredRow -->|1| StratifiedKFold
    RegressionScoredRow -->|1| PreprocessingPipeline
    SectorCleanFactory -->|2| SectorIn
    SectorCleanFactory -->|2| SectorOut
    SectorOut -->|5| FeatureLists
    SectorOut -->|3| SectorCleanFactory
    SimulationEngine -->|5| Lot
    SimulationEngine -->|4| PriceLoader
    SimulationEngine -->|4| LotStateVector
    SimulationEngine -->|2| TrackingErrorProxy
    SimulationEngine -->|2| OracleBoundary
    SimulationEngine -->|1| PortfolioState
    SimulationExporter -->|1| LotStateVector
    SoftLabelBuilder -->|3| LotStateVector
    SoftLabelBuilder -->|2| PriceLoader
    SoftLabelBuilder -->|2| OracleBoundary
    SoftLabelBuilder -->|1| GbmSimulator
    StratifiedKFold -->|6| LotStateVector
    StratifiedSplit -->|8| LotStateVector
    SymbolAggregate -->|2| LotStateVector
    SymbolAggregate -->|1| ClusterRow
    SymbolAggregate -->|1| ClusterPrediction
    SymbolAggregate -->|1| SilhouetteScore
    TrackingErrorProxy -->|2| PriceLoader
    WeightedRow -->|2| LotStateVector
```

## 3. Inheritance & interface implementation

```mermaid
classDiagram
    class CustomMappingFactory
    class LotStateVector { <<record>> }
    class SectorCleanFactory
    class WeightedRow { <<record>> }
    CustomMappingFactory <|-- SectorCleanFactory
    LotStateVector <|-- WeightedRow
```

(`WeightedRow : LotStateVector` is the load-bearing one — the per-row training weight rides on the same immutable schema the simulator wrote.)

## 4. Coupling metrics

Fan-out = types this type references (breadth) / total mentions (weight). Fan-in = types that reference it. High fan-in = load-bearing schema; high fan-out = orchestrator.

| Type | Kind | Namespace | Fan-out (types / refs) | Fan-in (types / refs) |
|---|---|---|---|---|
| `LotStateVector` | record | Core.Portfolio | 0 / 0 | 20 / 85 |
| `BaseMetrics` | record | ML.MLNet | 14 / 48 | 0 / 0 |
| `GridSearchCV` | class | ML.MLNet.Tuning | 7 / 13 | 4 / 8 |
| `BinaryMetrics` | class | ML.MLNet.Metrics | 4 / 14 | 6 / 7 |
| `ElasticNetTrainer` | class | ML.MLNet.Models | 9 / 21 | 1 / 3 |
| `GradientBoostedTreesTrainer` | class | ML.MLNet.Models | 9 / 20 | 1 / 3 |
| `RegressionScoredRow` | class | ML.MLNet.Models | 10 / 29 | 0 / 0 |
| `LogisticTrainer` | class | ML.MLNet.Models | 9 / 21 | 1 / 4 |
| `RandomForestTrainer` | class | ML.MLNet.Models | 9 / 20 | 1 / 3 |
| `MonteCarloEngine` | class | Core.Simulation | 8 / 23 | 1 / 1 |
| `MedianImputer` | class | ML.MLNet.Preprocessing | 2 / 7 | 7 / 23 |
| `Program` | entrypoint | (entrypoint) | 9 / 44 | 0 / 0 |
| `BinaryMetricsResult` | class | ML.MLNet.Metrics | 1 / 4 | 7 / 9 |
| `ClassWeights` | class | ML.MLNet.Preprocessing | 2 / 5 | 6 / 7 |
| `FeatureLists` | class | ML.MLNet.Schema | 0 / 0 | 8 / 26 |
| `SimulationEngine` | class | Core.Simulation | 6 / 18 | 1 / 1 |
| `StratifiedSplit` | class | ML.MLNet.Splits | 1 / 8 | 6 / 11 |
| `PreprocessingPipeline` | class | ML.MLNet.Preprocessing | 1 / 1 | 5 / 5 |
| `PriceLoader` | class | Core.Simulation | 0 / 0 | 5 / 11 |
| `SoftLabelBuilder` | class | Core.Simulation | 4 / 8 | 1 / 1 |
| `PcaOutput` | record | ML.MLNet.Models | 5 / 8 | 0 / 0 |
| `OracleBoundary` | class | Core.Oracle | 1 / 1 | 3 / 7 |
| `SymbolAggregate` | record | ML.MLNet.Models | 4 / 5 | 0 / 0 |
| `SectorCleanFactory` | class | ML.MLNet.Preprocessing | 2 / 4 | 2 / 4 |
| `Lot` | class | Core.Portfolio | 0 / 0 | 3 / 13 |
| `PortfolioState` | class | Core.Portfolio | 1 / 3 | 2 / 5 |
| `TrackingErrorProxy` | class | Core.Simulation | 1 / 2 | 2 / 3 |
| `WeightedRow` | record | ML.MLNet.Preprocessing | 1 / 2 | 2 / 4 |
| `MLReadyRow` | record | ML.MLNet.Preprocessing | 0 / 0 | 3 / 5 |
| `SectorOut` | class | ML.MLNet.Preprocessing | 2 / 8 | 1 / 2 |
| `StratifiedKFold` | class | ML.MLNet.Splits | 1 / 6 | 2 / 2 |
| `GbmSimulator` | class | Core.Simulation | 0 / 0 | 2 / 3 |
| `SimulationExporter` | class | Export | 1 / 1 | 1 / 2 |
| `LotStateVectorCsvReader` | class | ML.MLNet.Data | 1 / 3 | 1 / 10 |
| `Confusion` | record | ML.MLNet | 0 / 0 | 2 / 5 |
| `ScoredRow` | class | ML.MLNet.Metrics | 0 / 0 | 2 / 5 |
| `CurvePoint` | record | ML.MLNet.Metrics | 0 / 0 | 2 / 9 |
| `PythonRunner` | class | ML | 0 / 0 | 2 / 7 |
| `SigmaTeBuffer` | class | Core.Simulation | 0 / 0 | 1 / 2 |
| `MarketDataDownloader` | class | DataCollection | 0 / 0 | 1 / 1 |
| `Artifacts` | class | ML.MLNet.Io | 0 / 0 | 1 / 14 |
| `MLnetPipeline` | class | ML.MLNet | 0 / 0 | 1 / 21 |
| `CurvePointDto` | record | ML.MLNet | 0 / 0 | 1 / 4 |
| `SilhouetteScore` | class | ML.MLNet.Metrics | 0 / 0 | 1 / 1 |
| `KMeansPipeline` | class | ML.MLNet.Models | 0 / 0 | 1 / 2 |
| `ClusterRow` | class | ML.MLNet.Models | 0 / 0 | 1 / 1 |
| `ClusterPrediction` | class | ML.MLNet.Models | 0 / 0 | 1 / 1 |
| `LinearRegressionTrainer` | class | ML.MLNet.Models | 0 / 0 | 1 / 3 |
| `PcaPipeline` | class | ML.MLNet.Models | 0 / 0 | 1 / 1 |
| `SectorIn` | class | ML.MLNet.Preprocessing | 0 / 0 | 1 / 2 |

## 5. Cross-class call detail

Statically resolvable call sites: `Receiver.Method(...)` where the receiver is a known project type, plus `new Type(...)` constructions (`.ctor`).

| Caller | Callee | Members used |
|---|---|---|
| `BaseMetrics` | `Artifacts` | `WriteCsv`, `WriteJson` |
| `BaseMetrics` | `CurvePointDto` | `.ctor` |
| `BaseMetrics` | `ElasticNetTrainer` | `Run`, `RunCV` |
| `BaseMetrics` | `GradientBoostedTreesTrainer` | `Run`, `RunCV` |
| `BaseMetrics` | `KMeansPipeline` | `Run` |
| `BaseMetrics` | `LinearRegressionTrainer` | `Run`, `RunCV` |
| `BaseMetrics` | `LogisticTrainer` | `Run`, `RunCV` |
| `BaseMetrics` | `PcaPipeline` | `Run` |
| `BaseMetrics` | `PythonRunner` | `Run` |
| `BaseMetrics` | `RandomForestTrainer` | `Run`, `RunCV` |
| `BinaryMetrics` | `BinaryMetricsResult` | `.ctor` |
| `BinaryMetrics` | `CurvePoint` | `.ctor` |
| `ClassWeights` | `WeightedRow` | `From` |
| `ElasticNetTrainer` | `BinaryMetrics` | `Compute` |
| `ElasticNetTrainer` | `ClassWeights` | `AttachBalancedWeights` |
| `ElasticNetTrainer` | `GridSearchCV` | `Search` |
| `ElasticNetTrainer` | `MedianImputer` | `Apply`, `Fit` |
| `ElasticNetTrainer` | `PreprocessingPipeline` | `Build` |
| `ElasticNetTrainer` | `StratifiedSplit` | `Split` |
| `GradientBoostedTreesTrainer` | `BinaryMetrics` | `Compute` |
| `GradientBoostedTreesTrainer` | `ClassWeights` | `AttachBalancedWeights` |
| `GradientBoostedTreesTrainer` | `GridSearchCV` | `Search` |
| `GradientBoostedTreesTrainer` | `MedianImputer` | `Apply`, `Fit` |
| `GradientBoostedTreesTrainer` | `PreprocessingPipeline` | `Build` |
| `GradientBoostedTreesTrainer` | `StratifiedSplit` | `Split` |
| `GridSearchCV` | `BinaryMetrics` | `Compute` |
| `GridSearchCV` | `ClassWeights` | `AttachBalancedWeights` |
| `GridSearchCV` | `MedianImputer` | `Apply`, `Fit` |
| `GridSearchCV` | `StratifiedKFold` | `Folds` |
| `LogisticTrainer` | `BinaryMetrics` | `Compute` |
| `LogisticTrainer` | `ClassWeights` | `AttachBalancedWeights` |
| `LogisticTrainer` | `GridSearchCV` | `Search` |
| `LogisticTrainer` | `MedianImputer` | `Apply`, `Fit` |
| `LogisticTrainer` | `PreprocessingPipeline` | `Build` |
| `LogisticTrainer` | `StratifiedSplit` | `Split` |
| `LotStateVectorCsvReader` | `LotStateVector` | `.ctor` |
| `MedianImputer` | `MLReadyRow` | `.ctor` |
| `MonteCarloEngine` | `GbmSimulator` | `NextGaussian` |
| `MonteCarloEngine` | `Lot` | `.ctor` |
| `MonteCarloEngine` | `LotStateVector` | `.ctor` |
| `MonteCarloEngine` | `OracleBoundary` | `Label` |
| `MonteCarloEngine` | `PortfolioState` | `.ctor` |
| `MonteCarloEngine` | `SigmaTeBuffer` | `.ctor` |
| `MonteCarloEngine` | `TrackingErrorProxy` | `ComputeCovariance` |
| `PcaOutput` | `MedianImputer` | `Apply`, `Fit` |
| `PcaOutput` | `StratifiedSplit` | `Split` |
| `Program` | `LotStateVectorCsvReader` | `Read` |
| `Program` | `MLnetPipeline` | `RunAllSupervised`, `RunRender`, `RunSupervised`, `RunSupervisedModel`, `RunUnsupervised` |
| `Program` | `MarketDataDownloader` | `.ctor` |
| `Program` | `MonteCarloEngine` | `.ctor` |
| `Program` | `PriceLoader` | `.ctor` |
| `Program` | `PythonRunner` | `Run` |
| `Program` | `SimulationEngine` | `.ctor` |
| `Program` | `SimulationExporter` | `WriteCsv` |
| `Program` | `SoftLabelBuilder` | `.ctor` |
| `RandomForestTrainer` | `BinaryMetrics` | `Compute` |
| `RandomForestTrainer` | `ClassWeights` | `AttachBalancedWeights` |
| `RandomForestTrainer` | `GridSearchCV` | `Search` |
| `RandomForestTrainer` | `MedianImputer` | `Apply`, `Fit` |
| `RandomForestTrainer` | `PreprocessingPipeline` | `Build` |
| `RandomForestTrainer` | `StratifiedSplit` | `Split` |
| `RegressionScoredRow` | `BinaryMetrics` | `Compute` |
| `RegressionScoredRow` | `ClassWeights` | `AttachBalancedWeights` |
| `RegressionScoredRow` | `MedianImputer` | `Apply`, `Fit` |
| `RegressionScoredRow` | `PreprocessingPipeline` | `Build` |
| `RegressionScoredRow` | `ScoredRow` | `.ctor` |
| `RegressionScoredRow` | `StratifiedKFold` | `Folds` |
| `RegressionScoredRow` | `StratifiedSplit` | `Split` |
| `SectorOut` | `SectorCleanFactory` | `.ctor` |
| `SimulationEngine` | `Lot` | `.ctor` |
| `SimulationEngine` | `LotStateVector` | `.ctor` |
| `SimulationEngine` | `OracleBoundary` | `Label` |
| `SimulationEngine` | `TrackingErrorProxy` | `.ctor` |
| `SoftLabelBuilder` | `OracleBoundary` | `Label` |
| `SymbolAggregate` | `ClusterRow` | `.ctor` |
| `SymbolAggregate` | `SilhouetteScore` | `Compute` |

## 6. File inventory

| Namespace | Type | Kind | File |
|---|---|---|---|
| (entrypoint) | `Program` | entrypoint | `Program.cs` |
| Core.Oracle | `OracleBoundary` | class | `Core/Oracle/OracleBoundary.cs` |
| Core.Portfolio | `Lot` | class | `Core/Portfolio/Lot.cs` |
| Core.Portfolio | `LotStateVector` | record | `Core/Portfolio/LotStateVector.cs` |
| Core.Portfolio | `PortfolioState` | class | `Core/Portfolio/PortfolioState.cs` |
| Core.Simulation | `GbmSimulator` | class | `Core/Simulation/GbmSimulator.cs` |
| Core.Simulation | `MonteCarloEngine` | class | `Core/Simulation/MonteCarloEngine.cs` |
| Core.Simulation | `PriceLoader` | class | `Core/Simulation/PriceLoader.cs` |
| Core.Simulation | `SigmaTeBuffer` | class | `Core/Simulation/MonteCarloEngine.cs` |
| Core.Simulation | `SimulationEngine` | class | `Core/Simulation/SimulationEngine.cs` |
| Core.Simulation | `SoftLabelBuilder` | class | `Core/Simulation/SoftLabelBuilder.cs` |
| Core.Simulation | `TrackingErrorProxy` | class | `Core/Simulation/TrackingErrorProxy.cs` |
| DataCollection | `MarketDataDownloader` | class | `DataCollection/MarketDataDownloader.cs` |
| Export | `SimulationExporter` | class | `Export/SimulationExporter.cs` |
| ML | `PythonRunner` | class | `ML/CSharp/PythonRunner.cs` |
| ML.MLNet | `BaseMetrics` | record | `ML/CSharp/MLNet/MLnetPipeline.cs` |
| ML.MLNet | `Confusion` | record | `ML/CSharp/MLNet/MLnetPipeline.cs` |
| ML.MLNet | `CurvePointDto` | record | `ML/CSharp/MLNet/MLnetPipeline.cs` |
| ML.MLNet | `MLnetPipeline` | class | `ML/CSharp/MLNet/MLnetPipeline.cs` |
| ML.MLNet.Data | `LotStateVectorCsvReader` | class | `ML/CSharp/MLNet/Data/LotStateVectorCsvReader.cs` |
| ML.MLNet.Io | `Artifacts` | class | `ML/CSharp/MLNet/Io/Artifacts.cs` |
| ML.MLNet.Metrics | `BinaryMetrics` | class | `ML/CSharp/MLNet/Metrics/BinaryMetrics.cs` |
| ML.MLNet.Metrics | `BinaryMetricsResult` | class | `ML/CSharp/MLNet/Metrics/BinaryMetrics.cs` |
| ML.MLNet.Metrics | `CurvePoint` | record | `ML/CSharp/MLNet/Metrics/BinaryMetrics.cs` |
| ML.MLNet.Metrics | `ScoredRow` | class | `ML/CSharp/MLNet/Metrics/BinaryMetrics.cs` |
| ML.MLNet.Metrics | `SilhouetteScore` | class | `ML/CSharp/MLNet/Metrics/SilhouetteScore.cs` |
| ML.MLNet.Models | `ClusterPrediction` | class | `ML/CSharp/MLNet/Models/KMeansPipeline.cs` |
| ML.MLNet.Models | `ClusterRow` | class | `ML/CSharp/MLNet/Models/KMeansPipeline.cs` |
| ML.MLNet.Models | `ElasticNetTrainer` | class | `ML/CSharp/MLNet/Models/ElasticNetTrainer.cs` |
| ML.MLNet.Models | `GradientBoostedTreesTrainer` | class | `ML/CSharp/MLNet/Models/GradientBoostedTreesTrainer.cs` |
| ML.MLNet.Models | `KMeansPipeline` | class | `ML/CSharp/MLNet/Models/KMeansPipeline.cs` |
| ML.MLNet.Models | `LinearRegressionTrainer` | class | `ML/CSharp/MLNet/Models/LinearRegressionTrainer.cs` |
| ML.MLNet.Models | `LogisticTrainer` | class | `ML/CSharp/MLNet/Models/LogisticTrainer.cs` |
| ML.MLNet.Models | `PcaOutput` | record | `ML/CSharp/MLNet/Models/PcaPipeline.cs` |
| ML.MLNet.Models | `PcaPipeline` | class | `ML/CSharp/MLNet/Models/PcaPipeline.cs` |
| ML.MLNet.Models | `RandomForestTrainer` | class | `ML/CSharp/MLNet/Models/RandomForestTrainer.cs` |
| ML.MLNet.Models | `RegressionScoredRow` | class | `ML/CSharp/MLNet/Models/LinearRegressionTrainer.cs` |
| ML.MLNet.Models | `SymbolAggregate` | record | `ML/CSharp/MLNet/Models/KMeansPipeline.cs` |
| ML.MLNet.Preprocessing | `ClassWeights` | class | `ML/CSharp/MLNet/Preprocessing/ClassWeights.cs` |
| ML.MLNet.Preprocessing | `MLReadyRow` | record | `ML/CSharp/MLNet/Preprocessing/MLReadyRow.cs` |
| ML.MLNet.Preprocessing | `MedianImputer` | class | `ML/CSharp/MLNet/Preprocessing/MedianImputer.cs` |
| ML.MLNet.Preprocessing | `PreprocessingPipeline` | class | `ML/CSharp/MLNet/Preprocessing/PreprocessingPipeline.cs` |
| ML.MLNet.Preprocessing | `SectorCleanFactory` | class | `ML/CSharp/MLNet/Preprocessing/PreprocessingPipeline.cs` |
| ML.MLNet.Preprocessing | `SectorIn` | class | `ML/CSharp/MLNet/Preprocessing/PreprocessingPipeline.cs` |
| ML.MLNet.Preprocessing | `SectorOut` | class | `ML/CSharp/MLNet/Preprocessing/PreprocessingPipeline.cs` |
| ML.MLNet.Preprocessing | `WeightedRow` | record | `ML/CSharp/MLNet/Preprocessing/ClassWeights.cs` |
| ML.MLNet.Schema | `FeatureLists` | class | `ML/CSharp/MLNet/Schema/FeatureLists.cs` |
| ML.MLNet.Splits | `StratifiedKFold` | class | `ML/CSharp/MLNet/Splits/StratifiedKFold.cs` |
| ML.MLNet.Splits | `StratifiedSplit` | class | `ML/CSharp/MLNet/Splits/StratifiedSplit.cs` |
| ML.MLNet.Tuning | `GridSearchCV` | class | `ML/CSharp/MLNet/Tuning/GridSearchCV.cs` |
