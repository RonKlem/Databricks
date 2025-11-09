"""
Databricks demo script: Taxi Dataset Showcase
- Purpose: A single-file Python script that can be loaded into Databricks and run cell-by-cell.
- Usage: Paste into a Databricks Python notebook cell or upload as a file and run.
- NOTE: Some libraries (graphframes, xgboost, torch) may not be installed on your cluster. The script
  will try to import them and will show helpful messages if they are missing.

Sections (mapped from the original notebook):
1. Introduction (display HTML)
2. Load Taxi Data (Parquet)
3. Spark Transformations
4. Delta Lake demo (write/read/update)
5. MLlib Logistic Regression (baseline)
6. GraphFrames demo (optional)
7. Graph features -> ML pipeline
8. MLflow experiment tracking
9. XGBoost & PyTorch examples (small sample to avoid OOM)
10. MLflow visualization notes
11. Collaboration & Dashboards notes
12. Wrap-up

Set RUN_SECTIONS to control which sections to execute. Default runs everything and handles missing dependencies gracefully.
"""

# ---------------------
# Configuration
# ---------------------
USER = "RonKlem"  # set to your Databricks username (e.g. "RonKlem") or replace later
TAXI_PARQUET_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
DELTA_PATH = "/tmp/taxi_delta_demo"
SAMPLE_FOR_XGB_AND_TORCH = 5000  # reduce to avoid memory issues when converting to pandas
RUN_SECTIONS = {
    "intro": True,
    "load": True,
    "transform": True,
    "delta": True,
    "mllib": True,
    "graphframes": True,
    "graph_features": True,
    "mlflow": True,
    "xgboost_pytorch": True,
    "notes": True,
}

# ---------------------
# Helpers
# ---------------------
def safe_import(module_name, pip_install_cmd=None):
    try:
        mod = __import__(module_name)
        return mod, None
    except Exception as e:
        msg = f"Module '{module_name}' not available: {e}"
        if pip_install_cmd:
            msg += f"\nYou can install it on your cluster (init script or library) with: {pip_install_cmd}"
        return None, msg

def choose_column(df, candidates):
    """Return the first column name present in df.columns from the candidates list, else None."""
    cols = set([c.lower() for c in df.columns])
    for c in candidates:
        if c and c.lower() in cols:
            # return the actual column name with original case
            for orig in df.columns:
                if orig.lower() == c.lower():
                    return orig
    return None

# Databricks display helpers
def display_markdown(md: str):
    try:
        # displayHTML renders HTML; wrap markdown in <pre> if you want plain text
        displayHTML(f"<div style='font-family:Helvetica, Arial;line-height:1.4'>{md}</div>")
    except Exception:
        print(md)

# ---------------------
# 1. Introduction
# ---------------------
def section_intro():
    md = """
    <h1>Databricks Demo Notebook</h1>
    <p>Welcome! In this demo we explore:</p>
    <ul>
      <li>Scalable Spark data processing</li>
      <li>Delta Lake reliability</li>
      <li>Machine Learning with MLlib, XGBoost, PyTorch</li>
      <li>Graph analytics with GraphFrames</li>
      <li>MLflow experiment tracking and visualization</li>
      <li>Collaboration and dashboards</li>
    </ul>
    """
    display_markdown(md)

# ---------------------
# 2. Load Taxi Data
# ---------------------
def section_load():
    print("Loading taxi dataset from:", TAXI_PARQUET_URL)
    try:
        df = spark.read.parquet(TAXI_PARQUET_URL)
        print("Schema:")
        df.printSchema()
        print("Sample rows:")
        display(df.limit(5))
        return df
    except Exception as e:
        print("Failed to load parquet from the URL. Exception:", e)
        raise

# ---------------------
# 3. Spark Transformations
# ---------------------
def section_transform(df):
    print("\n-- Spark Transformations: average trip_distance by passenger_count --")
    # Safe check for column names
    trip_distance_col = choose_column(df, ["trip_distance", "tripdistance", "trip_distance_miles"])
    passenger_count_col = choose_column(df, ["passenger_count", "passengercount"])
    if not trip_distance_col or not passenger_count_col:
        print("Required columns not found for aggregation. Available columns:", df.columns)
        return
    df.groupBy(passenger_count_col).agg({trip_distance_col: "avg"}).show(20, truncate=False)

# ---------------------
# 4. Delta Lake Demo
# ---------------------
def section_delta(df):
    print("\n-- Delta Lake Demo: write -> read -> update --")
    try:
        df.write.format("delta").mode("overwrite").save(DELTA_PATH)
        delta_df = spark.read.format("delta").load(DELTA_PATH)
        print("Delta read sample:")
        display(delta_df.limit(5))

        # Update passenger_count == 0 -> 1 (if column exists)
        passenger_col = choose_column(delta_df, ["passenger_count", "passengercount"])
        if passenger_col:
            from delta.tables import DeltaTable
            delta_table = DeltaTable.forPath(spark, DELTA_PATH)
            delta_table.update(
                condition=f"{passenger_col} = 0",
                set={passenger_col: "1"}
            )
            print("Updated passenger_count == 0 to 1 in delta table. Showing sample after update:")
            display(spark.read.format("delta").load(DELTA_PATH).limit(5))
        else:
            print("No passenger_count column detected; skipping update.")
    except Exception as e:
        print("Delta demo failed:", e)
        print("Ensure Delta Lake is available on the cluster.")

# ---------------------
# 5. MLlib Logistic Regression (Baseline)
# ---------------------
def section_mllib(df):
    print("\n-- MLlib Logistic Regression (baseline) --")
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import LogisticRegression

    # Find sensible column names
    trip_distance_col = choose_column(df, ["trip_distance", "tripdistance"])
    passenger_count_col = choose_column(df, ["passenger_count", "passengercount"])
    fare_col = choose_column(df, ["fare_amount", "fare"])
    # payment_type used as label in the original flow; try to find it
    payment_col = choose_column(df, ["payment_type", "paymenttype", "payment"])

    if not (trip_distance_col and passenger_count_col and fare_col and payment_col):
        print("One or more required columns not found for MLlib example. Available columns include:", df.columns)
        print("Skipping MLlib section.")
        return None, None

    # Prepare dataframe (drop rows with nulls in these columns)
    train_df = df.select(trip_distance_col, passenger_count_col, fare_col, payment_col).na.drop()
    # Ensure label is numeric
    train_df = train_df.withColumn(payment_col, train_df[payment_col].cast("integer"))
    assembler = VectorAssembler(inputCols=[trip_distance_col, passenger_count_col, fare_col], outputCol="features")
    train_df = assembler.transform(train_df)
    # Fit logistic regression (this is a demonstration; in many taxi datasets payment_type has >2 classes)
    lr = LogisticRegression(featuresCol="features", labelCol=payment_col)
    try:
        lr_model = lr.fit(train_df)
        print("Logistic Regression model trained. Coefficients:", lr_model.coefficients)
        return train_df, payment_col
    except Exception as e:
        print("Training failed (likely due to label not being binary). Exception:", e)
        return train_df, payment_col

# ---------------------
# 6. GraphFrames Demo (Optional)
# ---------------------
def section_graphframes(df):
    print("\n-- GraphFrames demo: convert taxi trips to graph --")
    gf, err = safe_import("graphframes", pip_install_cmd="graphframes:0.8.1-spark3.0-s_2.12 (cluster library)")
    if gf is None:
        print(err or "graphframes not installed.")
        print("To use GraphFrames, install the library on the cluster and restart the cluster.")
        return None

    # Identify pickup/dropoff and fare columns
    pick_col = choose_column(df, ["pickup_location_id", "PULocationID", "pickup_locationid", "pickup"])
    drop_col = choose_column(df, ["dropoff_location_id", "DOLocationID", "dropoff_locationid", "dropoff"])
    fare_col = choose_column(df, ["fare_amount", "fare"])
    if not (pick_col and drop_col):
        print("Could not find pickup/dropoff columns. Available columns:", df.columns)
        return None

    from pyspark.sql.functions import lit
    # Build vertices (distinct locations) and edges
    vertices = df.selectExpr(f"{pick_col} as id").union(df.selectExpr(f"{drop_col} as id")).distinct().withColumn("type", lit("location"))
    edges = df.selectExpr(f"{pick_col} as src", f"{drop_col} as dst", f"{fare_col} as weight" if fare_col else "1 as weight")
    edges = edges.groupBy("src", "dst").count().withColumnRenamed("count", "trip_count")

    from graphframes import GraphFrame
    g = GraphFrame(vertices, edges)
    try:
        print("Running PageRank (this may take time depending on graph size).")
        pr_graph = g.pageRank(resetProbability=0.15, maxIter=5)  # reduced iterations for demo
        print("PageRank vertices sample:")
        pr_graph.vertices.show(5, truncate=False)
        print("Connected components sample:")
        g.connectedComponents().show(5, truncate=False)
        return g
    except Exception as e:
        print("Graph operations failed:", e)
        return None

# ---------------------
# 7. Graph Features -> ML Pipeline
# ---------------------
def section_graph_features(df, graph):
    print("\n-- Graph features for ML: derive pagerank, degrees, components and join back --")
    if graph is None:
        print("Graph not available; skipping graph features section.")
        return df

    # Derive features
    pr = graph.pageRank(resetProbability=0.15, maxIter=5).vertices.select("id", "pagerank")
    deg = graph.degrees  # columns: id, degree
    components = graph.connectedComponents().select("id", "component")
    graph_features = graph.vertices.select("id").join(pr, "id", "left").join(deg, "id", "left").join(components, "id", "left")

    # Find pickup id column used earlier
    pick_col = choose_column(df, ["pickup_location_id", "PULocationID", "pickup_locationid", "pickup"])
    if not pick_col:
        print("Pickup column not found; cannot join graph features.")
        return df

    df_with_graph = df.join(graph_features, df[pick_col] == graph_features.id, "left")
    # Assemble features including graph features if present
    from pyspark.ml.feature import VectorAssembler
    trip_distance_col = choose_column(df_with_graph, ["trip_distance", "tripdistance"])
    passenger_count_col = choose_column(df_with_graph, ["passenger_count", "passengercount"])
    fare_col = choose_column(df_with_graph, ["fare_amount", "fare"])
    # component may be non-numeric - cast
    if "component" in df_with_graph.columns:
        df_with_graph = df_with_graph.withColumn("component", df_with_graph["component"].cast("double"))
    # ensure numeric columns exist before assembling
    input_cols = [c for c in [trip_distance_col, passenger_count_col, fare_col, "pagerank", "degree", "component"] if c and c in df_with_graph.columns]
    if not input_cols:
        print("No suitable input columns found for graph-enhanced features. Columns available:", df_with_graph.columns)
        return df_with_graph
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    df_with_graph = assembler.transform(df_with_graph)
    print("Prepared df_with_graph with features (sample):")
    display(df_with_graph.select(*([trip_distance_col, passenger_count_col, fare_col] + ["pagerank","degree","component"])).limit(5))
    return df_with_graph

# ---------------------
# 8. MLflow Experiment Tracking
# ---------------------
def section_mlflow(train_baseline_df, train_graph_df, label_col):
    print("\n-- MLflow tracking: RandomForest baseline vs graph-enhanced --")
    mlflow_mod, err = safe_import("mlflow")
    if mlflow_mod is None:
        print(err or "mlflow not installed/configured.")
        return

    import mlflow
    import mlflow.spark
    from pyspark.ml.classification import RandomForestClassifier

    experiment_name = f"/Users/{USER}/TaxiGraphComparison"
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Could not set MLflow experiment to {experiment_name}: {e}")
        # continue; mlflow might create experiment automatically depending on config

    # Baseline Random Forest
    try:
        with mlflow.start_run(run_name="RandomForest_Baseline"):
            rf = RandomForestClassifier(featuresCol="features", labelCol=label_col, numTrees=50)
            rf_model = rf.fit(train_baseline_df)
            mlflow.spark.log_model(rf_model, "rf_baseline")
            mlflow.log_param("features", ",".join([c for c in ["trip_distance","passenger_count","fare_amount"] if c in train_baseline_df.columns]))
            print("Logged RandomForest_Baseline run to MLflow.")
    except Exception as e:
        print("Baseline RF training/logging failed:", e)

    # Graph-enhanced Random Forest
    try:
        with mlflow.start_run(run_name="RandomForest_GraphFeatures"):
            rf_graph = RandomForestClassifier(featuresCol="features", labelCol=label_col, numTrees=50)
            rf_graph_model = rf_graph.fit(train_graph_df)
            mlflow.spark.log_model(rf_graph_model, "rf_graph")
            mlflow.log_param("features", "trip_distance, passenger_count, fare_amount, pagerank, degree, component")
            print("Logged RandomForest_GraphFeatures run to MLflow.")
    except Exception as e:
        print("Graph RF training/logging failed:", e)

# ---------------------
# 9. XGBoost + PyTorch Examples (small sample)
# ---------------------
def section_xgboost_pytorch(train_baseline_df, train_graph_df, label_col):
    print("\n-- XGBoost & PyTorch demos (converted to pandas; sampling to avoid OOM) --")
    xgb_mod, xgb_err = safe_import("xgboost")
    torch_mod, torch_err = safe_import("torch")

    # Convert to pandas with a limited sample
    try:
        pdf_base = train_baseline_df.select("features", label_col).limit(SAMPLE_FOR_XGB_AND_TORCH).toPandas()
        X_base = np.array(pdf_base["features"].tolist())
        y_base = pdf_base[label_col].astype(int)
    except Exception as e:
        print("Failed to convert baseline features to pandas:", e)
        pdf_base = None

    try:
        pdf_graph = train_graph_df.select("features", label_col).limit(SAMPLE_FOR_XGB_AND_TORCH).toPandas()
        X_graph = np.array(pdf_graph["features"].tolist())
        y_graph = pdf_graph[label_col].astype(int)
    except Exception as e:
        print("Failed to convert graph features to pandas:", e)
        pdf_graph = None

    # XGBoost
    if xgb_mod is not None and pdf_base is not None:
        import xgboost as xgb
        import mlflow.xgboost
        try:
            dtrain_base = xgb.DMatrix(X_base, label=y_base)
            with mlflow.start_run(run_name="XGBoost_Baseline"):
                xgb_base = xgb.train({"objective":"binary:logistic","eval_metric":"auc"}, dtrain_base, num_boost_round=50)
                mlflow.xgboost.log_model(xgb_base, "xgb_baseline")
                print("Logged XGBoost_Baseline.")
        except Exception as e:
            print("XGBoost baseline failed:", e)

    if xgb_mod is not None and pdf_graph is not None:
        import xgboost as xgb
        import mlflow.xgboost
        try:
            dtrain_graph = xgb.DMatrix(X_graph, label=y_graph)
            with mlflow.start_run(run_name="XGBoost_GraphFeatures"):
                xgb_graph = xgb.train({"objective":"binary:logistic","eval_metric":"auc"}, dtrain_graph, num_boost_round=50)
                mlflow.xgboost.log_model(xgb_graph, "xgb_graph")
                print("Logged XGBoost_GraphFeatures.")
        except Exception as e:
            print("XGBoost graph failed:", e)
    else:
        if xgb_mod is None:
            print(xgb_err)

    # PyTorch small demo
    if torch_mod is None:
        print(torch_err)
    else:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import mlflow.pytorch

        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(16, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

        if pdf_base is not None:
            X_t = torch.tensor(X_base, dtype=torch.float32)
            y_t = torch.tensor(y_base.values, dtype=torch.float32)
            model_base = SimpleNN(X_base.shape[1])
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model_base.parameters(), lr=0.01)
            with mlflow.start_run(run_name="PyTorch_Baseline"):
                for epoch in range(3):
                    optimizer.zero_grad()
                    outputs = model_base(X_t).squeeze()
                    loss = criterion(outputs, y_t)
                    loss.backward()
                    optimizer.step()
                mlflow.pytorch.log_model(model_base, "pytorch_baseline")
                print("Logged PyTorch_Baseline.")

        if pdf_graph is not None:
            X_tg = torch.tensor(X_graph, dtype=torch.float32)
            y_tg = torch.tensor(y_graph.values, dtype=torch.float32)
            model_graph = SimpleNN(X_graph.shape[1])
            optimizer = optim.Adam(model_graph.parameters(), lr=0.01)
            with mlflow.start_run(run_name="PyTorch_GraphFeatures"):
                for epoch in range(3):
                    optimizer.zero_grad()
                    outputs = model_graph(X_tg).squeeze()
                    loss = criterion(outputs, y_tg)
                    loss.backward()
                    optimizer.step()
                mlflow.pytorch.log_model(model_graph, "pytorch_graph")
                print("Logged PyTorch_GraphFeatures.")

# ---------------------
# 10-12. Notes and Wrap-up
# ---------------------
def section_notes_and_wrapup():
    md = """
    <h2>MLflow Visualizations</h2>
    <ul>
      <li>Use the MLflow UI to compare baseline vs graph-enhanced models (Parallel Coordinates, Scatter Plots, Line Charts).</li>
      <li>Use the 'features' param to group runs in MLflow and separate baseline vs graph runs.</li>
      <li>Artifacts (ROC curves, confusion matrices) can be logged and inspected per run.</li>
    </ul>
    <h2>Collaboration & Dashboards</h2>
    <ul>
      <li>Notebook inline comments for peer review.</li>
      <li>Git integration for version control of notebooks.</li>
      <li>Schedule jobs for automated runs; pin charts to dashboards for stakeholders.</li>
    </ul>
    <h2>Wrap-Up</h2>
    <p>This demo covered Spark transformations, Delta Lake, MLlib/ XGBoost/ PyTorch, GraphFrames, MLflow tracking, and collaboration features in Databricks.</p>
    """
    display_markdown(md)

# ---------------------
# Main orchestration
# ---------------------
def main():
    # Section 1
    if RUN_SECTIONS.get("intro"):
        section_intro()

    # Section 2
    if RUN_SECTIONS.get("load"):
        df = section_load()
    else:
        df = None

    # Section 3
    if RUN_SECTIONS.get("transform") and df is not None:
        section_transform(df)

    # Section 4
    if RUN_SECTIONS.get("delta") and df is not None:
        section_delta(df)

    # Section 5
    train_baseline_df = None
    label_col = None
    if RUN_SECTIONS.get("mllib") and df is not None:
        train_baseline_df, label_col = section_mllib(df)

    # Section 6
    graph = None
    if RUN_SECTIONS.get("graphframes") and df is not None:
        graph = section_graphframes(df)

    # Section 7
    train_graph_df = None
    if RUN_SECTIONS.get("graph_features") and df is not None:
        train_graph_df = section_graph_features(df, graph)

    # If MLlib didn't produce a label or features, try to infer label_col from train_graph_df
    if label_col is None and train_graph_df is not None:
        # naive attempt
        label_col = choose_column(train_graph_df, ["payment_type", "payment", "paymenttype"])

    # Section 8
    if RUN_SECTIONS.get("mlflow") and train_baseline_df is not None and train_graph_df is not None and label_col is not None:
        section_mlflow(train_baseline_df, train_graph_df, label_col)
    else:
        print("Skipping MLflow section (missing dataframes or label column).")

    # Section 9
    if RUN_SECTIONS.get("xgboost_pytorch") and train_baseline_df is not None and train_graph_df is not None and label_col is not None:
        # numpy needed for conversions
        try:
            import numpy as np
        except Exception:
            print("numpy is required for XGBoost/PyTorch section. Install numpy on the cluster.")
            return
        section_xgboost_pytorch(train_baseline_df, train_graph_df, label_col)
    else:
        print("Skipping XGBoost/PyTorch section (missing prerequisites).")

    # Notes & wrap-up
    if RUN_SECTIONS.get("notes"):
        section_notes_and_wrapup()

if __name__ == "__main__":
    main()
