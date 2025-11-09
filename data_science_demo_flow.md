Got it — let’s stitch everything together into a **single, ordered Databricks demo notebook**. This way you can copy/paste cell by cell and run it smoothly. I’ll keep the flow logical: intro → Spark basics → Delta Lake → MLlib → GraphFrames → Graph‑enhanced ML → MLflow comparisons → Visualizations → Collaboration/Dashboard wrap‑up.  

---

# 📓 Databricks Demo Notebook (Taxi Dataset Showcase)

---

### 1. Introduction (Markdown)
```markdown
# Databricks Demo Notebook
Welcome! In this demo we’ll explore Databricks capabilities:
- Scalable Spark data processing
- Delta Lake reliability
- Machine Learning with MLlib, XGBoost, PyTorch
- Graph analytics with GraphFrames
- MLflow experiment tracking and visualization
- Collaboration and dashboards
```

---

### 2. Load Taxi Data (Code)
```python
# Load NYC taxi dataset (Parquet from TLC public source)
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
df = spark.read.parquet(url)
display(df.limit(5))
```

---

### 3. Spark Transformations (Markdown + Code)
```markdown
## Spark Transformations
Basic aggregations to show distributed processing.
```
```python
df.groupBy("passenger_count").agg({"trip_distance": "avg"}).show()
```

---

### 4. Delta Lake Demo (Markdown + Code)
```markdown
## Delta Lake
Persist data in Delta format, update it, and query history.
```
```python
df.write.format("delta").mode("overwrite").save("/tmp/taxi_delta")
delta_df = spark.read.format("delta").load("/tmp/taxi_delta")
display(delta_df.limit(5))

from delta.tables import DeltaTable
delta_table = DeltaTable.forPath(spark, "/tmp/taxi_delta")
delta_table.update(
    condition="passenger_count = 0",
    set={"passenger_count": "1"}
)
```

---

### 5. MLlib Model (Markdown + Code)
```markdown
## MLlib Logistic Regression
Train a simple logistic regression model.
```
```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

assembler = VectorAssembler(inputCols=["trip_distance", "passenger_count", "fare_amount"], outputCol="features")
train_df_baseline = assembler.transform(df.na.drop(subset=["trip_distance","passenger_count","fare_amount","payment_type"]))

lr = LogisticRegression(featuresCol="features", labelCol="payment_type")
lr_model = lr.fit(train_df_baseline)
print("Coefficients:", lr_model.coefficients)
```

---

### 6. GraphFrames (Taxi Data as Graph) (Markdown + Code)
```markdown
## Graph Analytics with GraphFrames
Convert taxi trips into graph data: locations as vertices, trips as edges.
```
```python
from graphframes import GraphFrame
from pyspark.sql.functions import lit

vertices = df.selectExpr("pickup_location_id as id").union(
    df.selectExpr("dropoff_location_id as id")
).distinct().withColumn("type", lit("location"))

edges = df.selectExpr("pickup_location_id as src","dropoff_location_id as dst","fare_amount as weight")
edges = edges.groupBy("src","dst").count().withColumnRenamed("count","trip_count")

g_taxi = GraphFrame(vertices, edges)

g_taxi.pageRank(resetProbability=0.15, maxIter=10).vertices.show()
g_taxi.connectedComponents().show()
```

---

### 7. Graph Features → ML Pipeline (Markdown + Code)
```markdown
## Graph Features for ML
Derive PageRank, degree, and connected components, then join back to trips.
```
```python
pagerank = g_taxi.pageRank(resetProbability=0.15, maxIter=10).vertices.select("id","pagerank")
degrees = g_taxi.degrees
components = g_taxi.connectedComponents().select("id","component")

graph_features = vertices.join(pagerank,"id").join(degrees,"id").join(components,"id")

df_with_graph = df.join(graph_features, df.pickup_location_id == graph_features.id, "left")

assembler_graph = VectorAssembler(
    inputCols=["trip_distance","passenger_count","fare_amount","pagerank","degree","component"],
    outputCol="features"
)
train_df_with_graph = assembler_graph.transform(df_with_graph.na.drop(subset=["trip_distance","passenger_count","fare_amount","payment_type"]))
```

---

### 8. MLflow Experiment Tracking (Markdown + Code)
```markdown
## MLflow Tracking
Compare baseline vs. graph‑enhanced models.
```
```python
import mlflow
import mlflow.spark
import mlflow.xgboost
import mlflow.pytorch

mlflow.set_experiment("/Users/<your_user>/TaxiGraphComparison")

# Baseline Random Forest
from pyspark.ml.classification import RandomForestClassifier
with mlflow.start_run(run_name="RandomForest_Baseline"):
    rf = RandomForestClassifier(featuresCol="features", labelCol="payment_type", numTrees=50)
    rf_model = rf.fit(train_df_baseline)
    mlflow.spark.log_model(rf_model, "rf_baseline")
    mlflow.log_param("features","trip_distance, passenger_count, fare_amount")

# Graph‑enhanced Random Forest
with mlflow.start_run(run_name="RandomForest_GraphFeatures"):
    rf_graph = RandomForestClassifier(featuresCol="features", labelCol="payment_type", numTrees=50)
    rf_graph_model = rf_graph.fit(train_df_with_graph)
    mlflow.spark.log_model(rf_graph_model, "rf_graph")
    mlflow.log_param("features","trip_distance, passenger_count, fare_amount, pagerank, degree, component")
```

---

### 9. XGBoost + PyTorch (Markdown + Code)
```markdown
## XGBoost and PyTorch Models
Train models with and without graph features.
```
```python
import xgboost as xgb
import numpy as np

pdf_baseline = train_df_baseline.select("features","payment_type").toPandas()
X_base = np.array(pdf_baseline["features"].tolist())
y_base = pdf_baseline["payment_type"].astype(int)

pdf_graph = train_df_with_graph.select("features","payment_type").toPandas()
X_graph = np.array(pdf_graph["features"].tolist())
y_graph = pdf_graph["payment_type"].astype(int)

# Baseline XGBoost
dtrain_base = xgb.DMatrix(X_base, label=y_base)
with mlflow.start_run(run_name="XGBoost_Baseline"):
    xgb_base = xgb.train({"objective":"binary:logistic","eval_metric":"auc"}, dtrain_base, num_boost_round=50)
    mlflow.xgboost.log_model(xgb_base,"xgb_baseline")

# Graph‑enhanced XGBoost
dtrain_graph = xgb.DMatrix(X_graph, label=y_graph)
with mlflow.start_run(run_name="XGBoost_GraphFeatures"):
    xgb_graph = xgb.train({"objective":"binary:logistic","eval_metric":"auc"}, dtrain_graph, num_boost_round=50)
    mlflow.xgboost.log_model(xgb_graph,"xgb_graph")
```

```python
# PyTorch Example
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self,input_dim):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(input_dim,16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

# Baseline
model_base = SimpleNN(X_base.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model_base.parameters(), lr=0.01)
X_torch_base = torch.tensor(X_base,dtype=torch.float32)
y_torch_base = torch.tensor(y_base,dtype=torch.float32)

with mlflow.start_run(run_name="PyTorch_Baseline"):
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model_base(X_torch_base).squeeze()
        loss = criterion(outputs,y_torch_base)
        loss.backward()
        optimizer.step()
    mlflow.pytorch.log_model(model_base,"pytorch_baseline")

# Graph‑enhanced
model_graph = SimpleNN(X_graph.shape[1])
optimizer = optim.Adam(model_graph.parameters(), lr=0.01)
X_torch_graph = torch.tensor(X_graph,dtype=torch.float32)
y_torch_graph = torch.tensor(y_graph,dtype=torch.float32)

with mlflow.start_run(run_name="PyTorch_GraphFeatures"):
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model_graph(X_torch_graph).squeeze()
        loss = criterion(outputs,y_torch_graph)
        loss.backward()
        optimizer.step()
    mlflow.pytorch.log_model(model_graph,"pytorch_graph")
```

---
Perfect — let’s finish the notebook cleanly. Here’s the continuation from **10. MLflow Visualizations**, followed by the **Collaboration & Dashboards** wrap‑up. You’ll then have a complete, copy‑paste‑ready demo notebook.

---

### 10. MLflow Visualizations (Markdown)
```markdown
## MLflow Visualizations

In the MLflow UI, you can compare baseline vs. graph‑enhanced models visually:

- **Parallel Coordinates**: Show how parameters (e.g., numTrees, learning_rate) affect metrics (e.g., AUC, accuracy).
- **Scatter Plot**: Plot training loss vs. AUC, grouped by `features` to separate baseline vs. graph runs.
- **Line Chart**: Compare convergence speed across epochs (PyTorch) or boosting rounds (XGBoost).
- **Group by Features**: Use the `features` parameter to clearly distinguish baseline vs. graph‑enhanced runs.
- **Artifacts**: View logged ROC curves, confusion matrices, or other plots alongside metrics.

👉 This makes the value of graph features visible not just in numbers, but in trends and patterns.
```

---

### 11. Collaboration & Dashboards (Markdown)
```markdown
## Collaboration & Dashboards

Databricks is more than a compute engine — it’s a collaborative workspace.

- **Notebook Comments**: Add inline comments on cells for peer review.
- **Git Integration**: Link notebooks to GitHub/GitLab for version control.
- **Job Scheduling**: Automate notebook runs with Databricks Jobs.
- **Dashboards**: Pin charts and MLflow visualizations to dashboards for stakeholders.
- **Permissions**: Fine‑grained access control ensures secure collaboration.

### Example: Pinning Charts
1. Hover over a chart in your notebook or MLflow run.
2. Click the 📌 pin icon.
3. Add it to a new or existing dashboard.

This allows non‑technical stakeholders to view live results without touching code.
```

---

### 12. Wrap‑Up (Markdown)
```markdown
# Wrap‑Up

In this demo, we explored:

- Spark transformations and Delta Lake reliability
- MLlib, XGBoost, and PyTorch models
- Graph analytics with GraphFrames
- Graph‑enhanced ML pipelines
- MLflow experiment tracking and visualizations
- Collaboration and dashboards

👉 Databricks unifies data, machine learning, and collaboration — enabling teams to go from raw data to production models in one platform.
```

---

✅ With this, you now have a **complete, ordered demo notebook**. You can copy/paste cell by cell into Databricks, run it on your ML Runtime cluster, and present a polished end‑to‑end story: **data → graph → ML → experiment tracking → dashboards**.

Would you like me to also prepare a **one‑page “executive summary” handout** (non‑technical, high‑level) that you can give stakeholders alongside the demo? That way they walk away with a clear takeaway of Databricks’ value.
