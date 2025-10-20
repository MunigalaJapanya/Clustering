import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os


# PAGE CONFIG
st.set_page_config(page_title="DBSCAN Clustering Interactive Dashboard", layout="wide")
st.title("🌀 Interactive DBSCAN Clustering Dashboard")

st.markdown("""
Explore how **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** behaves 
by adjusting its parameters interactively and viewing the resulting clusters live!
---
""")

# --------------------- LOAD OR INITIALIZE SCALER ---------------------

scaler_path = "scaler.pkl"
dbscan_path = "dbscan_model.pkl"
labels_path = "dbscan_labels.npy"

if os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        st.sidebar.success("✅ Scaler loaded successfully.")
    except Exception:
        st.sidebar.warning("⚠️ Scaler file found but incompatible — creating a new one.")
        scaler = StandardScaler()
else:
    scaler = StandardScaler()
    st.sidebar.info("ℹ️ No scaler found — new scaler will be fitted and saved after processing.")

if os.path.exists(dbscan_path):
    try:
        dbscan_saved = joblib.load(dbscan_path)
        st.sidebar.success("✅ DBSCAN model loaded successfully.")
    except Exception:
        dbscan_saved = None
        st.sidebar.warning("⚠️ DBSCAN model incompatible — a new one will be trained.")
else:
    dbscan_saved = None
    st.sidebar.info("ℹ️ No saved DBSCAN model found — one will be created.")

if os.path.exists(labels_path):
    try:
        labels_saved = np.load(labels_path)
        st.sidebar.success("✅ Saved cluster labels loaded.")
    except Exception:
        labels_saved = None
        st.sidebar.warning("⚠️ Labels file found but incompatible.")
else:
    labels_saved = None
    st.sidebar.info("ℹ️ No saved labels found.")


# --------------------- LOAD DATASET ---------------------

st.sidebar.header("📂 Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ Data loaded successfully.")
else:
    st.sidebar.info("No file uploaded — using a sample dataset.")
    from sklearn.datasets import make_blobs
    X_sample, _ = make_blobs(n_samples=500, centers=5, n_features=4, random_state=42)
    df = pd.DataFrame(X_sample, columns=[f"Feature_{i+1}" for i in range(X_sample.shape[1])])

st.write("### 🧾 Data Preview")
st.dataframe(df.head())

# --------------------- DATA CLEANING ---------------------

st.markdown("### 🧹 Data Preparation & Cleaning")
df_clean = df.copy()

# Step 1: Clean object columns
for col in df_clean.columns:
    if df_clean[col].dtype == "object":
        df_clean[col] = (
            df_clean[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

# Step 2: Drop non-numeric columns
non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    st.warning(f"Dropping non-numeric columns: {non_numeric_cols}")
    df_clean = df_clean.drop(columns=non_numeric_cols)

# Step 3: Handle missing values
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("No numeric columns found after cleaning.")
    st.stop()

X = df_clean[numeric_cols]
if X.isnull().sum().sum() > 0:
    st.warning("Missing values found — filling with mean.")
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=numeric_cols)
else:
    st.success("No missing values found.")

# Step 4: Scaling
try:
    if hasattr(scaler, "mean_"):
        X_scaled = scaler.transform(X)
        st.success("Data scaled using existing saved scaler.")
    else:
        raise ValueError("Scaler not fitted yet.")
except Exception:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)
    st.success("New scaler fitted and saved successfully.")

st.write("### 📊 Cleaned Numeric Data Sample")
st.dataframe(X.head())

# --------------------- SIDEBAR DBSCAN PARAMETERS ---------------------

st.sidebar.header("⚙️ DBSCAN Parameters")
eps = st.sidebar.slider("Epsilon (eps):", 0.1, 5.0, 2.3, 0.1)
min_samples = st.sidebar.slider("Min Samples:", 2, 20, 6, 1)

# --------------------- RUN OR LOAD DBSCAN ---------------------

if labels_saved is not None and len(labels_saved) == X_scaled.shape[0]:
    labels = labels_saved
    st.success("Using saved DBSCAN labels from previous run.")
else:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    joblib.dump(dbscan, dbscan_path)
    np.save(labels_path, labels)
    st.info("DBSCAN fitted, and model + labels saved for next run.")

df_clean["Cluster"] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
noise_pct = n_noise / len(labels) * 100

# --------------------- CLUSTER SUMMARY ---------------------

st.markdown("## 📈 Cluster Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Clusters Formed", n_clusters)
col2.metric("Noise Points", n_noise)
col3.metric("Noise %", f"{noise_pct:.2f}%")

cluster_summary = (
    df_clean["Cluster"].value_counts()
    .rename_axis("Cluster")
    .reset_index(name="Count")
    .assign(**{"% of Total": lambda x: (x["Count"] / len(df_clean) * 100).round(2)})
    .sort_values("Cluster")
)
st.dataframe(cluster_summary)

# --------------------- CLUSTER INSIGHTS ---------------------

st.markdown("## 🔍 Cluster Insights / Interpretations")

if n_clusters > 0:
    cluster_means = (
        df_clean.groupby("Cluster", as_index=False)[numeric_cols]
        .mean()
        .round(2)
        .sort_values("Cluster")
    )

    st.write("### Mean Feature Values by Cluster")
    st.dataframe(cluster_means)

    overall_means = df_clean[numeric_cols].mean()
    for cluster_id in sorted(df_clean["Cluster"].unique()):
        if cluster_id == -1:
            st.warning("Cluster -1 (Noise): Outliers that don’t belong to any cluster.")
            continue
        subset = df_clean[df_clean["Cluster"] == cluster_id]
        avg_vals = subset[numeric_cols].mean().round(2)
        top_feature = (avg_vals - overall_means).idxmax()
        bottom_feature = (avg_vals - overall_means).idxmin()
        st.markdown(f"""
        **Cluster {cluster_id}:**
        - **Data Points:** {len(subset)}
        - **Top Feature:** `{top_feature}` ({avg_vals[top_feature]} — above avg)
        - **Weak Feature:** `{bottom_feature}` ({avg_vals[bottom_feature]} — below avg)
        """)

    st.markdown("### 🧮 Feature Comparison Across Clusters")
    fig, ax = plt.subplots(figsize=(10,6))
    cluster_means.set_index("Cluster").T.plot(kind='bar', ax=ax)
    plt.title("Average Feature Values per Cluster")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("No valid clusters found.")

# --------------------- EVALUATION METRICS ---------------------

st.markdown("## 📊 Evaluation Metrics")
mask = labels != -1
if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
    silhouette = silhouette_score(X_scaled[mask], labels[mask])
    dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
    chi = calinski_harabasz_score(X_scaled[mask], labels[mask])
else:
    silhouette, dbi, chi = np.nan, np.nan, np.nan

metrics_df = pd.DataFrame({
    "Metric": ["Silhouette (↑)", "Davies–Bouldin (↓)", "Calinski–Harabasz (↑)"],
    "Value": [round(silhouette, 3), round(dbi, 3), round(chi, 3)],
})
st.table(metrics_df)

# --------------------- PCA VISUALIZATION ---------------------

st.markdown("## 🎨 Cluster Visualization (PCA Projection)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = labels

fig, ax = plt.subplots(figsize=(8,6))
for cluster_id in sorted(df_pca["Cluster"].unique()):
    subset = df_pca[df_pca["Cluster"] == cluster_id]
    label = "Noise (-1)" if cluster_id == -1 else f"Cluster {cluster_id}"
    color = "gray" if cluster_id == -1 else None
    ax.scatter(subset["PC1"], subset["PC2"], s=30, label=label, alpha=0.7, color=color)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
st.pyplot(fig)

# --------------------- PREDICT NEW DATA ---------------------

st.sidebar.markdown("---")
st.sidebar.header("🔮 Predict Cluster for New Data")

user_input = []
for col in numeric_cols:
    val = st.sidebar.number_input(f"{col}", value=float(df_clean[col].mean()))
    user_input.append(val)

if st.sidebar.button("Predict Cluster"):
    new_scaled = scaler.transform([user_input])
    temp = np.vstack([X_scaled, new_scaled])
    temp_db = DBSCAN(eps=eps, min_samples=min_samples)
    temp_labels = temp_db.fit_predict(temp)
    label = temp_labels[-1]

    if label == -1:
        st.sidebar.warning("Predicted as noise (-1).")
    else:
        st.sidebar.success(f"Predicted Cluster: {label}")
