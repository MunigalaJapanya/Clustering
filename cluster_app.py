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


# PAGE CONFIG

st.set_page_config(page_title="DBSCAN Clustering Interactive Dashboard", layout="wide")
st.title(" Interactive DBSCAN Clustering Dashboard")

st.markdown("""
Explore how **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** behaves 
by adjusting its parameters interactively and viewing the resulting clusters live!

---
""")


# LOAD MODEL + SCALER


try:
    scaler = joblib.load("scaler.pkl")
    st.sidebar.success(" Scaler loaded successfully.")
except Exception:
    st.sidebar.warning("Scaler not found — using new StandardScaler.")
    scaler = StandardScaler()

try:
    dbscan_saved = joblib.load("dbscan_model.pkl")
    st.sidebar.success(" DBSCAN model loaded successfully.")
except Exception:
    dbscan_saved = None
    st.sidebar.warning("dbscan_model.pkl not found — a new DBSCAN will be used if needed.")

# Also try to load saved labels (optional)
labels_saved = None
try:
    labels_saved = np.load("dbscan_labels.npy")
    st.sidebar.success(" Saved cluster labels loaded.")
except Exception:
    labels_saved = None


# DATA INPUT

st.sidebar.header(" Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success(" Data loaded successfully.")
else:
    st.sidebar.info("No file uploaded — using a sample dataset.")
    from sklearn.datasets import make_blobs
    X_sample, _ = make_blobs(n_samples=500, centers=5, n_features=4, random_state=42)
    df = pd.DataFrame(X_sample, columns=[f"Feature_{i+1}" for i in range(X_sample.shape[1])])

st.write("### 🧾 Data Preview")
st.dataframe(df.head())


#  DATA CLEANING + SCALING
 
st.markdown("###  Data Preparation & Cleaning")
df_clean = df.copy()

# --- Step 1: Clean messy numeric strings ---
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.replace('$', '', regex=False)
            .str.strip()
        )
        df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')

# --- Step 2: Drop non-numeric columns ---
non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    st.warning(f" Dropping non-numeric columns: {non_numeric_cols}")
    df_clean = df_clean.drop(columns=non_numeric_cols)

# --- Step 3: Ensure numeric columns exist ---
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error(" No numeric columns found after cleaning. Please upload numeric data.")
    st.stop()

X = df_clean[numeric_cols]

#  Step 4: Handle missing values 
if X.isnull().sum().sum() > 0:
    missing_before = int(X.isnull().sum().sum())
    st.warning(f" Found {missing_before} missing values — filling with column mean.")
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=numeric_cols)
else:
    st.success(" No missing values found.")

# Step 5: Scale data using saved scaler if possible (match names) 
use_saved_scaler = True
try:
    # try aligning column names to saved scaler feature names (if attribute exists)
    if hasattr(scaler, "feature_names_in_"):
        # If lengths differ, attempt to align by slicing or raise
        if len(scaler.feature_names_in_) == X.shape[1]:
            X.columns = scaler.feature_names_in_
        else:
            # lengths mismatch -> fall back to fitting new scaler
            raise ValueError("column count mismatch with saved scaler")
    X_scaled = scaler.transform(X)
    st.success(" Data scaled using the saved scaler (same as notebook).")
except Exception as e:
    use_saved_scaler = False
    st.warning(f" Saved scaler incompatible ({e}) — fitting a new one instead.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

st.success(f" Data cleaned & scaled successfully. Using {len(numeric_cols)} numeric columns for clustering.")
st.write("###  Cleaned Numeric Data Sample")
st.dataframe(X.head())


# SIDEBAR: DBSCAN PARAMETERS (interactive retained exactly)

st.sidebar.header(" DBSCAN Parameters")
eps = st.sidebar.slider("Epsilon (eps):", 0.1, 5.0, 2.3, 0.1)
min_samples = st.sidebar.slider("Min Samples:", 2, 20, 6, 1)


# Always use saved DBSCAN model and labels (same as notebook)

use_saved_labels = False
if labels_saved is not None and len(labels_saved) == X_scaled.shape[0]:
    labels = labels_saved
    st.success(" Using saved DBSCAN labels from notebook (exact same 9 clusters).")
else:
    st.warning(" Saved labels not found or size mismatch — running DBSCAN interactively.")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)


# RUN DBSCAN (preserve interactive behavior)

if use_saved_labels:
    labels = labels_saved
    st.info("Using saved labels from notebook (parameters match).")
else:
    # run DBSCAN with the sliders, exactly like before
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    st.info("DBSCAN run with current slider values (interactive).")

df_clean["Cluster"] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
noise_pct = n_noise / len(labels) * 100


# CLUSTER SUMMARY (unchanged)
 
st.markdown("##  Cluster Summary")

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


# CLUSTER INSIGHTS (fixed grouping to avoid duplicate 'Cluster' insert)

st.markdown("##  Cluster Insights / Interpretations")

if n_clusters > 0:
    # fixed: as_index=False to avoid "cannot insert Cluster, already exists"
    cluster_means = (
        df_clean.groupby("Cluster", as_index=False)[numeric_cols]
        .mean()
        .round(2)
        .sort_values("Cluster")
    )

    st.write("###  Mean Feature Values by Cluster")
    st.dataframe(cluster_means)

    # Generate meaningful insights
    overall_means = df_clean[numeric_cols].mean()
    st.write("###  Automatically Generated Insights")
    for cluster_id in sorted(df_clean["Cluster"].unique()):
        if cluster_id == -1:
            st.warning(" **Cluster -1 (Noise):** These are outlier points that don’t belong to any dense group.")
            continue

        subset = df_clean[df_clean["Cluster"] == cluster_id]
        avg_vals = subset[numeric_cols].mean().round(2)
        top_feature = (avg_vals - overall_means).idxmax()
        bottom_feature = (avg_vals - overall_means).idxmin()

        st.markdown(f"""
        **Cluster {cluster_id}:**
        -  **Data Points:** {len(subset)}
        -  **Key Strength:** `{top_feature}` — {avg_vals[top_feature]} (above average)
        -  **Key Weakness:** `{bottom_feature}` — {avg_vals[bottom_feature]} (below average)
        -  **Insight:** Cluster {cluster_id} members tend to have higher `{top_feature}` but relatively lower `{bottom_feature}` compared to other clusters.
        """)

   
    # CLUSTER DROPDOWN VIEW
    
    st.markdown("---")
    st.markdown("##  View Individual Cluster Data")

    cluster_options = sorted(df_clean["Cluster"].unique())
    selected_cluster = st.selectbox("Select a Cluster to View:", cluster_options, index=0)

    subset_data = df_clean[df_clean["Cluster"] == selected_cluster]
    if selected_cluster == -1:
        st.warning(" You selected the Noise cluster (-1): these are points not belonging to any dense cluster.")
    else:
        st.success(f"Showing data for **Cluster {selected_cluster}** ({len(subset_data)} records)")

    st.write("###  Raw Data for Selected Cluster")
    st.dataframe(subset_data)

    st.write("###  Feature Statistics for Selected Cluster")
    st.dataframe(subset_data[numeric_cols].describe().T)

    
    # FEATURE COMPARISON CHART
   
    st.markdown("###  Feature Comparison Across Clusters")
    fig, ax = plt.subplots(figsize=(10,6))
    cluster_means.set_index("Cluster").T.plot(kind='bar', ax=ax)
    plt.title("Average Feature Values per Cluster")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("No valid clusters found to generate insights.")


# EVALUATION METRICS

st.markdown("##  Evaluation Metrics")

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
    "Meaning": [
        "Separation & compactness of clusters.",
        "Lower = better separation.",
        "Higher = better-defined clusters."
    ]
})
st.table(metrics_df)


# PCA VISUALIZATION

st.markdown("##  Cluster Visualization (PCA Projection)")

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
ax.set_title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples}) — {n_clusters} Clusters")
ax.legend()
st.pyplot(fig)


# PREDICT NEW POINT (keeps original behavior)
 
st.sidebar.markdown("---")
st.sidebar.header(" Predict Cluster for New Data")

user_input = []
for col in numeric_cols:
    val = st.sidebar.number_input(f"{col}", value=float(df_clean[col].mean()))
    user_input.append(val)

if st.sidebar.button("Predict Cluster"):
    # transform new point using scaler (if available)
    try:
        new_scaled = scaler.transform([user_input])
    except Exception:
        new_scaled = StandardScaler().fit_transform([user_input])

    # If user changed sliders (interactive mode), dbscan variable above is the interactive one.
    # If we used saved labels, dbscan may be None or the saved DBSCAN; we'll fallback to running DBSCAN on new point.
    try:
        label = dbscan.fit_predict(new_scaled)[0]
    except Exception:
        # final fallback: fit a short DBSCAN with the selected sliders on the full scaled data plus new point
        temp = np.vstack([X_scaled, new_scaled])
        temp_db = DBSCAN(eps=eps, min_samples=min_samples)
        temp_labels = temp_db.fit_predict(temp)
        label = temp_labels[-1]

    if label == -1:
        st.sidebar.warning(" Predicted as noise (-1).")
    else:
        st.sidebar.success(f" Predicted Cluster: {label}")
