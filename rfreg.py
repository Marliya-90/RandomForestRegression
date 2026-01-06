# app.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Taxi Trip Pricing", layout="wide")
st.title("🚖 Taxi Trip Pricing Analysis")

# -----------------------------
# 1️⃣ Check CSV in repo
# -----------------------------
DATA_PATH = "taxi_trip_pricingR.csv"  # make sure this is in the same folder as app.py

# Function to load CSV
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.success(f"✅ Loaded CSV from repo: {DATA_PATH}")
    else:
        st.warning(f"❌ File not found in repo. Please upload your CSV.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV uploaded successfully!")
        else:
            st.stop()
    return df

# Load the data
df = load_data()

# -----------------------------
# 2️⃣ Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Dataset Info")
st.write("Shape:", df.shape)
st.write("Columns:", df.columns.tolist())
st.write(df.describe())

# -----------------------------
# 3️⃣ Data Visualization
# -----------------------------
st.subheader("Data Visualization")

# Numeric columns for plotting
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_cols:
    st.write("### Histogram of Numeric Columns")
    selected_col = st.selectbox("Select column for histogram", numeric_cols)
    bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
    plt.figure(figsize=(8,4))
    sns.histplot(df[selected_col], bins=bins, kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()
else:
    st.info("No numeric columns found for visualization.")

# -----------------------------
# 4️⃣ Column Selection for Regression
# -----------------------------
st.subheader("Select Features & Target for Regression")
features = st.multiselect("Select feature columns", df.columns.tolist(), default=numeric_cols[:-1])
target = st.selectbox("Select target column", df.columns.tolist(), index=len(numeric_cols)-1)

st.write("Selected Features:", features)
st.write("Target:", target)

st.success("✅ Ready to use features and target for model training!")
