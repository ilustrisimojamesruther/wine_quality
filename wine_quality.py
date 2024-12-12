import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    return data

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", [
    "1. Overview",
    "2. Data Exploration and Preparation",
    "3. Analysis and Insights",
    "4. Conclusions and Recommendations"
])

# Load data
data = load_data()

if section == "1. Overview":
    # App title and description
    st.title("Wine Quality Analysis")
    st.markdown("""
    This Streamlit app explores the Wine Quality dataset, performs data analysis using clustering and regression techniques, 
    and provides interactive visualizations for insights.
    """)

    # Display dataset structure
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    st.markdown("*Dataset Structure:*")
    st.write(data.info())
    st.markdown("*Descriptive Statistics:*")
    st.write(data.describe())

if section == "2. Data Exploration and Preparation":
    # Data Cleaning and Preparation
    st.title("2. Data Exploration and Preparation")
    st.subheader("Data Cleaning")
    if st.checkbox("Show missing values count"):
        st.write(data.isnull().sum())

    st.subheader("Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='left')
    ax.set_yticklabels(corr.columns)
    for (i, j), val in np.ndenumerate(corr):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white')
    st.pyplot(fig)

if section == "3. Analysis and Insights":
    # Clustering Analysis
    st.title("3. Analysis and Insights")
    st.subheader("Clustering Analysis")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop("quality", axis=1))

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    st.write("Cluster Centers:")
    st.write(kmeans.cluster_centers_)

    # Cluster Visualization
    st.subheader("Cluster Visualization")
    x_col = st.selectbox("Select X-axis feature", data.columns[:-1])
    y_col = st.selectbox("Select Y-axis feature", data.columns[:-1])

    fig, ax = plt.subplots()
    scatter = ax.scatter(data[x_col], data[y_col], c=data['Cluster'], cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

if section == "4. Conclusions and Recommendations":
    # Conclusions and Recommendations
    st.title("4. Conclusions and Recommendations")
    st.markdown("""
    - *Key Insight 1:* Highlight relationships between features and wine quality.
    - *Key Insight 2:* Provide actionable insights for winemakers.

    Explore the app's interactive visualizations to uncover more insights.
    """)