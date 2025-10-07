import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===============================
# Load Pretrained Models
# ===============================
similarity_df = joblib.load("product_similarity_matrix.pkl")
kmeans = joblib.load("rfm_kmeans_model.pkl")
scaler = joblib.load("rfm_scaler.pkl")

# Cluster labels mapping
cluster_labels = {
    0: 'High-Value',
    1: 'Regular',
    2: 'Occasional',
    3: 'At-Risk'
}


st.sidebar.title("Home")

# Sub-module selector inside Home
module = st.sidebar.radio("Select Module:", ["Clustering", "Recommendation"])

if module == "Clustering":
    st.header("🎯 Customer Segmentation (RFM-based)")

    recency = st.number_input("Recency (days since last purchase):", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases):", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend):", min_value=0.0, step=0.01, format="%.2f")

    if st.button("Predict Cluster"):
        customer_rfm = [[recency, frequency, monetary]]
        customer_scaled = scaler.transform(customer_rfm)
        cluster = kmeans.predict(customer_scaled)[0]
        segment = cluster_labels.get(cluster, "Unknown")
        st.success(f"This customer belongs to: **{segment}**")

elif module == "Recommendation":
    st.header("🎯 Product Recommender")
    product_name = st.text_input("Enter Product Name")

    def get_similar_products(product_name, top_n=5):
        if product_name not in similarity_df.columns:
            return []
        scores = similarity_df[product_name].sort_values(ascending=False).drop(product_name)
        return scores.head(top_n).index.tolist()

    if st.button("Recommendations"):
        if not product_name.strip():
            st.warning("Please enter a product name!")
        else:
            recs = get_similar_products(product_name)
            if not recs:
                st.error("Product not found in the database!")
            else:
                st.success(f"Recommended products:")
                for i, rec in enumerate(recs, start=1):
                    st.markdown(f"**{i}. {rec}**")
