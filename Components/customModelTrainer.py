import streamlit as st
import pandas as pd
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from datetime import datetime
from sklearn.preprocessing import StandardScaler


import os
from Components.metrics import load_data, load_metadata, plot_distribution, plot_column_shape, run_all_metrics, plot_distributions, generate_report

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns with all nulls
    df = df.dropna(axis=1, how='all')

    # Fill missing values (numerical with mean, categorical with mode)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Optional: encode categorical variables as strings
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    return df

def Section():
    tab1, tab2 = st.tabs(["📌 Train New Model", "♻️ Continue Existing Model"])

    # --- TAB 1: Train New Model ---
    with tab1:
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

        model_type = st.selectbox("Choose a model to train", ["CTGAN", "TVAE"])

        if uploaded_file:
            user_data = pd.read_csv(uploaded_file)
            user_data = preprocess_data(user_data)
            st.write("Preview of uploaded data:")
            st.dataframe(user_data)

            if st.button("🚀 Train Model"):
                with st.spinner("Training in progress..."):
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(user_data)

                    model = CTGANSynthesizer(metadata) if model_type == "CTGAN" else TVAESynthesizer(metadata)
                    model.fit(user_data)

                    # Save to session state
                    st.session_state.custom_model = model
                    st.session_state.user_data = user_data
                    st.session_state.custom_metadata = metadata

                st.success(f"{model_type} model trained successfully!")

    # --- TAB 2: Continue Existing Model ---
    with tab2:
        model_file = st.file_uploader("Upload trained model (.pkl)", type="pkl")
        new_data_file = st.file_uploader("Upload new training data (CSV)", type="csv", key="continue_data")

        if model_file and new_data_file:
            user_data = pd.read_csv(new_data_file)
            st.write("Preview of new data:")
            st.dataframe(user_data)

            if st.button("🔄 Continue Training"):
                with st.spinner("Loading and training..."):
                    model = CTGANSynthesizer.load(model_file)  # Works for both CTGAN and TVAE
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(user_data)

                    model.fit(user_data)

                    st.session_state.custom_model = model
                    st.session_state.user_data = user_data
                    st.session_state.custom_metadata = metadata

                st.success("Model retrained successfully!")

    # --- Shared: Generate, Save, Evaluate ---
    if 'custom_model' in st.session_state:
        st.divider()
        st.subheader("✅ Model Ready")

        n_samples = st.slider("Number of synthetic rows to generate", 10, 1000, 100)

        if st.button("🎲 Generate Synthetic Data"):
            synthetic = st.session_state.custom_model.sample(n_samples)
            st.session_state.synthetic_data = synthetic
            st.success(f"{n_samples} synthetic rows generated.")
            st.dataframe(synthetic)

        if st.button("💾 Save Trained Model"):
            os.makedirs("saved_models", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"saved_models/custom_model_{timestamp}.pkl"
            st.session_state.custom_model.save(model_path)
            st.success(f"Model saved to `{model_path}`")

    if 'synthetic_data' in st.session_state:
        st.subheader("📊 Evaluation")

        real_data = st.session_state.user_data
        synthetic_data = st.session_state.synthetic_data
        metadata = st.session_state.custom_metadata

        if st.button("🧪 Run Evaluation"):
            plot_distributions(real_data, synthetic_data)
            generate_report(real_data, synthetic_data, metadata)