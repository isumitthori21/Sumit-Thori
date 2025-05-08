import streamlit as st
import pandas as pd
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from datetime import datetime
import pickle
import os
from Components.metrics import  plot_distributions, generate_report
from Components.metrics_final import plot_single, heatmap_matrix

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
    tab1, tab2 = st.tabs(["📌 Train New Model", "♻️ Use Custom Model"])

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
            if 'custom_model' in st.session_state:
                if st.button("💾 Save Trained Model"):
                    os.makedirs("saved_models", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = f"saved_models/custom_model_{timestamp}.pkl"
                    st.session_state.custom_model.save(model_path)
                    st.success(f"Model saved to `{model_path}`")
                

    # --- TAB 2: Continue Existing Model ---
    with tab2:
        model_file = st.file_uploader("Upload trained model (.pkl)", type="pkl")

        if model_file:
            # Load model once uploaded
            loaded_model = pickle.load(model_file)
            st.session_state['custom_model'] = loaded_model
            st.divider()
            st.subheader("✅ Model Ready")

            n_samples = st.slider("Number of synthetic rows to generate", 10, 1000, 100)

            if st.button("🎲 Generate Synthetic Data"):
                synthetic = st.session_state['custom_model'].sample(n_samples)
                st.session_state['synthetic_data'] = synthetic
                st.success(f"{n_samples} synthetic rows generated.")
                st.dataframe(synthetic)
                if synthetic is not None:
                    st.dataframe(synthetic)

                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        st.download_button(
                            "⬇️ Download CSV",
                            synthetic.to_csv(index=False).encode("utf-8"),
                            "synthetic_data.csv",
                            "text/csv",
                            key="download-csv"
                        )

        if 'synthetic_data' in st.session_state:
            st.subheader("📊 Analyze")
            synthetic_data = st.session_state['synthetic_data']

            # Button sets a flag in session_state
            if st.button("🧪 Run Evaluation"):
                st.session_state['run_evaluation'] = True

            # Check persistent flag after rerun
            if st.session_state.get('run_evaluation', False):
                st.write("📈 Plot Pair Distribution")
                plot_single(synthetic_data)
                st.write("📊 HeatMap Matrix")
                heatmap_matrix(synthetic_data) 
