import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
from Components.metrics import load_data, load_metadata, plot_distribution, plot_column_shape, run_all_metrics

def Section():
    @st.cache_resource
    def load_ctgan_model():
        return CTGANSynthesizer.load('model/ctgan_model.pkl')

    @st.cache_resource
    def load_tvae_model():
        return TVAESynthesizer.load('model/tvae_model.pkl')

    ctgan_model = load_ctgan_model()
    tvae_model = load_tvae_model()

    # --- Session state init ---
    if 'synthetic_data' not in st.session_state:
        st.session_state.synthetic_data = None
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False

    num_samples = st.slider('Number of samples to generate', min_value=10, max_value=1000, step=10, value=100)

    if st.button('Generate Synthetic Data'):
        synthetic_data = ctgan_model.sample(num_samples)
        st.session_state.show_report = False
        st.session_state.plot_triggered = False
        st.session_state.synthetic_data = synthetic_data
        st.success(f'Generated {num_samples} synthetic samples!')

    synthetic_data = st.session_state.synthetic_data

    if synthetic_data is not None:
        st.dataframe(synthetic_data)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.download_button(
                "⬇️ Download CSV",
                synthetic_data.to_csv(index=False).encode("utf-8"),
                "synthetic_data.csv",
                "text/csv",
                key="download-csv"
            )

        with col2:
            plot_clicked = st.button("📊 Plot Distributions")

        with col3:
            report_clicked = st.button("🧪 Generate Report")

        if plot_clicked:
            st.session_state.show_report = False
            real_data = pd.read_csv("model/diabetes.csv")
            numeric_columns = real_data.select_dtypes(include=['number']).columns

            st.subheader('📈 Numeric Columns')
            for col in numeric_columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.kdeplot(real_data[col], label='Real', fill=True, color='blue', ax=ax, common_norm=False)
                sns.kdeplot(synthetic_data[col], label='Synthetic', fill=True, color='red', ax=ax, common_norm=False)
                ax.set_title(f'Distribution for {col}')
                ax.legend()
                st.pyplot(fig)

        if report_clicked:
            st.session_state.show_report = True

        if st.session_state.show_report:
            real_data = pd.read_csv("model/diabetes.csv")

            synthetic_data = synthetic_data.drop(columns=['source'])
            metadata = Metadata.detect_from_dataframe(data=real_data, table_name='diabetes table')

            columns = real_data.columns.tolist()
            col = st.selectbox("Select column", columns)

            st.pyplot(plot_distribution(real_data, synthetic_data, col))
            st.write(plot_column_shape(real_data, synthetic_data, metadata, col))

            diagnostic, quality = run_all_metrics(real_data, synthetic_data, metadata)

            st.subheader("Score")
            st.metric("Quality Score", f"{quality.get_score():.2%}")
            st.metric("Diagnostic Score", f"{diagnostic.get_score():.2%}")

            st.subheader("🔬 Diagnostic Details")
            st.write("data Structure")
            st.dataframe(diagnostic.get_details("Data Structure"))
            st.write("data Validity")
            st.dataframe(diagnostic.get_details("Data Validity"))
        
            st.subheader("📋 Quality Details: Column Shapes")
            st.dataframe(quality.get_details(property_name="Column Shapes"))

