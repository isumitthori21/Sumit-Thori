import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
from Components.metrics import plot_distribution, plot_column_shape, plot_distributions, generate_report

def Section():
    @st.cache_resource
    def load_ctgan_model():
        return CTGANSynthesizer.load('model/custom_ctgan_model.pkl')

    def load_tvae_model():
        return TVAESynthesizer.load('model/tvae_model.pkl')

    ctgan_model = load_ctgan_model()
    tvae_model = load_tvae_model()

    # --- Session state init ---
    if 'synthetic_data' not in st.session_state:
        st.session_state.synthetic_data = None
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False
    
    
    st.write("")
    st.write("")    
    st.write("")    
    st.write("")    
        
    
    num_samples = st.slider('Number of samples to generate', min_value=10, max_value=1000, step=10, value=100)
    st.write("")    
    st.write("")
    choice = st.radio(
    "Select an option:",
    ["CTGan Model", "TVAE Model"]
)
    st.write("")    
    st.write("")
    if st.button('Generate Synthetic Data'):
        if(choice == "CTGan Model"):
            synthetic_data = ctgan_model.sample(num_samples)
        else:
            synthetic_data = tvae_model.sample(num_samples)
        st.session_state.synthetic_data = None
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
            plot_distributions(real_data,synthetic_data)

        if report_clicked:
            st.session_state.show_report = True

        if st.session_state.show_report:
            real_data = pd.read_csv("model/diabetes.csv")
 
            synthetic_data = synthetic_data.drop(columns=['source'],errors='ignore')
            metadata = Metadata.detect_from_dataframe(data=real_data, table_name='diabetes table')

            columns = real_data.columns.tolist()
            col = st.selectbox("Select column", columns)

            st.pyplot(plot_distribution(real_data, synthetic_data, col))
            st.write(plot_column_shape(real_data, synthetic_data, metadata, col))

            generate_report(real_data,synthetic_data,metadata)

