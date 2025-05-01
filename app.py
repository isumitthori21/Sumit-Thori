import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sdv.single_table import CTGANSynthesizer

# Load saved model
@st.cache_resource
def load_model():
    model = CTGANSynthesizer.load('model/ctgan_model.pkl')
    return model

ctgan_model = load_model()

# Streamlit app
st.title('🧬 CTGAN Synthetic Data Generator')

# Sidebar options
st.sidebar.header('Settings')
num_samples = st.sidebar.slider('Number of samples to generate', min_value=10, max_value=1000, step=10, value=100)

# --- GENERATE DATA ---
if st.button('Generate Synthetic Data'):
    synthetic_data = ctgan_model.sample(num_samples)

    # Save it globally
    st.session_state.synthetic_data = synthetic_data

    st.success(f'Generated {num_samples} synthetic samples!')
    st.dataframe(synthetic_data)

    # Download button
    csv = synthetic_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Synthetic Data as CSV",
        csv,
        "synthetic_data.csv",
        "text/csv",
        key='download-csv'
    )

# --- CONDITIONAL DISPLAY OF PLOT BUTTON ---
if 'synthetic_data' in st.session_state:
    if st.button('📊 Plot Real vs Synthetic Distributions'):
        synthetic_data = st.session_state.synthetic_data
        real_data = pd.read_csv("model/diabetes.csv")

        # Numeric columns: KDE Plots
        numeric_columns = real_data.select_dtypes(include=['number']).columns

        st.subheader('📈 Numeric Columns')
        for col in numeric_columns:
            fig, ax = plt.subplots(figsize=(8, 4))

            sns.kdeplot(real_data[col], label='Real', fill=True, color='blue', ax=ax, common_norm=False)
            sns.kdeplot(synthetic_data[col], label='Synthetic', fill=True, color='red', ax=ax, common_norm=False)

            ax.set_title(f'Distribution for {col}')
            ax.legend()

            st.pyplot(fig)
        

