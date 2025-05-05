import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot

def load_data(rd,sd):
    real_data = rd
    synthetic_data = sd
    return real_data, synthetic_data

def load_metadata():
    from sdv.metadata import SingleTableMetadata
    return SingleTableMetadata.load("model/metadata.json")

def plot_distribution(real_data, synthetic_data, column):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(real_data[column], kde=True, label="Real", color="blue", stat="density", linewidth=2, ax=ax)
    sns.histplot(synthetic_data[column], kde=True, label="Synthetic", color="red", stat="density", linewidth=2, ax=ax)
    ax.set_title(f"Distribution for {column}")
    ax.legend()
    return fig

def plot_column_shape(real_data, synthetic_data, metadata, column):
    return get_column_plot(real_data, synthetic_data, metadata, column_name=column)


def run_all_metrics(real_data, synthetic_data, metadata):
    diagnostic = run_diagnostic(real_data, synthetic_data, metadata)
    quality = evaluate_quality(real_data, synthetic_data, metadata)
    return diagnostic, quality

def ReportDashboard():
    st.title("📊 Synthetic Data Evaluation Dashboard")

def plot_distributions(real_data,synthetic_data):
    numeric_columns = real_data.select_dtypes(include=['number']).columns
    st.subheader('📈 Numeric Columns')
    for col in numeric_columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(real_data[col], label='Real', fill=True, color='blue', ax=ax, common_norm=False)
        sns.kdeplot(synthetic_data[col], label='Synthetic', fill=True, color='red', ax=ax, common_norm=False)
        ax.set_title(f'Distribution for {col}')
        ax.legend()
        st.pyplot(fig)
        
def generate_report(real_data,synthetic_data,metadata):
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
