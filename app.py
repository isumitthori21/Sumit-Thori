import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from Components.metrics import load_data, load_metadata, plot_distribution, plot_column_shape, run_all_metrics
from Components.SyntheticDatagenerator import Section as FirstSection
from Components.customModelTrainer import Section as SecondSection

if "page" not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar options
with st.sidebar:
    st.header("Navigation")
    if st.button("Generate synthetic Data"):
        st.session_state.page = "🧬 Synthetic Diabetes Data Generator"
    if st.button("Train custom model"):
        st.session_state.page = "🛠️ Train Custom Synthetic Data Model"
    if st.button("Evaluation & Metrics"):
        st.session_state.page = "Evaluation & Metrics"

st.title(st.session_state.page)
if st.session_state.page == "🧬 Synthetic Diabetes Data Generator":
    FirstSection()
if st.session_state.page == "🛠️ Train Custom Synthetic Data Model":
    SecondSection()
        

