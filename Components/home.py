import streamlit as st
def Section():
    st.subheader("Welcome to the Synthetic Diabetes Data Generation App")
    st.markdown("""
    This app helps you generate, analyze, and evaluate synthetic diabetes datasets using advanced generative models.
    """)

    st.markdown('---')

    st.subheader("About the app")

    col1, col2, col3 = st.columns(3)

    with col1:
            st.subheader("📊 Data Generation")
            st.info("Generate synthetic diabetes data and analyze its characteristics. The model was trained using a combination of PIMA India Dataset and other diabetes dataset fom Kaggle")

    with col2:
            st.subheader("🛠️ Custom Model")
            st.info("Train a custom generative model on your dataset to generate synthetic data.")

    with col3:
            st.subheader("📈 Evaluation")
            st.info("Evaluate synthetic data quality using metrics like KDE plots, heatmaps, and SDMetrics.")
    st.markdown('---')
        # Individual technology cards
    st.subheader("🧠 Terms")

    col4, col5 = st.columns(2)
    col6, col7 = st.columns(2)
    col8 = st.columns(1)
    with col4:
        st.subheader("CTGAN Model")
        st.info("""
        Conditional GAN for tabular data generation.
        Helps in learning the distribution of tabular data.
        """)

    with col5:
        st.subheader("TVAE Model")
        st.info("""
        Variational Autoencoder for tables.
        Generates synthetic data by learning complex distributions.
        """)

    with col6:
        st.subheader("KDE Plots")
        st.info("""
        Kernel Density Estimation.
        Useful for comparing the distributions of synthetic and real data.
        """)

    with col7:
        st.subheader("GAN Model")
        st.info("""
        Generative Adversarial Networks.
        Powerful for generating synthetic data by training a generator and discriminator.
        """)
        
    with col8[0]:
        st.subheader("Ollama - Tinyllama")
        st.info("""
        This online tool, developed by Ollavision, specializes in making digital marketing more efficient for businesses of all sizes.
        TinyLlama is a compact AI model developed by Ollama community. It offers efficient language processing capabilities in a smaller package, making it suitable for applications with limited computational resources. The model is designed for a variety of tasks, including conversational AI and real-time text generation, and supports deployment on edge devices.
        """)

