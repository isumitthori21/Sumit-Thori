import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
confusion_matrix, ConfusionMatrixDisplay,
classification_report, roc_curve, auc
)
from sdv.metadata import Metadata


def plot_single(data):
    if 'source' in data.columns:
        data = data.drop(columns='source')
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Outcome', data=data, ax=ax1)
    st.pyplot(fig1)  # Show first plot

    fig2 = sns.pairplot(data, hue='Outcome')
    st.pyplot(fig2.figure)

def heatmap_matrix(data):
    if 'source' in data.columns:
        data = data.drop(columns='source')
    fig3, ax3 = plt.subplots()
    sns.heatmap(data.corr(), annot=True, vmin=-1.0, vmax=1.0, center=0, ax=ax3)
    st.pyplot(fig3)  # Show heatmap

def plot_distributions(data):
    sns.countplot(x='Outcome', data=data)
    plt.show()

    sns.pairplot(data, hue='Outcome')
    plt.show()

# Detailed breakdown
def getQualityReportDetails(report):
    qualityDetails={}
    qualityDetails['column_shape'] = report.get_details('Column Shapes')
    qualityDetails['column_pair_trends'] = report.get_details('Column Pair Trends')
    return qualityDetails
    
    
def sdv_quality_report(metadata,real_data,synthetic_data):
    
    metadata_dict = metadata.to_dict()
    metadata_dict = metadata_dict['tables']['diabetes']
    report = QualityReport()
    # Drop 'source' column from synthetic_data if it exists
    if 'source' in synthetic_data.columns:
        synthetic_data = synthetic_data.drop(columns='source')
    Quality_report = report.generate(real_data,synthetic_data,metadata_dict)

    # View overall score
    
    st.metric("Quality Score", f"{report.get_score():.2%}")
    st.write('Details')
    details = getQualityReportDetails(report)
    st.write('Column Shape : ', details['column_shape'])
    st.write('Column Pair Trends : ', details['column_pair_trends'])
    

def getDiagnosticreportDetails(diagnostic):
    
    diagnostic_report = {}
    diagnostic_report['validity'] = diagnostic.get_details('Data Validity')
    diagnostic_report['structure'] = diagnostic.get_details('Data Structure')
    return diagnostic_report

def sdv_diagnostic_report(real_data,synthetic_data,metadata):
    metadata_dict = metadata.to_dict()
    metadata_dict = metadata_dict['tables']['diabetes']
    diagnostic = DiagnosticReport()
    diagnostic.generate(real_data, synthetic_data, metadata_dict)

    # Summary
    
    st.metric("Diagnostic Score", f"{diagnostic.get_score():.2%}")
    st.write('Details')
    details = getDiagnosticreportDetails(diagnostic)
    st.write('Data Validity : ', details['validity'])
    st.write('Data Structure : ', details['structure'])
    
def kde_plot(real_data,synthetic_data):
    numeric_columns = real_data.select_dtypes(include=['number']).columns

    # Drop 'source' column from synthetic_data if it exists
    if 'source' in synthetic_data.columns:
        synthetic_data = synthetic_data.drop(columns='source')

    # Layout: 3 plots per row (adjust as needed)
    cols_per_row = 3
    total_plots = len(numeric_columns)
    rows = math.ceil(total_plots / cols_per_row)

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 4, rows * 3))
    axes = axes.flatten()  # Flatten in case of multiple rows

    for i, col in enumerate(numeric_columns):
        ax = axes[i]
        sns.kdeplot(real_data[col], label='Real', fill=True, color='blue', ax=ax)
        sns.kdeplot(synthetic_data[col], label='Synthetic', fill=True, color='red', ax=ax)
        ax.set_title(f'Distribution: {col}')
        ax.legend()

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def heatmap_comparision(data,synthetic_data):
    # Correlation matrices
    if 'source' in data.columns:
        data = data.drop(columns='source')

    real_corr = data.corr()
    synthetic_corr = synthetic_data.corr()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", annot=False)
    axes[0].set_title('Real Data Correlation')

    sns.heatmap(synthetic_corr, ax=axes[1], cmap="coolwarm", annot=False)
    axes[1].set_title('Synthetic Data Correlation')

    plt.show()
    # Mean Absolute Error between correlation matrices
    correlation_diff = np.abs(real_corr - synthetic_corr)
    mean_diff = correlation_diff.values[np.triu_indices_from(correlation_diff, k=1)].mean()

    print(f"Mean Absolute Correlation Difference: {mean_diff:.4f}")

def random_xboost_lr_test(real_data,synthetic_data):
    
    import matplotlib.pyplot as plt
    import seaborn as sns

   



    # 🎯 Define target column
    target_col = 'Outcome'  # Change this to your actual target column name

    # 🧪 Split features and labels
    X_synth = synthetic_data
    y_synth = synthetic_data[target_col]

    X_real = real_data
    y_real = real_data[target_col]

    # 🔁 Train/Test Models
    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}

    # Prepare 3 rows: Confusion Matrix, Class Dist, ROC Curve
    fig, axes = plt.subplots(nrows=3, ncols=len(models), figsize=(6 * len(models), 15))
    fig.suptitle('📊 Evaluation Summary by Model', fontsize=18)

    for col_idx, (name, model) in enumerate(models.items()):
        # Train
        model.fit(X_synth, y_synth)
        y_pred = model.predict(X_real)
        y_prob = model.predict_proba(X_real)[:, 1] if hasattr(model, 'predict_proba') else None

        print(f"\n📊 Model: {name}")
        print(classification_report(y_real, y_pred))

        # Row 0: Confusion Matrix
        cm = confusion_matrix(y_real, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=axes[0, col_idx])
        axes[0, col_idx].set_title(f'{name} - Confusion Matrix')

        # Row 1: Class Distribution
        sns.histplot(y_real, label='Actual', kde=False, color='blue', stat='density', ax=axes[1, col_idx])
        sns.histplot(y_pred, label='Predicted', kde=False, color='red', stat='density', ax=axes[1, col_idx])
        axes[1, col_idx].set_title(f'{name} - Class Distribution')
        axes[1, col_idx].legend()

        # Row 2: ROC Curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_real, y_prob)
            roc_auc = auc(fpr, tpr)
            axes[2, col_idx].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            axes[2, col_idx].plot([0, 1], [0, 1], 'k--')
            axes[2, col_idx].set_xlabel("FPR")
            axes[2, col_idx].set_ylabel("TPR")
            axes[2, col_idx].set_title(f'{name} - ROC Curve')
            axes[2, col_idx].legend()

        results[name] = {
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def Section():

    # Upload section
    real_file = st.file_uploader("Upload Real Dataset (CSV)", type=["csv"], key="real_data")
    synth_file = st.file_uploader("Upload Synthetic Dataset (CSV)", type=["csv"], key="synth_data")

    if real_file and synth_file:
        real_data = pd.read_csv(real_file)
        synthetic_data = pd.read_csv(synth_file)
        if 'source' in synthetic_data.columns:
            synthetic_data = synthetic_data.drop(columns='source')
        metadata = Metadata.detect_from_dataframe(
                   data=real_data,
                   table_name='diabetes')
        
        st.subheader("📈 KDE Distributions")
        kde_plot(real_data, synthetic_data)
        st.pyplot(plt.gcf())

        st.subheader("📌 Correlation Heatmap Comparison")
        heatmap_comparision(real_data, synthetic_data)
        st.pyplot(plt.gcf())

        st.subheader("🧪 Diagnostic Report")
        try:
            sdv_diagnostic_report(real_data, synthetic_data, metadata)
        except Exception as e:
            st.error(f"Error in diagnostic report: {e}")

        st.subheader("📋 Quality Report")
        try:
            sdv_quality_report(metadata, real_data, synthetic_data)
        except Exception as e:
            st.error(f"Error in quality report: {e}")

        # st.subheader("🧠 Classifier Performance")
    #     try:
    #         random_xboost_lr_test(real_data, synthetic_data)
    #         st.pyplot(plt.gcf())
    #     except Exception as e:
    #         st.error(f"Error in model evaluation: {e}")
    else:
        st.info("Please upload both real and synthetic CSV files to proceed.")
