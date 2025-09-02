import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Parkinson's Disease Prediction System")
st.markdown("""
This application uses machine learning to predict Parkinson's disease based on voice measurements.
The model analyzes various vocal features to determine the likelihood of Parkinson's disease.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose an option:",
    ("Home", "Single Prediction", "Batch Prediction", "Model Information")
)

# Load sample data (you can replace this with your actual dataset)
@st.cache_data
def load_sample_data():
    """Create sample data structure for Parkinson's dataset"""
    # These are typical features used in Parkinson's prediction
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ]
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 200
    data = []
    
    for i in range(n_samples):
        # Create realistic-looking sample data
        sample = {
            'MDVP:Fo(Hz)': np.random.normal(154, 41),
            'MDVP:Fhi(Hz)': np.random.normal(197, 91),
            'MDVP:Flo(Hz)': np.random.normal(116, 43),
            'MDVP:Jitter(%)': np.random.normal(0.006, 0.005),
            'MDVP:Jitter(Abs)': np.random.normal(0.000044, 0.000035),
            'MDVP:RAP': np.random.normal(0.003, 0.002),
            'MDVP:PPQ': np.random.normal(0.003, 0.002),
            'Jitter:DDP': np.random.normal(0.009, 0.007),
            'MDVP:Shimmer': np.random.normal(0.029, 0.018),
            'MDVP:Shimmer(dB)': np.random.normal(0.282, 0.194),
            'Shimmer:APQ3': np.random.normal(0.015, 0.010),
            'Shimmer:APQ5': np.random.normal(0.017, 0.013),
            'MDVP:APQ': np.random.normal(0.024, 0.017),
            'Shimmer:DDA': np.random.normal(0.046, 0.030),
            'NHR': np.random.normal(0.024, 0.040),
            'HNR': np.random.normal(21.679, 4.708),
            'RPDE': np.random.normal(0.498, 0.103),
            'DFA': np.random.normal(0.718, 0.055),
            'spread1': np.random.normal(-5.684, 1.090),
            'spread2': np.random.normal(0.226, 0.083),
            'D2': np.random.normal(2.381, 0.382),
            'PPE': np.random.normal(0.206, 0.090),
            'status': np.random.choice([0, 1], p=[0.25, 0.75])  # 75% positive cases
        }
        data.append(sample)
    
    return pd.DataFrame(data)

# Train model function
@st.cache_resource
def train_model():
    """Train the Parkinson's prediction model"""
    df = load_sample_data()
    
    # Separate features and target
    X = df.drop('status', axis=1)
    y = df['status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X.columns.tolist()

# Load the trained model
model, scaler, accuracy, feature_names = train_model()

# Home page
if option == "Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to Parkinson's Disease Prediction System")
        st.write("""
        This system uses advanced machine learning algorithms to predict Parkinson's disease 
        based on voice measurement data. The model analyzes various vocal features including:
        
        - **Frequency Measures**: Fundamental frequency variations
        - **Jitter Measures**: Frequency variation measures
        - **Shimmer Measures**: Amplitude variation measures
        - **Noise Measures**: Harmonics-to-noise ratios
        - **Nonlinear Measures**: Complexity and entropy measures
        """)
        
        st.info(f"**Current Model Accuracy: {accuracy:.2%}**")
        
    with col2:
        st.image("https://via.placeholder.com/300x200/4CAF50/FFFFFF?text=Parkinson's+Detection", 
                caption="AI-Powered Medical Diagnosis")

# Single Prediction page
elif option == "Single Prediction":
    st.header("🔍 Single Patient Prediction")
    st.write("Enter the vocal measurements to predict Parkinson's disease:")
    
    # Create input fields for all features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Frequency Measures")
        mdvp_fo = st.number_input("MDVP:Fo(Hz) - Average vocal fundamental frequency", 
                                 value=154.23, min_value=80.0, max_value=300.0, step=0.01)
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz) - Maximum vocal fundamental frequency", 
                                  value=197.08, min_value=100.0, max_value=400.0, step=0.01)
        mdvp_flo = st.number_input("MDVP:Flo(Hz) - Minimum vocal fundamental frequency", 
                                  value=116.68, min_value=50.0, max_value=250.0, step=0.01)
        
        st.subheader("Jitter Measures")
        mdvp_jitter_percent = st.number_input("MDVP:Jitter(%) - Frequency variation", 
                                             value=0.00632, min_value=0.0, max_value=0.1, step=0.00001, format="%.5f")
        mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs) - Absolute jitter", 
                                         value=0.0000454, min_value=0.0, max_value=0.001, step=0.0000001, format="%.7f")
        mdvp_rap = st.number_input("MDVP:RAP - Relative amplitude perturbation", 
                                  value=0.00349, min_value=0.0, max_value=0.05, step=0.00001, format="%.5f")
        mdvp_ppq = st.number_input("MDVP:PPQ - Period perturbation quotient", 
                                  value=0.00345, min_value=0.0, max_value=0.05, step=0.00001, format="%.5f")
        jitter_ddp = st.number_input("Jitter:DDP - Differential perturbation", 
                                    value=0.01047, min_value=0.0, max_value=0.1, step=0.00001, format="%.5f")
        
        st.subheader("Shimmer Measures")
        mdvp_shimmer = st.number_input("MDVP:Shimmer - Amplitude variation", 
                                      value=0.02971, min_value=0.0, max_value=0.2, step=0.00001, format="%.5f")
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB) - Shimmer in dB", 
                                         value=0.282, min_value=0.0, max_value=2.0, step=0.001)
        shimmer_apq3 = st.number_input("Shimmer:APQ3 - Amplitude perturbation quotient", 
                                      value=0.01438, min_value=0.0, max_value=0.1, step=0.00001, format="%.5f")
    
    with col2:
        shimmer_apq5 = st.number_input("Shimmer:APQ5 - 5-point amplitude perturbation", 
                                      value=0.01767, min_value=0.0, max_value=0.1, step=0.00001, format="%.5f")
        mdvp_apq = st.number_input("MDVP:APQ - Amplitude perturbation quotient", 
                                  value=0.02446, min_value=0.0, max_value=0.15, step=0.00001, format="%.5f")
        shimmer_dda = st.number_input("Shimmer:DDA - Differential amplitude", 
                                     value=0.04314, min_value=0.0, max_value=0.3, step=0.00001, format="%.5f")
        
        st.subheader("Noise Measures")
        nhr = st.number_input("NHR - Noise-to-harmonics ratio", 
                             value=0.02449, min_value=0.0, max_value=0.5, step=0.00001, format="%.5f")
        hnr = st.number_input("HNR - Harmonics-to-noise ratio", 
                             value=21.679, min_value=0.0, max_value=50.0, step=0.001)
        
        st.subheader("Nonlinear Measures")
        rpde = st.number_input("RPDE - Recurrence period density entropy", 
                              value=0.498, min_value=0.0, max_value=1.0, step=0.001)
        dfa = st.number_input("DFA - Detrended fluctuation analysis", 
                             value=0.718, min_value=0.0, max_value=1.0, step=0.001)
        spread1 = st.number_input("Spread1 - Nonlinear measure", 
                                 value=-5.684, min_value=-10.0, max_value=0.0, step=0.001)
        spread2 = st.number_input("Spread2 - Nonlinear measure", 
                                 value=0.226, min_value=0.0, max_value=1.0, step=0.001)
        d2 = st.number_input("D2 - Correlation dimension", 
                            value=2.381, min_value=0.0, max_value=5.0, step=0.001)
        ppe = st.number_input("PPE - Pitch period entropy", 
                             value=0.206, min_value=0.0, max_value=1.0, step=0.001)
    
    # Prediction button
    if st.button("🔮 Predict Parkinson's Disease", type="primary"):
        # Prepare input data
        input_data = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
                               mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
                               shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
                               rpde, dfa, spread1, spread2, d2, ppe]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error(f"**Prediction: Positive for Parkinson's Disease**")
            else:
                st.success(f"**Prediction: Negative for Parkinson's Disease**")
        
        with col2:
            st.metric("Confidence (Negative)", f"{probability[0]:.2%}")
        
        with col3:
            st.metric("Confidence (Positive)", f"{probability[1]:.2%}")
        
        # Additional information
        st.info("""
        **Important Note**: This prediction is based on a machine learning model and should not be used 
        as a substitute for professional medical diagnosis. Please consult with a healthcare professional 
        for proper medical evaluation.
        """)

# Batch Prediction page
elif option == "Batch Prediction":
    st.header("📁 Batch Prediction")
    st.write("Upload a CSV file with multiple patient data for batch prediction.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.write("**Data Preview:**")
            st.dataframe(df.head())
            
            # Check if all required columns are present
            if all(col in df.columns for col in feature_names):
                if st.button("🚀 Run Batch Prediction"):
                    # Prepare data
                    X = df[feature_names]
                    X_scaled = scaler.transform(X)
                    
                    # Make predictions
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)
                    
                    # Add results to dataframe
                    results_df = df.copy()
                    results_df['Prediction'] = ['Positive' if p == 1 else 'Negative' for p in predictions]
                    results_df['Confidence_Negative'] = probabilities[:, 0]
                    results_df['Confidence_Positive'] = probabilities[:, 1]
                    
                    # Display results
                    st.success(f"Processed {len(results_df)} patients successfully!")
                    
                    # Summary statistics
                    positive_cases = sum(predictions)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Patients", len(results_df))
                    with col2:
                        st.metric("Positive Cases", positive_cases)
                    with col3:
                        st.metric("Negative Cases", len(results_df) - positive_cases)
                    
                    # Show results
                    st.write("**Prediction Results:**")
                    st.dataframe(results_df[['Prediction', 'Confidence_Negative', 'Confidence_Positive']])
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="parkinsons_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f"Missing required columns. Expected columns: {feature_names}")
                st.write("Your file columns:", list(df.columns))
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Sample file download
    st.subheader("📥 Download Sample Format")
    sample_data = load_sample_data().drop('status', axis=1).head(5)
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV Format",
        data=csv,
        file_name="sample_parkinsons_data.csv",
        mime="text/csv"
    )

# Model Information page
else:
    st.header("ℹ️ Model Information")
    
    st.subheader("📋 About the Dataset")
    st.write("""
    This model is trained on voice measurement data to detect Parkinson's disease. 
    The dataset contains various vocal features extracted from voice recordings.
    
    **Features Used:**
    """)
    
    features_info = {
        "Frequency Measures": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
        "Jitter Measures": ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP"],
        "Shimmer Measures": ["MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"],
        "Noise Measures": ["NHR", "HNR"],
        "Nonlinear Measures": ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
    }
    
    for category, features in features_info.items():
        with st.expander(f"{category} ({len(features)} features)"):
            for feature in features:
                st.write(f"• {feature}")
    
    st.subheader("🤖 Model Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Algorithm:** Random Forest Classifier")
        st.write("**Number of Features:** 22")
        st.write(f"**Model Accuracy:** {accuracy:.2%}")
        st.write("**Cross-validation:** Stratified")
    
    with col2:
        st.write("**Preprocessing:** StandardScaler")
        st.write("**Training Split:** 80%")
        st.write("**Test Split:** 20%")
        st.write("**Random State:** 42")
    
    st.subheader("⚠️ Important Disclaimers")
    st.warning("""
    **Medical Disclaimer:**
    - This tool is for educational and research purposes only
    - It should NOT be used as a substitute for professional medical diagnosis
    - Always consult with qualified healthcare professionals for medical advice
    - The model's predictions may have false positives and false negatives
    - This is a demonstration model and may not reflect real-world accuracy
    """)
    
    st.subheader("📊 Model Performance")
    st.write(f"""
    The current model achieves an accuracy of **{accuracy:.2%}** on the test dataset.
    However, in real-world applications, additional validation and clinical trials 
    would be necessary before any medical use.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧠 Parkinson's Disease Prediction System | Built with Streamlit & Scikit-learn</p>
    <p><em>For educational purposes only. Not for medical diagnosis.</em></p>
</div>
""", unsafe_allow_html=True)