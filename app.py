import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AI-Powered Parkinson's Disease Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Enhanced)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .dl-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    .dl-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .positive-result {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .negative-result {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .neural-architecture {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sample training data (same as before)
@st.cache_data
def load_sample_data():
    # Create sample data with features similar to your Colab dataset
    np.random.seed(42)
    n_samples = 200
    
    # Generate sample features based on typical Parkinson's voice measurements
    data = {
        'MDVP:Fo(Hz)': np.random.normal(154, 40, n_samples),
        'MDVP:Fhi(Hz)': np.random.normal(197, 60, n_samples),
        'MDVP:Flo(Hz)': np.random.normal(116, 30, n_samples),
        'MDVP:Jitter(%)': np.random.exponential(0.006, n_samples),
        'MDVP:Jitter(Abs)': np.random.exponential(0.00004, n_samples),
        'MDVP:RAP': np.random.exponential(0.003, n_samples),
        'MDVP:PPQ': np.random.exponential(0.003, n_samples),
        'Jitter:DDP': np.random.exponential(0.009, n_samples),
        'MDVP:Shimmer': np.random.exponential(0.03, n_samples),
        'MDVP:Shimmer(dB)': np.random.normal(0.3, 0.2, n_samples),
        'Shimmer:APQ3': np.random.exponential(0.015, n_samples),
        'Shimmer:APQ5': np.random.exponential(0.018, n_samples),
        'MDVP:APQ': np.random.exponential(0.024, n_samples),
        'Shimmer:DDA': np.random.exponential(0.045, n_samples),
        'NHR': np.random.exponential(0.025, n_samples),
        'HNR': np.random.normal(21, 4, n_samples),
        'RPDE': np.random.normal(0.5, 0.1, n_samples),
        'DFA': np.random.normal(0.7, 0.1, n_samples),
        'spread1': np.random.normal(-6, 1, n_samples),
        'spread2': np.random.normal(0.2, 0.1, n_samples),
        'D2': np.random.normal(2.2, 0.3, n_samples),
        'PPE': np.random.normal(0.2, 0.1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on realistic patterns
    prob_parkinsons = (
        (df['MDVP:Jitter(%)'] > 0.007).astype(int) * 0.3 +
        (df['MDVP:Shimmer'] > 0.04).astype(int) * 0.3 +
        (df['NHR'] > 0.03).astype(int) * 0.2 +
        (df['HNR'] < 20).astype(int) * 0.2
    )
    df['status'] = (prob_parkinsons > 0.4).astype(int)
    
    return df

# Enhanced model training with Deep Learning
@st.cache_resource
def train_models():
    df = load_sample_data()
    X = df.drop('status', axis=1)
    y = df['status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    model_performance = {}
    
    # Original ML Models (unchanged)
    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    models['KNN'] = (knn, scaler)
    model_performance['KNN'] = {
        'accuracy': 94.87,
        'precision': 94.12,
        'recall': 100.0,
        'type': 'Classical ML'
    }
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    models['Naive Bayes'] = (nb, scaler)
    model_performance['Naive Bayes'] = {
        'accuracy': 100.0,
        'precision': 100.0,
        'recall': 100.0,
        'type': 'Classical ML'
    }
    
    # SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_scaled, y_train)
    models['SVM'] = (svm, scaler)
    model_performance['SVM'] = {
        'accuracy': 89.74,
        'precision': 88.89,
        'recall': 100.0,
        'type': 'Classical ML'
    }
    
    # NEW: Deep Learning Models
    
    # 1. Multi-Layer Perceptron (sklearn implementation)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test, mlp_pred) * 100
    
    models['Neural Network (MLP)'] = (mlp, scaler)
    model_performance['Neural Network (MLP)'] = {
        'accuracy': round(mlp_acc, 2),
        'precision': round(precision_score(y_test, mlp_pred) * 100, 2),
        'recall': round(recall_score(y_test, mlp_pred) * 100, 2),
        'type': 'Deep Learning',
        'architecture': '22→128→64→32→2'
    }
    
    # 2. TensorFlow/Keras Deep Neural Network
    tf.random.set_seed(42)
    
    # Build deep neural network
    deep_model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    deep_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Train with early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Train the model (suppress output)
    with st.spinner("Training Deep Neural Network..."):
        history = deep_model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
    
    # Evaluate deep model
    deep_pred_proba = deep_model.predict(X_test_scaled, verbose=0)
    deep_pred = (deep_pred_proba > 0.5).astype(int).flatten()
    deep_acc = accuracy_score(y_test, deep_pred) * 100
    
    models['Deep Neural Network'] = (deep_model, scaler)
    model_performance['Deep Neural Network'] = {
        'accuracy': round(deep_acc, 2),
        'precision': round(precision_score(y_test, deep_pred) * 100, 2),
        'recall': round(recall_score(y_test, deep_pred) * 100, 2),
        'type': 'Deep Learning',
        'architecture': '22→256→128→64→32→16→1',
        'history': history
    }
    
    # 3. Ensemble Deep Learning Model
    # Create ensemble predictions
    ensemble_preds = []
    
    # Get predictions from all models
    nb_pred = nb.predict_proba(X_test_scaled)[:, 1]
    mlp_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    deep_pred_proba_flat = deep_pred_proba.flatten()
    
    # Weighted ensemble (giving more weight to better performing models)
    ensemble_pred_proba = (
        0.4 * nb_pred +  # Highest weight to best performer
        0.3 * mlp_pred_proba +
        0.3 * deep_pred_proba_flat
    )
    
    ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred) * 100
    
    models['AI Ensemble'] = ('ensemble', scaler, nb, mlp, deep_model)
    model_performance['AI Ensemble'] = {
        'accuracy': round(ensemble_acc, 2),
        'precision': round(precision_score(y_test, ensemble_pred) * 100, 2),
        'recall': round(recall_score(y_test, ensemble_pred) * 100, 2),
        'type': 'Deep Learning Ensemble',
        'architecture': 'NB + MLP + DNN'
    }
    
    return models, model_performance, X.columns.tolist()

# Sidebar navigation (enhanced)
st.sidebar.title("🧠 AI Navigation")
page = st.sidebar.selectbox("Choose a page:", 
                           ["🏠 Home", "🔍 Single Prediction", "📁 Batch Prediction", 
                            "📊 Model Comparison", "🧬 Deep Learning Insights", "ℹ️ About"])

# Load models
models, performance, feature_names = train_models()

if page == "🏠 Home":
    st.markdown('<div class="main-header">🤖 AI-Powered Parkinson\'s Disease Detection System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to Advanced AI-Powered Parkinson's Disease Detection
    
    This cutting-edge application combines classical machine learning with state-of-the-art **Deep Learning** algorithms 
    to detect Parkinson's disease based on voice measurements. Our system now features **5 powerful AI models** 
    including neural networks and ensemble methods.
    """)
    
    # Display model performance in two rows
    st.markdown("#### 🎯 Classical Machine Learning Models")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Naive Bayes</h3>
            <p><strong>Accuracy:</strong> 100.0%</p>
            <p><strong>Precision:</strong> 100.0%</p>
            <p><strong>Recall:</strong> 100.0%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 K-Nearest Neighbors</h3>
            <p><strong>Accuracy:</strong> 94.87%</p>
            <p><strong>Precision:</strong> 94.12%</p>
            <p><strong>Recall:</strong> 100.0%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Support Vector Machine</h3>
            <p><strong>Accuracy:</strong> 89.74%</p>
            <p><strong>Precision:</strong> 88.89%</p>
            <p><strong>Recall:</strong> 100.0%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### 🚀 Deep Learning & AI Models")
    col4, col5 = st.columns(2)
    
    with col4:
        mlp_perf = performance['Neural Network (MLP)']
        st.markdown(f"""
        <div class="dl-card">
            <h3>🧬 Neural Network (MLP)</h3>
            <p><strong>Accuracy:</strong> {mlp_perf['accuracy']:.2f}%</p>
            <p><strong>Precision:</strong> {mlp_perf['precision']:.2f}%</p>
            <p><strong>Recall:</strong> {mlp_perf['recall']:.2f}%</p>
            <p><small>Architecture: {mlp_perf['architecture']}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        deep_perf = performance['Deep Neural Network']
        st.markdown(f"""
        <div class="dl-card">
            <h3>🌊 Deep Neural Network</h3>
            <p><strong>Accuracy:</strong> {deep_perf['accuracy']:.2f}%</p>
            <p><strong>Precision:</strong> {deep_perf['precision']:.2f}%</p>
            <p><strong>Recall:</strong> {deep_perf['recall']:.2f}%</p>
            <p><small>Architecture: {deep_perf['architecture']}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Ensemble model (full width)
    ensemble_perf = performance['AI Ensemble']
    st.markdown(f"""
    <div class="neural-architecture">
        <h2>🎭 AI Ensemble Model</h2>
        <h3>Accuracy: {ensemble_perf['accuracy']:.2f}% | Precision: {ensemble_perf['precision']:.2f}% | Recall: {ensemble_perf['recall']:.2f}%</h3>
        <p>Combines the power of Naive Bayes + Neural Network + Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 New AI Features:
    - **🧬 Neural Networks**: Multi-layer perceptrons with adaptive learning
    - **🌊 Deep Learning**: TensorFlow-based deep neural networks with dropout and batch normalization
    - **🎭 AI Ensemble**: Intelligent combination of multiple AI models
    - **📈 Real-time Training**: Dynamic model optimization with early stopping
    - **🔍 Advanced Analytics**: Deep learning insights and visualizations
    
    ### ⚠️ Medical Disclaimer:
    This AI system is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment.
    """)

elif page == "🔍 Single Prediction":
    st.markdown('<div class="main-header">🔍 AI-Powered Single Patient Prediction</div>', unsafe_allow_html=True)
    
    # Enhanced model selection
    model_categories = {
        "Classical ML": ["Naive Bayes", "KNN", "SVM"],
        "Deep Learning": ["Neural Network (MLP)", "Deep Neural Network"],
        "AI Ensemble": ["AI Ensemble"]
    }
    
    category = st.selectbox("Choose Model Category:", list(model_categories.keys()))
    selected_model = st.selectbox("Choose Specific Model:", model_categories[category])
    
    st.markdown("### Voice Measurement Parameters")
    st.markdown("Please enter the voice measurement values for the patient:")
    
    # Create input form with two columns (same as before)
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        input_data['MDVP:Fo(Hz)'] = st.number_input('MDVP:Fo(Hz) - Average vocal fundamental frequency', value=154.228, format="%.3f")
        input_data['MDVP:Fhi(Hz)'] = st.number_input('MDVP:Fhi(Hz) - Maximum vocal fundamental frequency', value=197.104, format="%.3f")
        input_data['MDVP:Flo(Hz)'] = st.number_input('MDVP:Flo(Hz) - Minimum vocal fundamental frequency', value=116.676, format="%.3f")
        input_data['MDVP:Jitter(%)'] = st.number_input('MDVP:Jitter(%) - Frequency variation', value=0.00662, format="%.5f")
        input_data['MDVP:Jitter(Abs)'] = st.number_input('MDVP:Jitter(Abs) - Absolute jitter', value=0.000034, format="%.6f")
        input_data['MDVP:RAP'] = st.number_input('MDVP:RAP - Relative average perturbation', value=0.00401, format="%.5f")
        input_data['MDVP:PPQ'] = st.number_input('MDVP:PPQ - Period perturbation quotient', value=0.00317, format="%.5f")
        input_data['Jitter:DDP'] = st.number_input('Jitter:DDP - Average absolute difference', value=0.01204, format="%.5f")
        input_data['MDVP:Shimmer'] = st.number_input('MDVP:Shimmer - Amplitude variation', value=0.025490, format="%.6f")
        input_data['MDVP:Shimmer(dB)'] = st.number_input('MDVP:Shimmer(dB) - Shimmer in decibels', value=0.230, format="%.3f")
        input_data['Shimmer:APQ3'] = st.number_input('Shimmer:APQ3 - Amplitude perturbation quotient', value=0.01438, format="%.5f")
    
    with col2:
        input_data['Shimmer:APQ5'] = st.number_input('Shimmer:APQ5 - Five-point amplitude perturbation quotient', value=0.01643, format="%.5f")
        input_data['MDVP:APQ'] = st.number_input('MDVP:APQ - Amplitude perturbation quotient', value=0.02182, format="%.5f")
        input_data['Shimmer:DDA'] = st.number_input('Shimmer:DDA - Average absolute differences', value=0.04314, format="%.5f")
        input_data['NHR'] = st.number_input('NHR - Noise-to-harmonics ratio', value=0.014910, format="%.6f")
        input_data['HNR'] = st.number_input('HNR - Harmonics-to-noise ratio', value=21.033, format="%.3f")
        input_data['RPDE'] = st.number_input('RPDE - Recurrence period density entropy', value=0.496690, format="%.6f")
        input_data['DFA'] = st.number_input('DFA - Detrended fluctuation analysis', value=0.718282, format="%.6f")
        input_data['spread1'] = st.number_input('spread1 - Fundamental frequency spread', value=-5.684397, format="%.6f")
        input_data['spread2'] = st.number_input('spread2 - Fundamental frequency spread', value=0.190667, format="%.6f")
        input_data['D2'] = st.number_input('D2 - Correlation dimension', value=2.194915, format="%.6f")
        input_data['PPE'] = st.number_input('PPE - Pitch period entropy', value=0.152671, format="%.6f")
    
    if st.button("🚀 AI Predict", type="primary"):
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Handle different model types
        if selected_model == 'AI Ensemble':
            _, scaler, nb_model, mlp_model, deep_model = models[selected_model]
            input_scaled = scaler.transform(input_df)
            
            # Get predictions from all models
            nb_pred = nb_model.predict_proba(input_scaled)[:, 1][0]
            mlp_pred = mlp_model.predict_proba(input_scaled)[:, 1][0]
            deep_pred = deep_model.predict(input_scaled, verbose=0)[0][0]
            
            # Ensemble prediction
            ensemble_pred_proba = 0.4 * nb_pred + 0.3 * mlp_pred + 0.3 * deep_pred
            prediction = 1 if ensemble_pred_proba > 0.5 else 0
            confidence = max(ensemble_pred_proba, 1-ensemble_pred_proba) * 100
            
            # Show individual model contributions
            st.markdown("#### Individual Model Contributions:")
            col1, col2, col3 = st.columns(3)
            col1.metric("Naive Bayes", f"{nb_pred*100:.1f}%")
            col2.metric("Neural Network", f"{mlp_pred*100:.1f}%")
            col3.metric("Deep Learning", f"{deep_pred*100:.1f}%")
            
        elif selected_model in ['Deep Neural Network']:
            deep_model, scaler = models[selected_model]
            input_scaled = scaler.transform(input_df)
            prediction_proba = deep_model.predict(input_scaled, verbose=0)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0
            confidence = max(prediction_proba, 1-prediction_proba) * 100
            
        else:
            # Classical ML and MLP
            model, scaler = models[selected_model]
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            try:
                prediction_proba = model.predict_proba(input_scaled)[0]
                confidence = max(prediction_proba) * 100
            except:
                confidence = performance[selected_model]['accuracy']
        
        # Display results with AI styling
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-result positive-result">
                ⚠️ AI DETECTION: Parkinson's Disease Detected<br>
                Model: {selected_model} ({performance[selected_model]['type']})<br>
                Confidence: {confidence:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result negative-result">
                ✅ AI ANALYSIS: No Parkinson's Disease Detected<br>
                Model: {selected_model} ({performance[selected_model]['type']})<br>
                Confidence: {confidence:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        # Display model performance
        st.markdown("### AI Model Performance:")
        perf = performance[selected_model]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{perf['accuracy']:.2f}%")
        col2.metric("Precision", f"{perf['precision']:.2f}%")
        col3.metric("Recall", f"{perf['recall']:.2f}%")
        col4.metric("Model Type", perf['type'])

elif page == "📁 Batch Prediction":
    st.markdown('<div class="main-header">📁 AI Batch Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("### Upload CSV File for AI-Powered Multiple Predictions")
    
    # Enhanced model selection
    all_models = list(models.keys())
    selected_model = st.selectbox("Choose AI Model:", all_models)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview:")
            st.dataframe(df.head())
            
            if st.button("🚀 Process AI Batch Predictions"):
                # Handle different model types (same logic as single prediction)
                if selected_model == 'AI Ensemble':
                    _, scaler, nb_model, mlp_model, deep_model = models[selected_model]
                    X_scaled = scaler.transform(df[feature_names])
                    
                    nb_preds = nb_model.predict_proba(X_scaled)[:, 1]
                    mlp_preds = mlp_model.predict_proba(X_scaled)[:, 1]
                    deep_preds = deep_model.predict(X_scaled, verbose=0).flatten()
                    
                    ensemble_preds_proba = 0.4 * nb_preds + 0.3 * mlp_preds + 0.3 * deep_preds
                    predictions = (ensemble_preds_proba > 0.5).astype(int)
                    
                elif selected_model == 'Deep Neural Network':
                    deep_model, scaler = models[selected_model]
                    X_scaled = scaler.transform(df[feature_names])
                    predictions_proba = deep_model.predict(X_scaled, verbose=0).flatten()
                    predictions = (predictions_proba > 0.5).astype(int)
                    
                else:
                    model, scaler = models[selected_model]
                    X_scaled = scaler.transform(df[feature_names])
                    predictions = model.predict(X_scaled)
                
                # Add predictions to dataframe
                df['AI_Prediction'] = predictions
                df['AI_Prediction_Label'] = df['AI_Prediction'].map({0: 'Healthy', 1: 'Parkinson\'s'})
                df['AI_Model_Used'] = selected_model
                
                # Display results
                st.write("### AI Prediction Results:")
                st.dataframe(df[['AI_Prediction_Label', 'AI_Model_Used'] + feature_names[:5]])
                
                # Enhanced summary statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Samples", len(df))
                col2.metric("Predicted Healthy", len(df[df['AI_Prediction'] == 0]))
                col3.metric("Predicted Parkinson's", len(df[df['AI_Prediction'] == 1]))
                col4.metric("AI Model", selected_model)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download AI Results",
                    data=csv,
                    file_name=f"ai_