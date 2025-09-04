import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# App Configuration
st.set_page_config(
    page_title="🧠 AI Parkinson's Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .model-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🧠 AI-Powered Parkinson's Disease Detection</h1>
    <p>Advanced Machine Learning & Deep Learning Analysis System</p>
</div>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    """Load and return the Parkinson's dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sidebar - Data Info
st.sidebar.title("📊 Dataset Information")
data = load_data()

if data is not None:
    st.sidebar.success("✅ Dataset loaded successfully!")
    st.sidebar.write(f"**Total Samples:** {len(data)}")
    st.sidebar.write(f"**Features:** {len(data.columns)-1}")
    st.sidebar.write(f"**Parkinson's Cases:** {data['status'].sum()}")
    st.sidebar.write(f"**Healthy Cases:** {len(data) - data['status'].sum()}")
    
    # Data preprocessing
    X = data.drop(['name', 'status'], axis=1)
    y = data['status']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    st.error("Failed to load dataset. Please check your internet connection.")
    st.stop()

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Models", "📈 Data Analysis", "🔮 Prediction", "📊 Model Comparison"])

with tab1:
    st.header("🤖 Machine Learning & Deep Learning Models")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Select AI Models")
        use_knn = st.checkbox("K-Nearest Neighbors", value=True)
        use_nb = st.checkbox("Naive Bayes", value=True)
        use_svm = st.checkbox("Support Vector Machine", value=True)
        use_mlp = st.checkbox("Neural Network (MLP)", value=True)
        
        if TENSORFLOW_AVAILABLE:
            use_deep = st.checkbox("Deep Neural Network (TensorFlow)", value=True)
        else:
            st.warning("⚠️ TensorFlow not available. Deep Learning model disabled.")
            use_deep = False
    
    with col2:
        st.markdown("### ⚙️ Model Parameters")
        
        if use_knn:
            k_value = st.slider("KNN - Number of Neighbors", 3, 15, 5)
        
        if use_mlp:
            hidden_layers = st.selectbox(
                "Neural Network - Hidden Layer Size",
                [(100,), (128, 64), (128, 64, 32)],
                index=2
            )
    
    # Train and evaluate models
    if st.button("🚀 Train All Models", type="primary"):
        models = {}
        results = {}
        
        with st.spinner("Training AI models... 🧠"):
            progress_bar = st.progress(0)
            
            model_count = 0
            total_models = sum([use_knn, use_nb, use_svm, use_mlp, use_deep])
            
            # K-Nearest Neighbors
            if use_knn:
                knn = KNeighborsClassifier(n_neighbors=k_value)
                knn.fit(X_train, y_train)
                knn_pred = knn.predict(X_test)
                models['KNN'] = knn
                results['KNN'] = {
                    'accuracy': accuracy_score(y_test, knn_pred),
                    'predictions': knn_pred
                }
                model_count += 1
                progress_bar.progress(model_count / total_models)
            
            # Naive Bayes
            if use_nb:
                nb = GaussianNB()
                nb.fit(X_train, y_train)
                nb_pred = nb.predict(X_test)
                models['Naive Bayes'] = nb
                results['Naive Bayes'] = {
                    'accuracy': accuracy_score(y_test, nb_pred),
                    'predictions': nb_pred
                }
                model_count += 1
                progress_bar.progress(model_count / total_models)
            
            # SVM
            if use_svm:
                svm = SVC(kernel='rbf', probability=True)
                svm.fit(X_train, y_train)
                svm_pred = svm.predict(X_test)
                models['SVM'] = svm
                results['SVM'] = {
                    'accuracy': accuracy_score(y_test, svm_pred),
                    'predictions': svm_pred
                }
                model_count += 1
                progress_bar.progress(model_count / total_models)
            
            # Neural Network (MLP)
            if use_mlp:
                mlp = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                )
                mlp.fit(X_train, y_train)
                mlp_pred = mlp.predict(X_test)
                models['Neural Network'] = mlp
                results['Neural Network'] = {
                    'accuracy': accuracy_score(y_test, mlp_pred),
                    'predictions': mlp_pred
                }
                model_count += 1
                progress_bar.progress(model_count / total_models)
            
            # Deep Neural Network (TensorFlow)
            if use_deep and TENSORFLOW_AVAILABLE:
                # Build deep neural network
                deep_model = keras.Sequential([
                    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                    layers.Dropout(0.3),
                    layers.BatchNormalization(),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(0.3),
                    layers.BatchNormalization(),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
                
                # Compile model
                deep_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train model
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                )
                
                deep_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Make predictions
                deep_pred_proba = deep_model.predict(X_test, verbose=0)
                deep_pred = (deep_pred_proba > 0.5).astype(int).flatten()
                
                models['Deep Neural Network'] = deep_model
                results['Deep Neural Network'] = {
                    'accuracy': accuracy_score(y_test, deep_pred),
                    'predictions': deep_pred
                }
                model_count += 1
                progress_bar.progress(model_count / total_models)
        
        # Display Results
        st.success("🎉 All models trained successfully!")
        
        # Create results visualization
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in model_names]
        
        # Accuracy comparison chart
        fig = px.bar(
            x=model_names,
            y=accuracies,
            title="🏆 Model Accuracy Comparison",
            labels={'x': 'AI Models', 'y': 'Accuracy'},
            color=accuracies,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.markdown("### 📊 Detailed Results")
        
        cols = st.columns(len(results))
        for i, (model_name, result) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{model_name}</h4>
                    <h2>{result['accuracy']:.4f}</h2>
                    <p>Accuracy Score</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Confusion matrices
        st.markdown("### 🎯 Confusion Matrices")
        
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Store results in session state
        st.session_state['models'] = models
        st.session_state['results'] = results
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = X.columns.tolist()

with tab2:
    st.header("📈 Parkinson's Dataset Analysis")
    
    if data is not None:
        # Dataset overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Parkinson's Cases", data['status'].sum())
        with col3:
            st.metric("Healthy Cases", len(data) - data['status'].sum())
        
        # Class distribution
        fig = px.pie(
            values=data['status'].value_counts().values,
            names=['Healthy', 'Parkinson\'s'],
            title="Class Distribution",
            color_discrete_sequence=['#74b9ff', '#fd79a8']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation
        st.subheader("🔗 Feature Correlation Analysis")
        
        # Select top correlated features with status
        feature_cols = [col for col in data.columns if col not in ['name', 'status']]
        correlations = data[feature_cols + ['status']].corr()['status'].abs().sort_values(ascending=False)
        top_features = correlations.head(10).index[1:]  # Exclude 'status' itself
        
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = data[list(top_features) + ['status']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Top 10 Features Correlation with Parkinson\'s Status')
        st.pyplot(fig)
        
        # Feature distribution
        st.subheader("📊 Feature Distributions")
        
        selected_features = st.multiselect(
            "Select features to analyze:",
            feature_cols[:10],  # Show first 10 features
            default=feature_cols[:3]
        )
        
        if selected_features:
            fig, axes = plt.subplots(len(selected_features), 1, figsize=(12, 4*len(selected_features)))
            if len(selected_features) == 1:
                axes = [axes]
            
            for i, feature in enumerate(selected_features):
                for status in [0, 1]:
                    subset = data[data['status'] == status][feature]
                    axes[i].hist(subset, alpha=0.7, label=f'{"Parkinson\'s" if status else "Healthy"}', bins=30)
                
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
            
            plt.tight_layout()
            st.pyplot(fig)

with tab3:
    st.header("🔮 Make New Predictions")
    
    if 'models' in st.session_state:
        st.success("✅ Models are trained and ready for predictions!")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Random Sample", "Upload CSV"]
        )
        
        if input_method == "Manual Input":
            st.subheader("📝 Enter Voice Measurements")
            
            # Create input fields for key features
            col1, col2 = st.columns(2)
            
            with col1:
                mdvp_fo = st.number_input("MDVP:Fo(Hz) - Average vocal fundamental frequency", 
                                         value=154.0, min_value=50.0, max_value=300.0)
                mdvp_fhi = st.number_input("MDVP:Fhi(Hz) - Maximum vocal fundamental frequency",
                                          value=197.0, min_value=80.0, max_value=400.0)
                mdvp_flo = st.number_input("MDVP:Flo(Hz) - Minimum vocal fundamental frequency",
                                          value=116.0, min_value=40.0, max_value=200.0)
                mdvp_jitter = st.number_input("MDVP:Jitter(%) - Jitter percentage",
                                             value=0.00662, min_value=0.0, max_value=0.1)
                mdvp_rap = st.number_input("MDVP:RAP - Relative amplitude perturbation",
                                          value=0.00300, min_value=0.0, max_value=0.1)
            
            with col2:
                mdvp_ppq = st.number_input("MDVP:PPQ - Period perturbation quotient",
                                          value=0.00426, min_value=0.0, max_value=0.1)
                mdvp_shimmer = st.number_input("MDVP:Shimmer - Shimmer percentage",
                                              value=0.02971, min_value=0.0, max_value=0.2)
                mdvp_apq = st.number_input("MDVP:APQ - Amplitude perturbation quotient",
                                          value=0.02309, min_value=0.0, max_value=0.2)
                hnr = st.number_input("HNR - Harmonic-to-noise ratio",
                                     value=21.033, min_value=0.0, max_value=40.0)
                dfa = st.number_input("DFA - Detrended fluctuation analysis",
                                     value=0.641, min_value=0.0, max_value=1.0)
            
            # Use default values for remaining features (simplified for demo)
            feature_values = [
                mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter*1.2, 
                mdvp_rap, mdvp_ppq, 0.00567, mdvp_shimmer, mdvp_shimmer*0.8, 
                mdvp_apq, mdvp_apq*1.1, 0.027778, 0.029, 0.158, hnr, 0.197, 
                0.226, 2.168, 0.498, 0.527, dfa
            ]
            
            input_data = np.array(feature_values).reshape(1, -1)
        
        elif input_method == "Random Sample":
            if st.button("🎲 Generate Random Sample"):
                # Get a random sample from the test set
                random_idx = np.random.randint(0, len(X_test))
                input_data = X_test[random_idx:random_idx+1]
                actual_label = y_test.iloc[random_idx] if hasattr(y_test, 'iloc') else y_test[random_idx]
                st.info(f"Selected sample - Actual diagnosis: {'Parkinson\'s' if actual_label else 'Healthy'}")
        
        if 'input_data' in locals():
            # Make predictions with all models
            if st.button("🔍 Predict", type="primary"):
                predictions = {}
                probabilities = {}
                
                for model_name, model in st.session_state['models'].items():
                    if model_name == "Deep Neural Network" and TENSORFLOW_AVAILABLE:
                        pred_proba = model.predict(input_data, verbose=0)[0][0]
                        pred = 1 if pred_proba > 0.5 else 0
                        probabilities[model_name] = pred_proba
                    else:
                        pred = model.predict(input_data)[0]
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(input_data)[0][1]
                            probabilities[model_name] = pred_proba
                    
                    predictions[model_name] = pred
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                cols = st.columns(len(predictions))
                for i, (model_name, pred) in enumerate(predictions.items()):
                    with cols[i]:
                        color = "red" if pred else "green"
                        result = "Parkinson's" if pred else "Healthy"
                        confidence = probabilities.get(model_name, 0) * 100
                        
                        st.markdown(f"""
                        <div style="background-color: {'#ffebee' if pred else '#e8f5e8'}; 
                                   padding: 1rem; border-radius: 8px; 
                                   border-left: 4px solid {color};">
                            <h4>{model_name}</h4>
                            <h3 style="color: {color};">{result}</h3>
                            <p>Confidence: {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Ensemble prediction
                if len(predictions) > 1:
                    ensemble_pred = sum(predictions.values()) / len(predictions)
                    ensemble_result = "Parkinson's" if ensemble_pred > 0.5 else "Healthy"
                    
                    st.markdown("### 🏆 Ensemble Prediction")
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 1.5rem; 
                               border-radius: 10px; border: 2px solid #2196f3;">
                        <h3 style="color: #1976d2;">AI Ensemble Result: {ensemble_result}</h3>
                        <p>Consensus Score: {ensemble_pred:.2f}</p>
                        <p>Models Agreement: {sum(predictions.values())}/{len(predictions)} models predict Parkinson's</p>
                    </div>
                    """, unsafe_allow_html=True)

with tab4:
    st.header("📊 Model Performance Comparison")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Performance metrics
        st.subheader("🏆 Accuracy Comparison")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results.keys()]
        })
        
        # Interactive bar chart
        fig = px.bar(
            comparison_df, 
            x='Model', 
            y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        fig.update_traces(texttemplate='%{y:.4f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model]['accuracy']
        
        st.success(f"🏆 **Best Model:** {best_model} with accuracy of {best_accuracy:.4f}")
        
        # Model comparison table
        st.subheader("📋 Detailed Comparison")
        st.dataframe(comparison_df.style.highlight_max(subset=['Accuracy']))
    
    else:
        st.info("👆 Please train the models first in the 'AI Models' tab to see comparisons.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h3>🧠 AI Parkinson's Detection System</h3>
    <p>Powered by Machine Learning & Deep Learning | Built with Streamlit</p>
    <p>⚡ Advanced AI Models: KNN • Naive Bayes • SVM • Neural Networks • Deep Learning</p>
</div>
""", unsafe_allow_html=True)