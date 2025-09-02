import streamlit as st 
import pandas as pd 
import numpy as np 
 
st.title("My Web App from Colab") 
st.write("Hello! This is my converted Colab project.") 
 
# Simple example to test: 
data = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)}) 
 
st.write("Sample Data:") 
st.dataframe(data.head()) 
 
st.write("Chart:") 
st.line_chart(data) 
