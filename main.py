import streamlit as st
import os
import sys

# Tell Python to look inside your 'app' folder for your code
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

st.title("🏥 MRI DICOM Viewer")
st.write("Welcome to your medical image viewer web application!")

try:
    # This automatically safely connects your app folder
    from app import viewer
    st.success("Successfully connected to your viewer logic!")
    
    # This automatically tries to run your viewer file
    if hasattr(viewer, 'main'):
        viewer.main()
    elif hasattr(viewer, 'run'):
        viewer.run()
except Exception as e:
    st.error(f"We connected the files, but hit a small hiccup: {e}")
    st.write("Don't worry, your AI guide will help you fix this error message next!")