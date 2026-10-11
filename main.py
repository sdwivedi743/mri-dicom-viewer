import streamlit as st
import sys
import os

st.title("🏥 MRI DICOM Viewer")
st.write("Welcome to your medical image viewer web application!")

# Smart search: Let's find where 'viewer.py' is hiding
found_viewer = False
possible_paths = [
    os.path.abspath(os.path.dirname(__file__)),
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "app"),
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "MRI Dicom viewer", "app")
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.append(path)

try:
    # Try importing directly if it's right next to main.py or in system path
    import viewer
    st.success("Successfully found and connected your viewer.py file!")
    found_viewer = True
except ImportError:
    try:
        # Try importing if it's inside an app folder
        from app import viewer
        st.success("Successfully found and connected your app/viewer.py file!")
        found_viewer = True
    except ImportError:
        pass

if found_viewer:
    try:
        # Try to run the viewer
        if hasattr(viewer, 'main'):
            viewer.main()
        elif hasattr(viewer, 'run'):
            viewer.run()
        else:
            st.warning("Connected! But viewer.py doesn't have a 'main' or 'run' function to start. Let's look inside next.")
    except Exception as e:
        st.error(f"Error running viewer: {e}")
else:
    st.error("We still couldn't find your 'viewer.py' file on GitHub.")
    st.write("Current folder contents seen by the website:")
    st.write(os.listdir('.'))