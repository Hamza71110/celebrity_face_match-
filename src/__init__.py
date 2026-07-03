"""Celebrity Face Match - shared source package.

Holds configuration, data loading, face utilities and inference code that are
reused by the training scripts (``training/``) and the serving API
(``deployment/api.py``). The original Streamlit app (``app.py``) does NOT
import from here, so it keeps working exactly as before.
"""
