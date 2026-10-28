"""Atomic Environment Explorer: service layer over the CN12 analysis package.

The Streamlit app (``02_code/atomenv_app.py``) and any Python caller use the
same functions in :mod:`atomenv.api`. UI code lives only in the app file and
:mod:`atomenv.viz`; parsing, analysis and serialization live here.
"""

__version__ = "1.0.0"

from .api import analyze_cif, analyze_site, feature_record, load_paths, site_table

__all__ = ["analyze_cif", "analyze_site", "feature_record", "load_paths", "site_table", "__version__"]
