"""
NumPy compatibility module for handling numpy 2.0 changes.
This module provides backward compatibility for deprecated numpy attributes.
"""

import numpy as np
import warnings

def setup_numpy_compatibility():
    """
    Setup numpy compatibility for deprecated attributes in numpy 2.0+
    """
    # Suppress numpy deprecation warnings for compatibility
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    
    # Add compatibility for np.float_ which was removed in numpy 2.0
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    
    # Add compatibility for np.int_ which was removed in numpy 2.0  
    if not hasattr(np, 'int_'):
        np.int_ = np.int64
        
    # Add compatibility for np.complex_ which was removed in numpy 2.0
    if not hasattr(np, 'complex_'):
        np.complex_ = np.complex128
        
    # Add compatibility for np.bool_ if needed
    if not hasattr(np, 'bool_'):
        np.bool_ = np.bool

def patch_prophet_compatibility():
    """
    Patch Prophet's imports to work with numpy 2.0+
    """
    try:
        import prophet.forecaster
        # If prophet is already imported, we need to patch it
        if hasattr(prophet.forecaster, 'np'):
            if not hasattr(prophet.forecaster.np, 'float_'):
                prophet.forecaster.np.float_ = np.float64
    except ImportError:
        pass  # Prophet not installed or not importable yet
    
    try:
        import prophet.plot
        if hasattr(prophet.plot, 'np'):
            if not hasattr(prophet.plot.np, 'float_'):
                prophet.plot.np.float_ = np.float64
    except ImportError:
        pass
        
def apply_all_patches():
    """
    Apply all numpy compatibility patches
    """
    setup_numpy_compatibility()
    patch_prophet_compatibility()
