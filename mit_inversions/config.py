"""Configuration module for MIT inversions package.

Handles path configuration with user-editable defaults.
"""
import os
from pathlib import Path


def get_data_path(path=None):
    """Get the data path with configurable override.
    
    Priority order:
    1. Explicit path argument
    2. MITINV_DATA_PATH environment variable
    3. Default: ~/agage/
    
    Args:
        path: Optional explicit path to use
        
    Returns:
        Path object pointing to the data directory
    """
    if path:
        return Path(path)
    
    if env_path := os.getenv('MITINV_DATA_PATH'):
        return Path(env_path)
    
    return Path.home() / 'agage'


# Default data path - users can override by setting MITINV_DATA_PATH
data_path = get_data_path()
