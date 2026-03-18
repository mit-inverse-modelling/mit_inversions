# -*- coding: utf-8 -*-
"""0.1 degree global grid constants (used by emissions distribution)."""
import numpy as np

TARGET_RES_DEG = 0.1
TARGET_LON = np.arange(-180.0, 180.0, TARGET_RES_DEG)
TARGET_LAT = np.arange(-90.0, 90.0, TARGET_RES_DEG)
