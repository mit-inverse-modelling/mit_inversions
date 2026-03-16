# -*- coding: utf-8 -*-
"""Emissions prior distribution and spatial aggregation."""
from .distribution import generate_emissions_distribution
from .aggregation import get_country_emissions, get_region_emissions

__all__ = [
    "generate_emissions_distribution",
    "get_country_emissions",
    "get_region_emissions",
]
