""" Metrics enumeration for system-calculated metrics.
"""
from enum import Enum


class Metrics(str, Enum):
    """
    Metrics calculated by the system.
    """
    IMMEDIATE = 'immediate_impact'
    LEAVE_OUT = 'leave_out_impact'
