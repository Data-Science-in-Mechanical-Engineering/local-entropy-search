from enum import Enum

# Enumeration to make computation mode selection more readable
class ComputationMode(Enum):
    ZEROTH_ORDER = 0    # Only zeroth order observations provided
    FIRST_ORDER = 1     # Zeroth and first order observations provided
    SECOND_ORDER = 2    # Zeroth, first and second order observations provided