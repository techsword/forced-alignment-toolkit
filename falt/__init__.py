"""
FALT (Forced Alignment Toolkit)
A toolkit that uses forced alignment TextGrids to pool transformer hidden state activations into linguistically relevant segments.
"""

# Version of the falt package
__version__ = "0.1.0"

# Import main components to make them available at package level
# Add these as your package grows
from falt.generate_activations import *  # uncomment and modify once you have core modules