"""
FALT (Forced Alignment Toolkit)
A toolkit that uses forced alignment TextGrids to pool transformer hidden state activations into linguistically relevant segments.
"""

# Version of the falt package
__version__ = "0.1.0"

from falt.generate_activations import (
    extract_activations,
    extract_and_save_processed_activations,
)

__all__ = ["extract_activations", "extract_and_save_processed_activations"]
