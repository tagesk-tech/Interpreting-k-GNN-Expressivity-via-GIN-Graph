"""
PROTEINS dataset handler for GIN-Graph.

PROTEINS contains protein structures classified by enzyme vs non-enzyme.
Node features represent amino acid properties or secondary structure.
"""

from .handler import ProteinsHandler

__all__ = ['ProteinsHandler']
