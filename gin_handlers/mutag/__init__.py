"""
MUTAG dataset handler for GIN-Graph.

MUTAG contains 188 mutagenic compounds classified as mutagen or non-mutagen.
Node features are one-hot encoded atom types (C, N, O, F, I, Cl, Br).
"""

from .handler import MutagHandler

__all__ = ['MutagHandler']
