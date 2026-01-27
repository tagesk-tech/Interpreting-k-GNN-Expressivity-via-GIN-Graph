"""
DD (D&D) dataset handler for GIN-Graph.

DD contains protein structures classified as enzyme or non-enzyme.
Node features represent amino acid types (89 types).
"""

from .handler import DDHandler

__all__ = ['DDHandler']
