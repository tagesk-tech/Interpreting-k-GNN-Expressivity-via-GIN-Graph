"""
gin_handlers - Dataset-specific GIN-Graph handlers.

Each dataset has its own handler with:
- Dataset-specific configuration (node labels, colors, class names)
- Visualization utilities tailored to the domain
- Factory function to get the appropriate handler
"""

from .base import DatasetHandler, get_handler

__all__ = ['DatasetHandler', 'get_handler']
