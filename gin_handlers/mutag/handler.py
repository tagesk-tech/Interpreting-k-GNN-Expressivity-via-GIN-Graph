"""
gin_handlers/mutag/handler.py
MUTAG-specific GIN handler for molecular graphs.

MUTAG Dataset:
- 188 mutagenic compounds
- Binary classification: Mutagen (0) vs Non-Mutagen (1)
- 7 atom types: C, N, O, F, I, Cl, Br
- Graphs represent molecular structures
"""

from typing import Dict
import matplotlib.pyplot as plt
from ..base import DatasetHandler


class MutagHandler(DatasetHandler):
    """Handler for MUTAG molecular dataset."""

    @property
    def name(self) -> str:
        return "MUTAG"

    @property
    def node_labels(self) -> Dict[int, str]:
        """Atom type mapping for MUTAG."""
        return {
            0: 'C',   # Carbon
            1: 'N',   # Nitrogen
            2: 'O',   # Oxygen
            3: 'F',   # Fluorine
            4: 'I',   # Iodine
            5: 'Cl',  # Chlorine
            6: 'Br'   # Bromine
        }

    @property
    def node_colors(self) -> Dict[str, str]:
        """Colors for atom types (chemistry-inspired)."""
        return {
            'C': '#FFA500',    # Orange (Carbon)
            'N': '#00BFFF',    # Deep sky blue (Nitrogen)
            'O': '#FF0000',    # Red (Oxygen)
            'F': '#32CD32',    # Lime green (Fluorine)
            'I': '#800080',    # Purple (Iodine)
            'Cl': '#90EE90',   # Light green (Chlorine)
            'Br': '#8B4513',   # Saddle brown (Bromine)
            '?': '#808080'     # Gray (unknown)
        }

    @property
    def class_names(self) -> Dict[int, str]:
        """Class names for MUTAG."""
        return {
            0: 'Mutagen',
            1: 'Non-Mutagen'
        }

    def plot_legend(self, save_path: str = None):
        """Create a legend showing atom types and colors for MUTAG."""
        fig, ax = plt.subplots(figsize=(10, 2))

        atoms = list(self.node_labels.values())
        colors = [self.node_colors[a] for a in atoms]

        for i, (atom, color) in enumerate(zip(atoms, colors)):
            circle = plt.Circle((i + 0.5, 0.5), 0.3, color=color, ec='black')
            ax.add_patch(circle)
            ax.text(i + 0.5, 0.5, atom, ha='center', va='center',
                    fontsize=12, fontweight='bold')

        ax.set_xlim(0, len(atoms))
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Atom Types in MUTAG', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Legend saved to: {save_path}")

        return fig
