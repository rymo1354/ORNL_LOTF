"""Visual-check helpers for the construction notebooks (py3Dmol via HeteroBuilder)."""
from __future__ import annotations

from pymatgen.core import Structure

COLORS = {"Cr": "#1f4fd1", "Te": "#d98c1a", "Al": "#9aa0a6", "O": "#d62728",
          "Ba": "#2ca02c", "Ti": "#8c564b"}


def show(structure: Structure, look: str = "b", supercell=(1, 1, 1)):
    from vdW_structures.viewer import view_structure

    s = structure.copy()
    if supercell != (1, 1, 1):
        s.make_supercell(supercell)
    cmap = {el.symbol: COLORS.get(el.symbol, "#7f7f7f") for el in s.composition.elements}
    return view_structure(s, cmap, look=look)
