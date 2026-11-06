"""Visual-check helpers for the construction notebooks.

* ``side_views``: static matplotlib x-z projections, z up, one panel per structure,
  with the title drawn in the same axes. Use it as the reference check, because
  a label cannot drift away from its structure.
* ``show_grid``: py3Dmol views. Each grid is one 3Dmol viewer (one WebGL context)
  holding several panels, and each panel is labelled inside its own canvas.
  Browsers keep only about 16 live WebGL contexts per page. A loop of ``show()``
  calls, one viewer per structure, goes over that limit, and the browser then
  draws structures under the wrong labels.
* ``show``: one py3Dmol viewer (HeteroBuilder's view_structure). Use it for a
  single structure only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from pymatgen.core import Element, Structure

COLORS = {"Cr": "#1f4fd1", "Te": "#d98c1a", "Al": "#9aa0a6", "O": "#d62728",
          "Ba": "#2ca02c", "Ti": "#8c564b"}


def _color(el: str) -> str:
    return COLORS.get(el, "#7f7f7f")


def show(structure: Structure, look: str = "b", supercell=(1, 1, 1)):
    from vdW_structures.viewer import view_structure

    s = structure.copy()
    if supercell != (1, 1, 1):
        s.make_supercell(supercell)
    cmap = {el.symbol: _color(el.symbol) for el in s.composition.elements}
    return view_structure(s, cmap, look=look)


def show_grid(structures: Sequence[Structure], labels: Sequence[str], cols: int = 4,
              supercell=(2, 2, 1), panel: int = 280, look: str = "b"):
    """Several structures in ONE py3Dmol viewer (viewergrid), each panel labelled in-canvas."""
    import py3Dmol

    n = len(structures)
    rows = int(np.ceil(n / cols))
    ncols = min(cols, n)
    view = py3Dmol.view(viewergrid=(rows, ncols), width=panel * ncols, height=int(panel * 1.6) * rows,
                        linked=False)
    for k, (structure, label) in enumerate(zip(structures, labels)):
        cell = (k // ncols, k % ncols)
        s = structure.copy()
        if supercell != (1, 1, 1):
            s.make_supercell(supercell)
        view.addModel(s.to(fmt="cif"), "cif", viewer=cell)
        for el in s.composition.elements:
            view.setStyle({"elem": el.symbol},
                          {"sphere": {"radius": 0.6 * float(Element(el.symbol).atomic_radius or 1.0),
                                      "color": _color(el.symbol)}}, viewer=cell)
        view.addUnitCell(viewer=cell)
        view.zoomTo(viewer=cell)
        if look.lower() == "b":
            view.rotate(-90, "x", viewer=cell)
        elif look.lower() == "a":
            view.rotate(-90, "y", viewer=cell)
        view.addLabel(label, {"useScreen": True, "position": {"x": 4, "y": 4, "z": 0},
                              "fontSize": 11, "fontColor": "black", "backgroundColor": "white",
                              "backgroundOpacity": 0.8, "inFront": True}, viewer=cell)
    return view.show()


def side_views(structures: Sequence[Structure], titles: Sequence[str], cols: int = 6,
               supercell=(2, 1, 1), frozen: bool = True):
    """Static x-z projections (z up). Open markers = frozen (selective dynamics F F F)."""
    import matplotlib.pyplot as plt

    n = len(structures)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 3.6 * rows), squeeze=False)
    for ax, structure, title in zip(axes.flat, structures, titles):
        s = structure.copy()
        sd = s.site_properties.get("selective_dynamics")
        mobile = np.array([bool(v[0]) for v in sd]) if (frozen and sd is not None) else np.ones(len(s), bool)
        reps = int(np.prod(supercell))
        if supercell != (1, 1, 1):
            s.make_supercell(supercell)
            mobile = np.repeat(mobile, reps)       # make_supercell keeps each site's images together
        xyz = s.cart_coords
        for el in sorted({sp.symbol for sp in s.species}, key=lambda e: -Element(e).atomic_radius):
            m = np.array([sp.symbol == el for sp in s.species])
            size = 25 * float(Element(el).atomic_radius or 1.0) ** 2
            col = _color(el)
            ax.scatter(xyz[m, 0], xyz[m, 2], s=size, edgecolors=col, linewidths=0.8,
                       facecolors=np.where(mobile[m], col, "none"), label=el)
        ax.set_title(title, fontsize=7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].set_ylabel("z (Å)")
    axes.flat[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    return fig


def interface_top_views(structures: Sequence[Structure], titles: Sequence[str], film_elements: set[str],
                        depth: float = 1.0, cols: int = 3):
    """x-y view of the interface: top substrate plane(s) within ``depth`` Å of the top substrate
    atom (filled) and the bottom film plane (open), with the supercell outline."""
    import matplotlib.pyplot as plt

    n = len(structures)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for ax, s, title in zip(axes.flat, structures, titles):
        xyz = s.cart_coords
        film = np.array([sp.symbol in film_elements for sp in s.species])
        z_sub_top = xyz[~film, 2].max()
        z_film_bot = xyz[film, 2].min()
        sub_sel = (~film) & (xyz[:, 2] > z_sub_top - depth)
        film_sel = film & (xyz[:, 2] < z_film_bot + depth)
        for sel, filled in ((sub_sel, True), (film_sel, False)):
            for el in sorted({s.species[i].symbol for i in np.where(sel)[0]}):
                m = sel & np.array([sp.symbol == el for sp in s.species])
                col = _color(el)
                ax.scatter(xyz[m, 0], xyz[m, 1], s=30 if filled else 45, c=col if filled else "none",
                           edgecolors=col, linewidths=1.0, label=f"{el} ({'substrate' if filled else 'film'})")
        a, b = s.lattice.matrix[0][:2], s.lattice.matrix[1][:2]
        outline = np.array([[0, 0], a, a + b, b, [0, 0]])
        ax.plot(outline[:, 0], outline[:, 1], "k-", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig
