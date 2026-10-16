"""
Maxima tension principal sobre la configuracion deformada de la presa.

Compatible con CST, LST, Q4 y Q8 y con estados planos de tension o
Deformacion. Se usa la formulacion original del elemento para evaluar sigma;
la deformada es una representacion amplificada, NO un nuevo analisis.

En deformacion plana la tension fuera del plano NO es nula:
    sigma_z = nu * (sigma_x + sigma_y).
La maxima principal del tensor 3D es max(sigma_1_del_plano_XY, sigma_z).

En tension plana sigma_z = 0, por lo que la maxima principal 3D tambien
puede diferir de sigma_1_del_plano_XY si las principales XY son negativas.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

from source.output.stressPlot import (
    _reference_triangulation,
    sample_stress_field,
)


def _natural_edges(n_nodes):
    """Parametrizacion de aristas compatible con shape_functions()."""
    if n_nodes in (3, 6):
        return (
            lambda s: (s, 0.0),
            lambda s: (1.0 - s, s),
            lambda s: (0.0, 1.0 - s),
        )
    if n_nodes in (4, 8):
        return (
            lambda s: (-1.0 + 2.0*s, -1.0),
            lambda s: (1.0, -1.0 + 2.0*s),
            lambda s: (1.0 - 2.0*s, 1.0),
            lambda s: (-1.0, 1.0 - 2.0*s),
        )
    raise ValueError(f"Tipo de elemento no soportado: {n_nodes} nodos")


def _corner_ids(element):
    """IDs de vertices (no de nodos medios), en orden antihorario."""
    ids = element.node_global_ids
    if len(ids) in (3, 6):
        return list(ids[:3])
    if len(ids) in (4, 8):
        return list(ids[:4])
    raise ValueError("Solo se admiten CST, LST, Q4 y Q8")


def sample_principal_stress_deformed(
    mesh, elements, u, scale_factor=500.0, subdivisions=10,
    principal_mode="3d"
):
    """
    Muestrea la tension principal maxima y mapea cada punto a la deformada.

    principal_mode:
      - '3d': mayor principal de [sigma_x, sigma_y, sigma_z, tau_xy],
              apropiada para 'maxima tension principal' en deformacion plana.
      - 'xy': mayor principal solo del plano XY; igual a stressPlot.py.

    Cada elemento mantiene su propia triangulacion grafica: NO hay
    promediado de tensiones entre elementos.
    """
    if not np.isfinite(scale_factor) or scale_factor < 0.0:
        raise ValueError("scale_factor debe ser finito y >= 0")
    if principal_mode not in ("3d", "xy"):
        raise ValueError("principal_mode debe ser '3d' o 'xy'")

    u = np.asarray(u, dtype=float).reshape(-1)
    fields = sample_stress_field(mesh, elements, u, subdivisions=subdivisions)
    result = []

    for element, field in zip(elements, fields):
        n_nodes = len(element.node_global_ids)
        points, _ = _reference_triangulation(n_nodes, subdivisions)

        # Nodos locales en el mismo orden que las funciones de forma.
        ue = u[element.global_dofs()].reshape(n_nodes, 2)

        # u(xi,eta) = sum_a N_a(xi,eta) u_a, consistente con FEM.
        uv = np.asarray([
            np.asarray(element.shape_functions(xi, eta), dtype=float) @ ue
            for xi, eta in points
        ])

        # x_def = x + alpha*ux, y_def = y + alpha*uy.
        x_def = field["x"] + scale_factor * uv[:, 0]
        y_def = field["y"] + scale_factor * uv[:, 1]

        material = element.material
        if material.plane_state == "strain":
            # epsilon_z=0 => sigma_z=nu*(sigma_x+sigma_y).
            sigma_z = material.nu * (field["sigma_x"] + field["sigma_y"])
        elif material.plane_state == "stress":
            sigma_z = np.zeros_like(field["sigma_1"])
        else:
            raise ValueError(f"Estado plano desconocido: {material.plane_state}")

        sigma_1_xy = field["sigma_1"]
        sigma_max = np.maximum(sigma_1_xy, sigma_z)
        sigma_plot = sigma_max if principal_mode == "3d" else sigma_1_xy

        if not (np.all(np.isfinite(x_def)) and np.all(np.isfinite(y_def))
                and np.all(np.isfinite(sigma_plot))):
            raise ValueError(f"Campo no finito en elemento {element.global_id}")

        result.append({
            "element_id": element.global_id,
            "triangles": field["triangles"],
            "x": field["x"],
            "y": field["y"],
            "x_deformed": x_def,
            "y_deformed": y_def,
            "ux": uv[:, 0],
            "uy": uv[:, 1],
            "sigma_1_xy": sigma_1_xy,
            "sigma_z": sigma_z,
            "sigma_max_3d": sigma_max,
            "sigma_plot": sigma_plot,
        })

    return result


def _stress_norm(fields):
    """Escala de colores unica en toda la presa, en MPa."""
    values = np.concatenate([field["sigma_plot"] for field in fields]) / 1e6
    vmin = float(values.min())
    vmax = float(values.max())

    if np.isclose(vmin, vmax, rtol=0.0, atol=1e-14):
        extra = max(0.01 * abs(vmin), 1e-6)
        vmin -= extra
        vmax += extra

    if vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def plot_principal_stress_deformed(
    mesh,
    elements,
    u,
    scale_factor=500.0,
    subdivisions=12,
    principal_mode="3d",
    title="Presa de gravedad",
    save_filename=None,
    show=True,
):
    """
    Dibuja sigma principal maxima [MPa] sobre la geometria deformada.

    El desplazamiento se amplifica solo para visualizacion. Los valores de
    tension no se amplifican ni se calculan sobre la geometria perturbada.
    """
    fields = sample_principal_stress_deformed(
        mesh, elements, u, scale_factor=scale_factor,
        subdivisions=subdivisions, principal_mode=principal_mode
    )

    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    norm = _stress_norm(fields)
    cmap = "coolwarm"

    for field in fields:
        # Cada elemento se grafica por separado para conservar saltos de sigma.
        triangulation = mtri.Triangulation(
            field["x_deformed"], field["y_deformed"],
            triangles=field["triangles"]
        )
        ax.tripcolor(
            triangulation, field["sigma_plot"] / 1e6,
            shading="gouraud", cmap=cmap, norm=norm, rasterized=True
        )

    # Identificar contorno exterior por cantidad de apariciones de cada arista.
    edge_counts = Counter()
    for element in elements:
        corners = _corner_ids(element)
        for k in range(len(corners)):
            edge_counts[tuple(sorted((corners[k], corners[(k+1) % len(corners)])))] += 1

    s_values = np.linspace(0.0, 1.0, 16)
    for element in elements:
        corners = _corner_ids(element)
        node_ids = element.node_global_ids
        n_nodes = len(node_ids)
        coords = np.asarray([
            [mesh.node(nid).x, mesh.node(nid).y] for nid in node_ids
        ], dtype=float)
        ue = np.asarray(u, dtype=float).reshape(-1)[element.global_dofs()].reshape(n_nodes, 2)

        for k, edge in enumerate(_natural_edges(n_nodes)):
            points = np.asarray([
                element.shape_functions(*edge(s)) for s in s_values
            ], dtype=float)
            original = points @ coords
            deformed = original + scale_factor * (points @ ue)

            # Todas las aristas en la configuracion deformada.
            ax.plot(
                deformed[:, 0], deformed[:, 1],
                color="black", alpha=0.38, lw=0.55, zorder=3
            )

            key = tuple(sorted((corners[k], corners[(k + 1) % len(corners)])))
            if edge_counts[key] == 1:
                # Contorno exterior original: referencia visual.
                ax.plot(
                    original[:, 0], original[:, 1],
                    color="dimgray", lw=1.3, ls="--", zorder=4
                )

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.86
    )
    cbar.set_label("Tension principal [MPa]")

    label = ("Maxima tension principal 3D" if principal_mode == "3d"
             else "Mayor tension principal del plano XY")
    ax.set_title(
        f"{title}\n{label} sobre la configuracion deformada (escala x{scale_factor:g})",
        fontsize=12
    )
    ax.set_xlabel("X deformada amplificada [m]")
    ax.set_ylabel("Y deformada amplificada [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15, linestyle=":")
    ax.legend(handles=[
        Line2D([0], [0], color="black", lw=0.8, label="Malla deformada"),
        Line2D([0], [0], color="dimgray", ls="--", lw=1.3,
               label="Contorno exterior original"),
    ], loc="upper right", fontsize=9)

    if save_filename:
        output_dir = Path(__file__).resolve().parents[2] / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / save_filename
        fig.savefig(filepath, dpi=220, bbox_inches="tight")
        print(f"Tension principal sobre deformada guardada en: {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
