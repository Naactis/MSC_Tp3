"""
Mapas de tensiones del metodo de elementos finitos.

Compatible con CST (3 nodos), LST (6 nodos), Q4 (4 nodos) y Q8 (8 nodos).
La geometria y las tensiones se calculan con las funciones de forma y la
matriz B reales de cada elemento.

IMPORTANTE:
    - Las tensiones se interpolan SOLO para dibujar dentro de cada elemento.
    - No se promedian ni se suavizan entre elementos vecinos.
    - El muestreo grafico no modifica el calculo FEM.
    - Los extremos del grafico no son necesariamente los extremos obtenidos
      en los puntos de Gauss del informe numerico.
    - Las tensiones principales corresponden al plano XY.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

# Reutilizamos las aristas isoparametricas que ya usa el grafico de U.
from source.output.displacementPlot import plot_element_boundaries


# =============================================================
# 1. SUBDIVISION DEL ELEMENTO DE REFERENCIA
# =============================================================

def _reference_triangulation(n_nodes, subdivisions):
    """
    Construye puntos naturales y subtriangulos SOLO para dibujar.

    Triangulos CST/LST: xi >= 0, eta >= 0, xi + eta <= 1.
    Cuadrilateros Q4/Q8: -1 <= xi, eta <= 1.

    Retorna:
        points: coordenadas naturales (n_points, 2).
        triangles: indices locales de subtriangulos (n_triangles, 3).
    """
    points = []
    triangles = []
    idx = {}

    if n_nodes in (3, 6):
        for i in range(subdivisions + 1):
            for j in range(subdivisions + 1 - i):
                idx[(i, j)] = len(points)
                points.append((i / subdivisions, j / subdivisions))

        for i in range(subdivisions):
            for j in range(subdivisions - i):
                triangles.append((
                    idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]
                ))
                if i + j < subdivisions - 1:
                    triangles.append((
                        idx[(i + 1, j)],
                        idx[(i + 1, j + 1)],
                        idx[(i, j + 1)]
                    ))

    elif n_nodes in (4, 8):
        for i in range(subdivisions + 1):
            for j in range(subdivisions + 1):
                idx[(i, j)] = len(points)
                points.append((
                    -1.0 + 2.0 * i / subdivisions,
                    -1.0 + 2.0 * j / subdivisions
                ))

        for i in range(subdivisions):
            for j in range(subdivisions):
                triangles.append((
                    idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)]
                ))
                triangles.append((
                    idx[(i, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]
                ))
    else:
        raise ValueError(f"Elemento no soportado: {n_nodes} nodos")

    return np.asarray(points, dtype=float), np.asarray(triangles, dtype=int)


# =============================================================
# 2. MUESTREO DE TENSIONES POR ELEMENTO
# =============================================================

def sample_stress_field(mesh, elements, u, subdivisions=10):
    """
    Evalua epsilon y sigma en puntos del interior de cada elemento.

    Devuelve una LISTA de campos, uno por elemento. No se fusionan
    nodos graficos compartidos, porque las tensiones FEM pueden ser
    discontinuas en las interfaces entre elementos.

    Cada campo almacena x, y, triangles y los componentes de tension
    en Pa y deformacion adimensional.
    """
    if not isinstance(subdivisions, int) or subdivisions < 2:
        raise ValueError("subdivisions debe ser un entero >= 2")

    u = np.asarray(u, dtype=float).reshape(-1)

    if u.size != mesh.number_of_dofs():
        raise ValueError("El vector U no coincide con los GDL de la malla")
    if not np.all(np.isfinite(u)):
        raise ValueError("El vector U contiene valores no finitos")
    if not elements:
        raise ValueError("No hay elementos para representar")

    fields = []

    for element in elements:
        node_ids = element.node_global_ids
        n_nodes = len(node_ids)
        points, triangles = _reference_triangulation(n_nodes, subdivisions)

        # Coordenadas geometricas y desplazamientos en el orden local.
        coords = np.asarray([
            [mesh.node(nid).x, mesh.node(nid).y]
            for nid in node_ids
        ], dtype=float)
        ue = u[element.global_dofs()]
        C = np.asarray(element.material.matriz_constitutiva(), dtype=float)

        physical_coords = np.zeros((len(points), 2))
        strains = np.zeros((len(points), 3))
        stresses = np.zeros((len(points), 3))
        sigma_1 = np.zeros(len(points))

        for i, (xi, eta) in enumerate(points):
            # El mapeo fisico usa las mismas N que la formulacion FEM.
            N = np.asarray(element.shape_functions(xi, eta), dtype=float)
            B, _detJ = element.B_matrix(xi, eta)

            physical_coords[i] = N @ coords
            strains[i] = B @ ue
            stresses[i] = C @ strains[i]

            sx, sy, txy = stresses[i]
            sigma_1[i] = (
                0.5 * (sx + sy)
                + np.hypot(0.5 * (sx - sy), txy)
            )

        if not (np.all(np.isfinite(stresses)) and
                np.all(np.isfinite(physical_coords))):
            raise ValueError(
                f"Tensiones o coordenadas no finitas en "
                f"elemento {element.global_id}"
            )

        fields.append({
            "element_id": element.global_id,
            "x": physical_coords[:, 0],
            "y": physical_coords[:, 1],
            "triangles": triangles,
            "epsilon_x": strains[:, 0],
            "epsilon_y": strains[:, 1],
            "gamma_xy": strains[:, 2],
            "sigma_x": stresses[:, 0],
            "sigma_y": stresses[:, 1],
            "tau_xy": stresses[:, 2],
            "sigma_1": sigma_1,
        })

    return fields


# =============================================================
# 3. ESCALA DE COLORES
# =============================================================

def _stress_norm(fields, key):
    """Crea una escala comun para TODOS los elementos de un mapa."""
    values = np.concatenate([field[key] for field in fields]) / 1e6
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    # Si hay traccion y compresion, el cero queda en el centro.
    if vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Si el campo es constante, evitar limites identicos.
    if np.isclose(vmin, vmax, rtol=0.0, atol=1e-14):
        margin = max(abs(vmin) * 0.01, 1e-6)
        vmin -= margin
        vmax += margin

    return Normalize(vmin=vmin, vmax=vmax)


# =============================================================
# 4. GRAFICOS DE CAMPOS DE TENSIONES
# =============================================================

def plot_stress_fields(
    mesh,
    elements,
    u,
    subdivisions=10,
    title="Presa de gravedad",
    save_filename=None,
    show=True
):
    """
    Grafica cuatro campos de tensiones sobre la geometria original:
        sigma_x, sigma_y, tau_xy y sigma_1 del plano XY.

    - Las tensiones se presentan en MPa.
    - Cada elemento se dibuja con una triangulacion INDEPENDIENTE.
      Esto evita promediar valores de tension entre elementos.
    - El suavizado visual es SOLO interno a cada elemento.
    - Las tensiones se evaluan en puntos graficos, no solo en Gauss.
    """
    fields = sample_stress_field(
        mesh, elements, u, subdivisions=subdivisions
    )

    plots = [
        ("sigma_x", r"Tension normal $\sigma_x$"),
        ("sigma_y", r"Tension normal $\sigma_y$"),
        ("tau_xy", r"Tension cortante $\tau_{xy}$"),
        ("sigma_1", r"Tension principal mayor $\sigma_1$"),
    ]

    fig, axes = plt.subplots(
        1, 4, figsize=(20, 6.5), constrained_layout=True
    )
    fig.suptitle(f"{title} - Campos de tensiones", fontsize=15)

    for ax, (key, label) in zip(axes, plots):
        norm = _stress_norm(fields, key)
        cmap = "coolwarm"

        for field in fields:
            # Se crean triangulaciones por separado para que las
            # tensiones no se mezclen entre elementos vecinos.
            triangulation = mtri.Triangulation(
                field["x"], field["y"],
                triangles=field["triangles"]
            )

            # Gouraud suaviza graficamente dentro de cada elemento,
            # conservando posibles saltos en las interfaces.
            ax.tripcolor(
                triangulation,
                field[key] / 1e6,
                shading="gouraud",
                cmap=cmap,
                norm=norm,
                rasterized=True
            )

        # Se reutiliza la representacion isoparametrica de las aristas.
        plot_element_boundaries(ax, mesh, elements)

        # Barra de colores unica para todos los elementos del panel.
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            shrink=0.78
        )
        colorbar.set_label("Tension [MPa]")

        ax.set_title(label)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15, linestyle=":")

    if save_filename:
        # project/source/output/stressPlot.py -> project/output/
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / save_filename
        fig.savefig(filepath, dpi=250, bbox_inches="tight")
        print(f"Mapas de tensiones guardados en: {filepath}")

    if show:
        plt.show()

    return fig
