"""
Mapas de desplazamiento del metodo de elementos finitos.

Soporta CST (3 nodos), LST (6 nodos), Q4 (4 nodos) y Q8 (8 nodos).
La geometria y los desplazamientos se interpolan con las funciones de forma
reales de cada elemento, incluyendo los nodos intermedios.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


# =============================================================
# 1. MUESTREO DEL CAMPO DE DESPLAZAMIENTOS
# =============================================================

def sample_displacement_field(mesh, elements, u, subdivisions=10):
    """
    Evalua (x, y, ux, uy, |u|) dentro de cada elemento.

    El dominio natural se subdivide en triangulos pequenos
    exclusivamente para la visualizacion; NO modifica la malla FEM
    ni vuelve a resolver el problema.

    Retorna un diccionario con puntos, valores y triangulos graficos.
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

    x_values = []
    y_values = []
    ux_values = []
    uy_values = []
    triangles = []

    for element in elements:
        node_ids = element.node_global_ids
        n_nodes = len(node_ids)

        if n_nodes not in (3, 4, 6, 8):
            raise ValueError(f"Elemento no soportado: {n_nodes} nodos")

        coords = np.array([
            [mesh.node(nid).x, mesh.node(nid).y]
            for nid in node_ids
        ], dtype=float)

        # En el orden local: [u1, v1, u2, v2, ...].
        nodal_u = u[element.global_dofs()].reshape(n_nodes, 2)

        # El mapeo natural y la interpolacion de U se realizan
        # usando las funciones de forma del propio elemento.
        def append_point(xi, eta):
            N = np.asarray(element.shape_functions(xi, eta), dtype=float)
            if N.shape != (n_nodes,):
                raise ValueError("Dimension inesperada de las funciones de forma")

            xy = N @ coords
            uv = N @ nodal_u

            x_values.append(xy[0])
            y_values.append(xy[1])
            ux_values.append(uv[0])
            uy_values.append(uv[1])

        idx = {}

        if n_nodes in (3, 6):
            # Triangulo natural: xi >= 0, eta >= 0, xi+eta <= 1.
            for i in range(subdivisions + 1):
                for j in range(subdivisions + 1 - i):
                    idx[(i, j)] = len(x_values)
                    append_point(i / subdivisions, j / subdivisions)

            for i in range(subdivisions):
                for j in range(subdivisions - i):
                    triangles.append([
                        idx[(i, j)],
                        idx[(i + 1, j)],
                        idx[(i, j + 1)]
                    ])

                    if i + j < subdivisions - 1:
                        triangles.append([
                            idx[(i + 1, j)],
                            idx[(i + 1, j + 1)],
                            idx[(i, j + 1)]
                        ])

        else:
            # Cuadrado natural: -1 <= xi, eta <= 1.
            for i in range(subdivisions + 1):
                for j in range(subdivisions + 1):
                    idx[(i, j)] = len(x_values)
                    append_point(
                        -1.0 + 2.0 * i / subdivisions,
                        -1.0 + 2.0 * j / subdivisions
                    )

            for i in range(subdivisions):
                for j in range(subdivisions):
                    triangles.append([
                        idx[(i, j)],
                        idx[(i + 1, j)],
                        idx[(i + 1, j + 1)]
                    ])
                    triangles.append([
                        idx[(i, j)],
                        idx[(i + 1, j + 1)],
                        idx[(i, j + 1)]
                    ])

    ux = np.asarray(ux_values)
    uy = np.asarray(uy_values)

    return {
        "x": np.asarray(x_values),
        "y": np.asarray(y_values),
        "ux": ux,
        "uy": uy,
        "magnitude": np.hypot(ux, uy),
        "triangles": np.asarray(triangles, dtype=int),
    }


# =============================================================
# 2. CONTORNOS DE LOS ELEMENTOS ORIGINALES
# =============================================================

def plot_element_boundaries(ax, mesh, elements, line_points=12):
    """Dibuja las aristas interpoladas de cada elemento."""

    for element in elements:
        n_nodes = len(element.node_global_ids)
        coords = np.array([
            [mesh.node(nid).x, mesh.node(nid).y]
            for nid in element.node_global_ids
        ], dtype=float)

        s_values = np.linspace(0.0, 1.0, line_points)

        if n_nodes in (3, 6):
            # Aristas del triangulo natural, en sentido antihorario.
            edges = [
                lambda s: (s, 0.0),
                lambda s: (1.0 - s, s),
                lambda s: (0.0, 1.0 - s),
            ]
        else:
            # Aristas del cuadrado natural, en sentido antihorario.
            edges = [
                lambda s: (-1.0 + 2.0 * s, -1.0),
                lambda s: (1.0, -1.0 + 2.0 * s),
                lambda s: (1.0 - 2.0 * s, 1.0),
                lambda s: (-1.0, 1.0 - 2.0 * s),
            ]

        for edge in edges:
            points = np.array([
                element.shape_functions(*edge(s)) @ coords
                for s in s_values
            ])
            ax.plot(
                points[:, 0], points[:, 1],
                color="black", linewidth=0.6, alpha=0.40
            )


# =============================================================
# 3. MAPAS DE DESPLAZAMIENTOS
# =============================================================

def plot_displacement_fields(
    mesh,
    elements,
    u,
    subdivisions=10,
    title="Presa de gravedad",
    save_filename=None,
    show=True
):
    """
    Genera tres mapas: ux, uy y modulo del desplazamiento.

    Los valores se muestran en milimetros sobre la geometria
    original (sin amplificar los desplazamientos).

    subdivisions controla solo la suavidad visual del grafico.
    """

    field = sample_displacement_field(
        mesh, elements, u, subdivisions=subdivisions
    )

    triangulation = mtri.Triangulation(
        field["x"], field["y"], triangles=field["triangles"]
    )

    plots = [
        ("ux", "Desplazamiento horizontal $u_x$", "coolwarm"),
        ("uy", "Desplazamiento vertical $u_y$", "coolwarm"),
        ("magnitude", "Modulo de desplazamiento $|u|$", "viridis")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    fig.suptitle(f"{title} - Campos de desplazamiento", fontsize=15)

    for ax, (key, label, cmap) in zip(axes, plots):
        values_mm = 1000.0 * field[key]

        min_value = float(np.min(values_mm))
        max_value = float(np.max(values_mm))

        if np.isclose(min_value, max_value, rtol=0.0, atol=1e-14):
            width = max(abs(min_value) * 0.01, 1e-6)
            levels = np.linspace(min_value - width, max_value + width, 25)
        else:
            levels = np.linspace(min_value, max_value, 25)

        contour = ax.tricontourf(
            triangulation, values_mm, levels=levels, cmap=cmap
        )

        # Contornos de la malla FEM, sin dibujar subdivisiones auxiliares.
        plot_element_boundaries(ax, mesh, elements)

        colorbar = fig.colorbar(contour, ax=ax, shrink=0.78)
        colorbar.set_label("Desplazamiento [mm]")

        ax.set_title(label)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15, linestyle=":")

    if save_filename:
        # project/source/output/displacementPlot.py -> project/
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / save_filename
        fig.savefig(filepath, dpi=250, bbox_inches="tight")
        print(f"Mapas de desplazamientos guardados en: {filepath}")

    if show:
        plt.show()

    return fig
