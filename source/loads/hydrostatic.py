
import numpy as np


# Indices locales de las aristas de cada elemento.
# Incluyen los nodos intermedios de LST y Q8.
EDGE_NODES = {
    3: [(0, 1), (1, 2), (2, 0)],
    6: [(0, 3, 1), (1, 4, 2), (2, 5, 0)],
    4: [(0, 1), (1, 2), (2, 3), (3, 0)],
    8: [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
}


def edge_shape_functions(s, n_nodes):
    """
    Funciones de forma unidimensionales.

    Los nodos se ordenan desde abajo hacia arriba:
        Lineal:    inferior, superior.
        Cuadratico: inferior, medio, superior.
    """

    if n_nodes == 2:
        return np.array([
            0.5 * (1.0 - s),
            0.5 * (1.0 + s)
        ])

    if n_nodes == 3:
        return np.array([
            0.5 * s * (s - 1.0),
            1.0 - s**2,
            0.5 * s * (s + 1.0)
        ])

    raise ValueError("Cantidad de nodos de arista no soportada")


def compute_hydrostatic_forces(
    mesh,
    elements,
    rho_water=1000.0,
    g=9.81,
    y_water_level=28.0,
    x_upstream=0.0,
    tolerance=1e-7
):
    """
    Calcula las fuerzas nodales equivalentes de la presion
    hidrostatica sobre un paramento vertical x=x_upstream.

    Suposiciones:
        - El agua empuja hacia +X.
        - Las aristas son rectas y verticales.
        - Los nodos intermedios estan en el punto medio.
        - El espesor se obtiene de cada elemento.

    Si el agua corta una arista, solo integra su parte mojada.
    """

    f_hydro = np.zeros(mesh.number_of_dofs())

    # ---------------------------------------------------------
    # 1. Recorrido de elementos
    # ---------------------------------------------------------
    for element in elements:

        node_ids = element.node_global_ids
        n = len(node_ids)

        if n not in EDGE_NODES:
            raise ValueError(
                f"Elemento de {n} nodos no soportado"
            )

        # -----------------------------------------------------
        # 2. Recorrido de aristas del elemento
        # -----------------------------------------------------
        for local_edge in EDGE_NODES[n]:

            edge_ids = [node_ids[i] for i in local_edge]

            coords = np.array([
                [mesh.node(nid).x, mesh.node(nid).y]
                for nid in edge_ids
            ])

            # Seleccionar solamente las aristas ubicadas
            # sobre el paramento de aguas arriba.
            if not np.all(
                np.abs(coords[:, 0] - x_upstream) <= tolerance
            ):
                continue

            # Ordenar nodos desde abajo hacia arriba.
            order = np.argsort(coords[:, 1])

            edge_ids = [edge_ids[int(i)] for i in order]
            y = coords[order, 1]

            y_bottom = y[0]
            y_top = y[-1]
            length = y_top - y_bottom

            if length <= tolerance:
                raise ValueError(
                    "Arista hidrostatica degenerada"
                )

            # Para el elemento cuadratico, comprobar
            # que el nodo central sea el punto medio.
            if len(edge_ids) == 3:

                y_mid_expected = 0.5 * (y_bottom + y_top)

                if not np.isclose(
                    y[1],
                    y_mid_expected,
                    rtol=0.0,
                    atol=tolerance
                ):
                    raise ValueError(
                        "La arista cuadratica no tiene "
                        "el nodo medio en el punto medio"
                    )

            # Arista completamente por encima del agua
            if y_bottom >= y_water_level:
                continue

            # -------------------------------------------------
            # 3. Determinar el tramo mojado
            # -------------------------------------------------

            # Coordenada natural del nivel del agua
            s_water = (
                2.0 * (y_water_level - y_bottom) / length
                - 1.0
            )

            # Limite superior del tramo sumergido
            s_end = min(1.0, s_water)

            # Transformacion desde [-1,1] al intervalo
            # mojado [-1,s_end].
            s_mid = 0.5 * (s_end - 1.0)
            s_half = 0.5 * (s_end + 1.0)

            # Jacobiano de una arista recta vertical
            J_edge = length / 2.0

            # -------------------------------------------------
            # 4. Elegir la cuadratura de arista
            # -------------------------------------------------
            if len(edge_ids) == 2:

                gauss_pts = [
                    -1.0 / np.sqrt(3.0),
                     1.0 / np.sqrt(3.0)
                ]
                gauss_weights = [1.0, 1.0]

            else:

                gauss_pts = [
                    -np.sqrt(3.0/5.0),
                     0.0,
                     np.sqrt(3.0/5.0)
                ]
                gauss_weights = [
                    5.0/9.0,
                    8.0/9.0,
                    5.0/9.0
                ]

            # -------------------------------------------------
            # 5. Integracion del empuje
            # -------------------------------------------------
            for r, w in zip(gauss_pts, gauss_weights):

                # Punto dentro del intervalo mojado
                s = s_mid + s_half * r

                # Funciones de forma sobre la arista
                N = edge_shape_functions(
                    s, len(edge_ids)
                )

                # Altura del punto de integracion
                y_gp = np.dot(N, y)

                # Presion hidrostatica
                depth = max(
                    0.0,
                    y_water_level - y_gp
                )

                pressure = rho_water * g * depth

                # Fuerza diferencial
                factor = (
                    w
                    * pressure
                    * J_edge
                    * s_half
                    * element.thickness
                )

                # Ensamblaje en los GDL horizontales
                for i, nid in enumerate(edge_ids):

                    dof_x = mesh.node_global_dofs(nid)[0]

                    f_hydro[dof_x] += N[i] * factor

    return f_hydro
