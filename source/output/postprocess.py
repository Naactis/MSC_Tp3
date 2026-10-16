
import numpy as np


# =============================================================
# 1. POSTPROCESAMIENTO DE DESPLAZAMIENTOS
# =============================================================

def compute_displacements(mesh, U):
    """
    Procesa el vector global de desplazamientos del MEF.

    Parametros:
        mesh: malla utilizada en el analisis.
        U: vector global de desplazamientos.

    Calcula:
        - Desplazamiento horizontal de cada nodo.
        - Desplazamiento vertical de cada nodo.
        - Modulo del desplazamiento de cada nodo.
        - Maximos absolutos y sus ubicaciones.

    Retorna:
        Diccionario con los resultados nodales y maximos.
    """

    # ---------------------------------------------------------
    # 1. Preparacion y validacion del vector global
    # ---------------------------------------------------------

    # Convertimos U a un vector unidimensional.
    # Esto permite trabajar tanto con (2n,) como con (2n,1).
    U = np.asarray(U, dtype=float).reshape(-1)

    # Verificamos que el vector tenga la cantidad correcta
    # de grados de libertad de la malla.
    if U.size != mesh.number_of_dofs():
        raise ValueError(
            "Postproceso: dimensiones incompatibles "
            "entre la malla y el vector U."
        )

    # Evitamos procesar desplazamientos indefinidos.
    if not np.all(np.isfinite(U)):
        raise ValueError(
            "Postproceso: U contiene valores no finitos."
        )

    n_nodes = len(mesh.nodes)

    if n_nodes == 0:
        raise ValueError(
            "Postproceso: la malla no contiene nodos."
        )

    # ---------------------------------------------------------
    # 2. Desplazamientos nodales
    # ---------------------------------------------------------

    # Inicializamos los vectores con un valor por nodo.
    ux = np.zeros(n_nodes)
    uy = np.zeros(n_nodes)
    u_magnitude = np.zeros(n_nodes)

    # Recorremos los nodos de la malla.
    for i, node in enumerate(mesh.nodes):

        # Recuperamos los GDL globales del nodo.
        dof_x, dof_y = mesh.node_global_dofs(node.global_id)

        # Componentes del desplazamiento.
        ux[i] = U[dof_x]
        uy[i] = U[dof_y]

        # Modulo del desplazamiento nodal:
        # |u| = sqrt(ux^2 + uy^2)
        u_magnitude[i] = np.hypot(ux[i], uy[i])

    # ---------------------------------------------------------
    # 3. Localizacion de desplazamientos maximos
    # ---------------------------------------------------------

    # Indices de los nodos con mayores valores absolutos.
    idx_max_ux = int(np.argmax(np.abs(ux)))
    idx_max_uy = int(np.argmax(np.abs(uy)))
    idx_max_u = int(np.argmax(u_magnitude))

    # ---------------------------------------------------------
    # 4. Organizacion de los resultados
    # ---------------------------------------------------------

    # Los indices anteriores corresponden a posiciones
    # dentro de mesh.nodes, no necesariamente a IDs globales.
    return {
        "ux": ux,
        "uy": uy,
        "magnitude": u_magnitude,

        "max_ux": {
            "node_id": mesh.nodes[idx_max_ux].global_id,
            "value": float(ux[idx_max_ux]),
            "x": mesh.nodes[idx_max_ux].x,
            "y": mesh.nodes[idx_max_ux].y
        },

        "max_uy": {
            "node_id": mesh.nodes[idx_max_uy].global_id,
            "value": float(uy[idx_max_uy]),
            "x": mesh.nodes[idx_max_uy].x,
            "y": mesh.nodes[idx_max_uy].y
        },

        "max_magnitude": {
            "node_id": mesh.nodes[idx_max_u].global_id,
            "value": float(u_magnitude[idx_max_u]),
            "x": mesh.nodes[idx_max_u].x,
            "y": mesh.nodes[idx_max_u].y
        }
    }
