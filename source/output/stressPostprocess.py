"""
Postprocesamiento de deformaciones y tensiones planas en puntos de Gauss.

Compatible con CST, LST, Q4 y Q8 siempre que los elementos
implementen shape_functions(), B_matrix(), global_dofs() y material.

Convenciones:
    epsilon = [epsilon_x, epsilon_y, gamma_xy]
    sigma   = [sigma_x, sigma_y, tau_xy]

Las tensiones principales calculadas son las del plano XY.
"""

import numpy as np

from source.loads.quadrature import integration_points


def compute_strain_stress(mesh, elements, U):
    """
    Evalua deformaciones y tensiones en los puntos de Gauss.

    Parametros:
        mesh: instancia de Mesh.
        elements: elementos finitos ya instanciados.
        U: vector global de desplazamientos del solver.

    Retorna:
        Diccionario de arrays, con una fila por punto de Gauss:
            element_ids       (n,)
            natural_coords    (n, 2)
            physical_coords   (n, 2)
            strains           (n, 3)
            stresses          (n, 3)
            principal_stresses (n, 2)  [sigma_1, sigma_2]
            max_inplane_shear (n,)

    Nota: no se extrapolan tensiones a los nodos.
    """
    U = np.asarray(U, dtype=float).reshape(-1)

    if U.size != mesh.number_of_dofs():
        raise ValueError("Postproceso: longitud de U incompatible con la malla")

    if not np.all(np.isfinite(U)):
        raise ValueError("Postproceso: U contiene valores no finitos")

    element_ids = []
    natural_coords = []
    physical_coords = []
    strains = []
    stresses = []
    principal_stresses = []
    max_inplane_shear = []

    for element in elements:
        # Recuperar los desplazamientos nodales en el orden local.
        dofs = element.global_dofs()
        ue = U[dofs]

        # Matriz constitutiva del elemento (tension o deformacion plana).
        C = element.material.matriz_constitutiva()

        # Coordenadas fisicas de sus nodos, en el mismo orden que N.
        coords = np.array([
            [mesh.node(nid).x, mesh.node(nid).y]
            for nid in element.node_global_ids
        ], dtype=float)

        # Cuadraturas: CST 1, LST 3, Q4 4 y Q8 9 puntos.
        for xi, eta, _weight in integration_points(element):
            N = element.shape_functions(xi, eta)
            B, _detJ = element.B_matrix(xi, eta)

            # Posicion del punto por interpolacion isoparametrica.
            xy = N @ coords

            # Cinematica: deformacion cortante ingenieril gamma_xy.
            epsilon = B @ ue

            # Elasticidad lineal: ley constitutiva.
            sigma = C @ epsilon
            sx, sy, txy = sigma

            # Tensiones principales dentro del plano XY.
            sigma_mean = 0.5 * (sx + sy)
            radius = np.hypot(0.5 * (sx - sy), txy)
            sigma_1 = sigma_mean + radius
            sigma_2 = sigma_mean - radius

            element_ids.append(element.global_id)
            natural_coords.append([xi, eta])
            physical_coords.append(xy)
            strains.append(epsilon)
            stresses.append(sigma)
            principal_stresses.append([sigma_1, sigma_2])
            max_inplane_shear.append(radius)

    if not strains:
        raise ValueError("Postproceso: no hay puntos de integracion")

    return {
        "element_ids": np.asarray(element_ids),
        "natural_coords": np.asarray(natural_coords),
        "physical_coords": np.asarray(physical_coords),
        "strains": np.asarray(strains),
        "stresses": np.asarray(stresses),
        "principal_stresses": np.asarray(principal_stresses),
        "max_inplane_shear": np.asarray(max_inplane_shear),
    }
