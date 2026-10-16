import numpy as np

from source.loads.quadrature import integration_points


def compute_body_forces(
    mesh,
    elements,
    rho_concrete,
    g=9.81
):
    """
    Calcula el vector global de fuerzas nodales equivalentes
    al peso propio.

    Formulacion:
        fe = t * integral(N.T @ b) dA

    donde:
        b = [0, -rho_concrete*g]

    El espesor se obtiene de cada elemento.
    """

    # Vector global de fuerzas
    f_body = np.zeros(mesh.number_of_dofs())

    # Fuerza de volumen vertical, negativa por gravedad
    b_y = -rho_concrete * g

    # ---------------------------------------------------------
    # 1. Recorrido de los elementos
    # ---------------------------------------------------------
    for element in elements:

        # Grados de libertad globales del elemento
        dofs = element.global_dofs()

        # Vector local de fuerzas
        fe = np.zeros(len(dofs))

        # -----------------------------------------------------
        # 2. Integracion de Gauss
        # -----------------------------------------------------
        for xi, eta, weight in integration_points(element):

            # Funciones de forma
            N = element.shape_functions(xi, eta)

            # Determinante del Jacobiano
            _, detJ = element.B_matrix(xi, eta)

            # Factor diferencial de volumen
            dV = element.thickness * detJ * weight

            # El peso propio actua solamente en Y.
            # Los GDL verticales tienen indices impares.
            fe[1::2] += N * b_y * dV

        # -----------------------------------------------------
        # 3. Ensamblaje en el vector global
        # -----------------------------------------------------
        f_body[dofs] += fe

    return f_body
