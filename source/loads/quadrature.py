import numpy as np


def integration_points(element):
    """
    Devuelve los puntos de integracion del elemento.

    Cada punto tiene el formato:
        (xi, eta, peso)

    El peso ya incluye el area del elemento de referencia
    en el caso de los triangulos.
    """

    n = len(element.node_global_ids)

    # ---------------------------------------------------------
    # CST: triangulo lineal, regla de un punto
    # ---------------------------------------------------------
    if n == 3:
        return [
            (1.0/3.0, 1.0/3.0, 0.5)
        ]

    # ---------------------------------------------------------
    # LST: triangulo cuadratico, regla de tres puntos
    # ---------------------------------------------------------
    if n == 6:
        return [
            (1.0/6.0, 1.0/6.0, 1.0/6.0),
            (2.0/3.0, 1.0/6.0, 1.0/6.0),
            (1.0/6.0, 2.0/3.0, 1.0/6.0)
        ]

    # ---------------------------------------------------------
    # Q4: cuadratura tensorial de Gauss 2x2
    # ---------------------------------------------------------
    if n == 4:
        points = [
            -1.0 / np.sqrt(3.0),
             1.0 / np.sqrt(3.0)
        ]
        weights = [1.0, 1.0]

    # ---------------------------------------------------------
    # Q8: cuadratura tensorial de Gauss 3x3
    # ---------------------------------------------------------
    elif n == 8:
        points = [
            -np.sqrt(3.0/5.0),
             0.0,
             np.sqrt(3.0/5.0)
        ]
        weights = [
            5.0/9.0,
            8.0/9.0,
            5.0/9.0
        ]

    else:
        raise ValueError(
            f"Cuadratura no definida para {n} nodos"
        )

    # Producto tensorial de los puntos 1D
    return [
        (xi, eta, wx * wy)
        for xi, wx in zip(points, weights)
        for eta, wy in zip(points, weights)
    ]
