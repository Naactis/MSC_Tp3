from source.loads.quadrature import integration_points


def compute_area(elements):
    """
    Calcula el area total de la malla mediante integracion
    de los determinantes de Jacobiano de sus elementos.

    A = suma_e integral(detJ dxi deta)
    """

    area = 0.0

    for element in elements:

        for xi, eta, weight in integration_points(element):

            _, detJ = element.B_matrix(xi, eta)

            area += detJ * weight

    return area
