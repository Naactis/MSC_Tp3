"""
Patch test del elemento Q4.

Se utiliza una malla de cuatro cuadrilateros con un nodo
interior distorsionado. Se impone un campo lineal GENERAL de
desplazamientos sobre el contorno (las tres deformaciones
distintas de cero) y se verifica que el sistema reproduzca la
solucion analitica en desplazamientos, reacciones,
deformaciones y tensiones.

El ensayo se repite para tension plana y deformacion plana.

Ejecutar desde la raiz del proyecto:
    python tests/testQ4.py
"""

import os
import sys

import numpy as np

# Se agrega la raiz del proyecto al path para poder ejecutar
# el archivo directamente con: python tests/testQ4.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.mesh.node import Node
from source.mesh.mesh import Mesh
from source.materials.LinearElastic import LinearElastic
from source.elements.q4 import Q4
from source.systems.assembly import assemble_stiffness
from source.systems.bc import solve_system_direct


# =============================================================
# CAMPO ANALITICO DE DESPLAZAMIENTOS
# =============================================================

# Campo lineal general:
#
#   ux(x,y) = a0 + a1*x + a2*y
#   uy(x,y) = b0 + b1*x + b2*y
#
# Deformaciones exactas (constantes):
#   epsilon_x = a1
#   epsilon_y = b2
#   gamma_xy  = a2 + b1
#
# Se eligen coeficientes sin relacion especial entre si para que
# epsilon_x, epsilon_y y gamma_xy sean distintas de cero. Asi
# participan todos los terminos de B y de C, incluido el de corte.
# Los terminos constantes a0, b0 agregan una traslacion rigida.

A0, A1, A2 = 0.0002, 0.0010, 0.0020
B0, B1, B2 = -0.0001, 0.0005, -0.0015


def desplazamiento_exacto(node):
    ux = A0 + A1 * node.x + A2 * node.y
    uy = B0 + B1 * node.x + B2 * node.y
    return np.array([ux, uy])


EPSILON_EXACTA = np.array([A1, B2, A2 + B1])


def tension_exacta(E, nu, plane_state):
    """
    Tensiones exactas del campo impuesto, calculadas a mano a partir
    de E y nu. NO se usa material.matriz_constitutiva(): si la matriz
    C del codigo tuviera un error, el valor esperado tendria el mismo
    error y el test no lo detectaria.

    Se escriben con los parametros de Lame:
        mu     = E / (2 (1 + nu))          (modulo de corte G)
        lambda = E nu / ((1 + nu)(1 - 2 nu))

    Deformacion plana:
        sigma_x = (lambda + 2 mu) eps_x + lambda eps_y
        sigma_y = lambda eps_x + (lambda + 2 mu) eps_y
        tau_xy  = mu gamma_xy

    Tension plana: igual, reemplazando lambda por
        lambda* = 2 lambda mu / (lambda + 2 mu) = E nu / (1 - nu^2)
    """

    eps_x, eps_y, gamma_xy = EPSILON_EXACTA

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    if plane_state == "stress":
        lam = 2.0 * lam * mu / (lam + 2.0 * mu)

    return np.array([
        (lam + 2.0 * mu) * eps_x + lam * eps_y,
        lam * eps_x + (lam + 2.0 * mu) * eps_y,
        mu * gamma_xy
    ])


# =============================================================
# PATCH TEST
# =============================================================

def run_patch_test_q4(plane_state="stress", plot=False):
    """
    Ejecuta el patch test de cuatro elementos Q4.

    Parametros:
        plane_state: "stress" (tension plana) o "strain"
                     (deformacion plana).
        plot: permite generar opcionalmente la imagen
              de la malla deformada.

    Verifica:
        - Dimensiones y simetria de las matrices de rigidez.
        - Movimientos de cuerpo rigido elementales.
        - Reproduccion del campo lineal de desplazamientos.
        - Reacciones analiticas y equilibrio global.
        - Deformaciones y tensiones constantes en los puntos de Gauss.
    """

    etiqueta = f"Q4 [{plane_state}]"

    # =========================================================
    # 1. MATERIAL Y GEOMETRIA DEL PARCHE
    # =========================================================

    mat = LinearElastic(E=1000.0, nu=0.3, plane_state=plane_state)
    thickness = 1.0

    # Parche de 2x2 con ocho nodos de contorno y uno interior,
    # desplazado respecto de la posicion central.
    mesh = Mesh()

    mesh.nodes = [
        Node(0, 0.0, 0.0),
        Node(1, 1.0, 0.0),
        Node(2, 2.0, 0.0),
        Node(3, 2.0, 1.0),
        Node(4, 2.0, 2.0),
        Node(5, 1.0, 2.0),
        Node(6, 0.0, 2.0),
        Node(7, 0.0, 1.0),
        Node(8, 0.8, 1.2)
    ]

    # Cuatro elementos Q4 con vertices en sentido antihorario,
    # que comparten el nodo interior 8.
    elements = [
        Q4(0, [0, 1, 8, 7], mesh, mat, thickness),
        Q4(1, [1, 2, 3, 8], mesh, mat, thickness),
        Q4(2, [8, 3, 4, 5], mesh, mat, thickness),
        Q4(3, [7, 8, 5, 6], mesh, mat, thickness)
    ]

    # Aristas del contorno exterior: (nodos de la arista, normal exterior).
    # Todas tienen longitud 1.
    aristas_contorno = [
        ([0, 1], ( 0.0, -1.0)),   # borde inferior
        ([1, 2], ( 0.0, -1.0)),
        ([2, 3], ( 1.0,  0.0)),   # borde derecho
        ([3, 4], ( 1.0,  0.0)),
        ([4, 5], ( 0.0,  1.0)),   # borde superior
        ([5, 6], ( 0.0,  1.0)),
        ([6, 7], (-1.0,  0.0)),   # borde izquierdo
        ([7, 0], (-1.0,  0.0))
    ]

    # Fuerzas nodales equivalentes de una traccion uniforme sobre
    # una arista lineal de longitud L: t * L * [1/2, 1/2].
    pesos_arista = [0.5, 0.5]
    longitud_arista = 1.0

    # =========================================================
    # 2. ENSAMBLAJE Y VERIFICACION DE RIGIDEZ
    # =========================================================

    K = assemble_stiffness(mesh, elements)

    # No se aplican cargas externas: el ensayo se realiza
    # imponiendo desplazamientos en el contorno.
    f = np.zeros(mesh.number_of_dofs())

    # Nueve nodos con dos GDL producen una matriz 18x18.
    assert K.shape == (18, 18)
    np.testing.assert_allclose(K, K.T, rtol=1e-12, atol=1e-10)

    for element in elements:

        Ke = element.stiffness_matrix()

        # Cada Q4 tiene cuatro nodos y ocho GDL.
        assert Ke.shape == (8, 8)
        np.testing.assert_allclose(Ke, Ke.T, rtol=1e-12, atol=1e-10)

        # Movimientos de cuerpo rigido: dos traslaciones y una
        # rotacion infinitesimal. No deben generar fuerzas: Ke @ U = 0.
        coords = element.coords

        traslacion_x = np.tile([1.0, 0.0], 4)
        traslacion_y = np.tile([0.0, 1.0], 4)
        rotacion_z = np.column_stack((-coords[:, 1], coords[:, 0])).reshape(-1)

        escala_K = np.linalg.norm(Ke, ord=np.inf)

        for movimiento in [traslacion_x, traslacion_y, rotacion_z]:
            residuo = Ke @ movimiento
            error_relativo = np.linalg.norm(residuo, ord=np.inf) / (
                escala_K * np.linalg.norm(movimiento, ord=np.inf)
            )
            assert error_relativo < 1e-11

    print(f"{etiqueta}: matrices de rigidez y modos rigidos verificados.")

    # =========================================================
    # 3. CONDICIONES DE BORDE Y RESOLUCION
    # =========================================================

    # Se impone el campo exacto en los ocho nodos del contorno.
    # El nodo 8 es el unico nodo interior libre.
    boundary_nodes = [0, 1, 2, 3, 4, 5, 6, 7]

    fixed_dofs = []
    fixed_values = []

    for nid in boundary_nodes:
        fixed_dofs.extend(mesh.node_global_dofs(nid))
        fixed_values.extend(desplazamiento_exacto(mesh.node(nid)))

    u, reacciones = solve_system_direct(K, f, fixed_dofs, fixed_values)
    u = np.asarray(u).reshape(-1)

    assert u.size == mesh.number_of_dofs()

    # =========================================================
    # 4. VERIFICACION DE DESPLAZAMIENTOS
    # =========================================================

    # Se comparan todos los nodos; la comprobacion del nodo
    # interior es la que evalua la reproduccion del campo.
    for node in mesh.nodes:
        np.testing.assert_allclose(
            u[mesh.node_global_dofs(node.global_id)],
            desplazamiento_exacto(node),
            rtol=1e-9, atol=1e-10,
            err_msg=f"{etiqueta}: desplazamientos incorrectos en nodo {node.global_id}"
        )

    print(f"{etiqueta}: desplazamientos nodales verificados.")

    # =========================================================
    # 5. VERIFICACION DE REACCIONES
    # =========================================================

    # Verificar desplazamientos y deformaciones no alcanza: con
    # f = 0, si K estuviera multiplicada por una constante
    # (pesos de Gauss, espesor o E mal), la solucion seria la misma.
    # Las reacciones si dependen de la magnitud de K.
    #
    # El estado de tension exacto es uniforme y se calcula a mano
    # (ver tension_exacta), independiente de la matriz C del codigo.
    # En cada arista del contorno, la traccion es t = S @ n, con
    #
    #   S = [[sigma_x, tau_xy],
    #        [tau_xy,  sigma_y]]
    #
    # y se reparte en los nodos con los pesos de la arista.
    # Los nodos de esquina reciben el aporte de dos aristas.
    sigma_exacta = tension_exacta(mat.E, mat.nu, plane_state)

    S = np.array([
        [sigma_exacta[0], sigma_exacta[2]],
        [sigma_exacta[2], sigma_exacta[1]]
    ])

    reacciones_exactas = np.zeros(mesh.number_of_dofs())

    for nodos_arista, normal in aristas_contorno:
        traccion = S @ np.array(normal)

        for nid, peso in zip(nodos_arista, pesos_arista):
            dofs = mesh.node_global_dofs(nid)
            reacciones_exactas[dofs] += traccion * peso * longitud_arista * thickness

    np.testing.assert_allclose(
        reacciones, reacciones_exactas,
        rtol=1e-9, atol=1e-10,
        err_msg=f"{etiqueta}: reacciones incorrectas (revisar K y C)"
    )

    # Equilibrio global de fuerzas
    assert abs(np.sum(reacciones[0::2])) < 1e-10
    assert abs(np.sum(reacciones[1::2])) < 1e-10

    print(f"{etiqueta}: reacciones y equilibrio global verificados.")

    # =========================================================
    # 6. VERIFICACION DE DEFORMACIONES Y TENSIONES
    # =========================================================

    # Un campo lineal produce deformaciones y tensiones constantes
    # en todo el parche. Se verifican en los cuatro puntos de Gauss
    # de cada elemento.
    gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

    for element in elements:
        for xi in gauss_pts:
            for eta in gauss_pts:

                np.testing.assert_allclose(
                    element.strain(u, xi, eta), EPSILON_EXACTA,
                    rtol=1e-9, atol=1e-10,
                    err_msg=f"{etiqueta}: deformaciones incorrectas en elemento {element.global_id}"
                )

                np.testing.assert_allclose(
                    element.stress(u, xi, eta), sigma_exacta,
                    rtol=1e-9, atol=1e-8,
                    err_msg=f"{etiqueta}: tensiones incorrectas en elemento {element.global_id}"
                )

    print(f"{etiqueta}: deformaciones y tensiones constantes verificadas.")

    # =========================================================
    # 7. VISUALIZACION OPCIONAL
    # =========================================================

    if plot:
        from source.output.plot import plot_deformed_mesh

        plot_deformed_mesh(
            mesh, elements, u,
            scale_factor=100.0,
            save_filename=f"patch_test_q4_{plane_state}.png"
        )

    print(f"\n>>> Patch Test {etiqueta}: SUPERADO EXITOSAMENTE <<<\n")


if __name__ == "__main__":
    for estado in ["stress", "strain"]:
        run_patch_test_q4(plane_state=estado, plot=True)