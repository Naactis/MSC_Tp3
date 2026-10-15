

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------
# Configuracion de rutas del proyecto
# ---------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Importaciones de los modulos del proyecto
# ---------------------------------------------------------

from source.mesh.node import Node
from source.mesh.mesh import Mesh
from source.materials.LinearElastic import LinearElastic
from source.systems.assembly import assemble_stiffness
from source.systems.bc import solve_system_direct
from source.elements.cst import CST


# =============================================================
# PRUEBA 1: TEST DE UN ELEMENTO CST AISLADO
# =============================================================

def test_elemento_cst(material):
    """
    Verifica la formulacion de un elemento CST aislado.

    Se comprueban:
        - Funciones de forma e interpolacion nodal.
        - Particion de la unidad.
        - Area y Jacobiano.
        - Matriz de rigidez y simetria.
        - Tres modos de cuerpo rigido.
        - Reproduccion de un campo lineal de desplazamientos.
        - Deformaciones y tensiones analiticas.
    """

    # ---------------------------------------------------------
    # 1. Geometria del elemento
    # ---------------------------------------------------------

    # Triangulo rectangulo de lados 2 y 3.
    # Area = 3 y determinante del Jacobiano = 6.
    mesh = Mesh()

    mesh.nodes = [
        Node(0, 0.0, 0.0),
        Node(1, 2.0, 0.0),
        Node(2, 0.0, 3.0)
    ]

    elemento = CST(
        0, [0, 1, 2], mesh, material, thickness=1.0
    )

    # ---------------------------------------------------------
    # 2. Funciones de forma
    # ---------------------------------------------------------

    # En cada nodo natural, la funcion de forma asociada
    # debe valer 1 y las otras dos deben valer 0.
    puntos_naturales = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0)
    ]

    for i, (xi, eta) in enumerate(puntos_naturales):

        N = elemento.shape_functions(xi, eta)

        np.testing.assert_allclose(
            N,
            np.eye(3)[i],
            rtol=0.0,
            atol=1e-12
        )

    # La suma de las funciones de forma debe valer 1.
    for xi, eta in puntos_naturales + [(1/3, 1/3)]:

        N = elemento.shape_functions(xi, eta)

        assert np.isclose(
            np.sum(N), 1.0, atol=1e-12
        )

    # ---------------------------------------------------------
    # 3. Area, Jacobiano y matriz B
    # ---------------------------------------------------------

    assert np.isclose(elemento.area, 3.0, atol=1e-12)

    # Para un CST, B y detJ son constantes.
    for xi, eta in [(0.0, 0.0), (1/3, 1/3), (0.2, 0.3)]:

        B, detJ = elemento.B_matrix(xi, eta)

        assert B.shape == (3, 6)
        assert np.isclose(detJ, 6.0, atol=1e-12)

        np.testing.assert_allclose(
            B,
            elemento.B,
            rtol=0.0,
            atol=1e-12
        )

    # Las derivadas de las funciones de forma deben sumar cero.
    np.testing.assert_allclose(
        np.sum(elemento.dN_dx), 0.0, atol=1e-12
    )

    np.testing.assert_allclose(
        np.sum(elemento.dN_dy), 0.0, atol=1e-12
    )

    # ---------------------------------------------------------
    # 4. Matriz de rigidez elemental
    # ---------------------------------------------------------

    Ke = elemento.stiffness_matrix()

    # Tres nodos con dos GDL por nodo: matriz 6x6.
    assert Ke.shape == (6, 6)

    # La matriz de rigidez debe ser simetrica.
    np.testing.assert_allclose(
        Ke, Ke.T,
        rtol=1e-12,
        atol=1e-10
    )

    # ---------------------------------------------------------
    # 5. Movimientos de cuerpo rigido
    # ---------------------------------------------------------

    # El CST debe poseer exactamente tres modos de
    # energia nula: dos traslaciones y una rotacion.
    autovalores = np.linalg.eigvalsh(Ke)

    umbral = 1e-9 * np.max(np.abs(autovalores))

    assert np.count_nonzero(
        np.abs(autovalores) < umbral
    ) == 3

    assert np.count_nonzero(
        autovalores > umbral
    ) == 3

    coords = elemento.coords

    # Traslacion horizontal uniforme
    traslacion_x = np.tile([1.0, 0.0], 3)

    # Traslacion vertical uniforme
    traslacion_y = np.tile([0.0, 1.0], 3)

    # Rotacion infinitesimal:
    # ux = -y, uy = x
    rotacion_z = np.column_stack((
        -coords[:, 1],
        coords[:, 0]
    )).reshape(-1)

    escala_K = np.linalg.norm(Ke, ord=np.inf)

    for movimiento in [
        traslacion_x,
        traslacion_y,
        rotacion_z
    ]:

        # Ningun movimiento rigido debe producir deformacion.
        np.testing.assert_allclose(
            elemento.B @ movimiento,
            0.0,
            rtol=0.0,
            atol=1e-12
        )

        # Tampoco debe generar fuerzas elasticas internas.
        error_relativo = (
            np.linalg.norm(Ke @ movimiento, ord=np.inf)
            / (
                escala_K
                * np.linalg.norm(movimiento, ord=np.inf)
            )
        )

        assert error_relativo < 1e-11

    # ---------------------------------------------------------
    # 6. Reproduccion de un campo lineal
    # ---------------------------------------------------------

    # Campo analitico:
    #
    # ux = a0 + a1*x + a2*y
    # uy = b0 + b1*x + b2*y
    #
    # Las deformaciones exactas son:
    #
    # epsilon_x  = a1
    # epsilon_y  = b2
    # gamma_xy   = a2 + b1

    a0, a1, a2 = 0.005, 0.002, 0.003
    b0, b1, b2 = -0.004, 0.004, -0.001

    U = np.zeros(mesh.number_of_dofs())

    # Se asignan los desplazamientos exactos a los nodos.
    for node in mesh.nodes:

        dofs = mesh.node_global_dofs(node.global_id)

        ux = a0 + a1*node.x + a2*node.y
        uy = b0 + b1*node.x + b2*node.y

        U[dofs] = [ux, uy]

    epsilon_exacta = np.array([
        a1,
        b2,
        a2 + b1
    ])

    # Tensiones exactas para un material en tension plana.
    E = material.E
    nu = material.nu

    sigma_exacta = np.array([
        E / (1 - nu**2) * (a1 + nu*b2),
        E / (1 - nu**2) * (nu*a1 + b2),
        E / (2*(1 + nu)) * (a2 + b1)
    ])

    # Como el CST tiene deformacion constante,
    # el resultado debe ser el mismo en cualquier punto.
    for xi, eta in [
        (0.0, 0.0),
        (1/3, 1/3),
        (0.2, 0.3)
    ]:

        np.testing.assert_allclose(
            elemento.strain(U, xi, eta),
            epsilon_exacta,
            rtol=1e-10,
            atol=1e-12
        )

        np.testing.assert_allclose(
            elemento.stress(U, xi, eta),
            sigma_exacta,
            rtol=1e-10,
            atol=1e-10
        )

    print(
        "CST aislado: funciones de forma, Jacobiano, "
        "rigidez, modos rigidos y campo lineal verificados."
    )


# =============================================================
# PRUEBA 2: PATCH TEST GLOBAL DE CUATRO CST
# =============================================================

def run_patch_test_cst(plot=False):
    """
    Patch test sobre una malla de cuatro elementos CST.

    La malla contiene:
        - 4 nodos de contorno.
        - 1 nodo interior libre.
        - 4 elementos triangulares.

    Se imponen desplazamientos lineales en el contorno
    y se comprueba que el nodo interior reproduce el
    campo exacto.

    Tambien se verifican:
        - Ensamblaje y simetria de la matriz global.
        - Equilibrio de los GDL libres.
        - Deformaciones y tensiones constantes.
    """

    # ---------------------------------------------------------
    # 1. Material
    # ---------------------------------------------------------

    material = LinearElastic(
        E=1000.0,
        nu=0.3,
        plane_state="stress"
    )

    thickness = 1.0

    # Ejecutamos primero los testeos del elemento aislado.
    test_elemento_cst(material)

    # ---------------------------------------------------------
    # 2. Malla del patch test
    # ---------------------------------------------------------

    # Cuadrado de 2x2 con un nodo interior distorsionado.
    #
    #           3-------------2
    #           | \         / |
    #           |   \     /   |
    #           |     4       |
    #           |   /   \     |
    #           | /       \   |
    #           0-------------1
    #
    # Nodo 4: (0.8, 1.2)
    #
    # El desplazamiento del nodo interior evita
    # que los cuatro triangulos sean simetricos.

    mesh = Mesh()

    mesh.nodes = [
        Node(0, 0.0, 0.0),
        Node(1, 2.0, 0.0),
        Node(2, 2.0, 2.0),
        Node(3, 0.0, 2.0),
        Node(4, 0.8, 1.2)
    ]

    # ---------------------------------------------------------
    # 3. Elementos CST
    # ---------------------------------------------------------

    # Los cuatro triangulos se definen con nodos
    # en sentido antihorario.
    #
    # Todos comparten el nodo interior 4.

    elements = [
        CST(0, [0, 1, 4], mesh, material, thickness),
        CST(1, [1, 2, 4], mesh, material, thickness),
        CST(2, [2, 3, 4], mesh, material, thickness),
        CST(3, [3, 0, 4], mesh, material, thickness)
    ]

    # ---------------------------------------------------------
    # 4. Ensamblaje global
    # ---------------------------------------------------------

    K = assemble_stiffness(mesh, elements)

    # Se considera un vector de fuerzas externas nulo.
    # El problema se resuelve mediante desplazamientos
    # impuestos sobre el contorno.
    f = np.zeros(mesh.number_of_dofs())

    # Cinco nodos con dos GDL: matriz global 10x10.
    assert K.shape == (10, 10)

    # La matriz global debe ser simetrica.
    np.testing.assert_allclose(
        K, K.T,
        rtol=1e-12,
        atol=1e-10
    )

    # ---------------------------------------------------------
    # 5. Verificacion del ensamblaje
    # ---------------------------------------------------------

    # La energia cuadratica calculada con la matriz global
    # debe coincidir con la suma de las contribuciones
    # elementales para cualquier vector de prueba.
    U_prueba = np.linspace(
        -0.002, 0.003, mesh.number_of_dofs()
    )

    energia_global = U_prueba @ K @ U_prueba

    energia_elemental = 0.0

    for element in elements:

        dofs = element.global_dofs()
        ue = U_prueba[dofs]
        Ke = element.stiffness_matrix()

        energia_elemental += ue @ Ke @ ue

    assert np.isclose(
        energia_global,
        energia_elemental,
        rtol=1e-11,
        atol=1e-12
    )

    # ---------------------------------------------------------
    # 6. Campo analitico de desplazamientos
    # ---------------------------------------------------------

    # Campo lineal de traccion uniaxial:
    #
    # ux = c*x
    # uy = -nu*c*y
    #
    # Para tension plana:
    #
    # epsilon = [c, -nu*c, 0]
    # sigma   = [E*c, 0, 0]

    c = 0.001

    def desplazamiento_exacto(node):

        ux = c * node.x
        uy = -material.nu * c * node.y

        return np.array([ux, uy])

    # ---------------------------------------------------------
    # 7. Condiciones de borde
    # ---------------------------------------------------------

    # Se restringen ambos GDL de los cuatro vertices.
    boundary_nodes = [0, 1, 2, 3]

    # El nodo central permanece libre.
    internal_nodes = [4]

    fixed_dofs = []
    fixed_values = []

    for nid in boundary_nodes:

        node = mesh.node(nid)
        dofs = mesh.node_global_dofs(nid)

        fixed_dofs.extend(dofs)
        fixed_values.extend(desplazamiento_exacto(node))

    # ---------------------------------------------------------
    # 8. Resolucion del sistema
    # ---------------------------------------------------------

    U, reacciones = solve_system_direct(
        K,
        f,
        fixed_dofs,
        fixed_values
    )

    U = np.asarray(U).reshape(-1)

    assert U.size == mesh.number_of_dofs()

    # ---------------------------------------------------------
    # 9. Verificacion de desplazamientos
    # ---------------------------------------------------------

    # Se comparan los desplazamientos numericos con
    # los exactos en los cinco nodos.
    #
    # El nodo 4 es especialmente importante porque
    # sus desplazamientos fueron calculados por el solver.

    error_max_desplazamientos = 0.0

    for node in mesh.nodes:

        dofs = mesh.node_global_dofs(node.global_id)

        esperado = desplazamiento_exacto(node)
        calculado = U[dofs]

        error = np.max(np.abs(calculado - esperado))

        error_max_desplazamientos = max(
            error_max_desplazamientos,
            error
        )

        np.testing.assert_allclose(
            calculado,
            esperado,
            rtol=1e-9,
            atol=1e-10,
            err_msg=(
                f"CST: desplazamientos incorrectos "
                f"en nodo {node.global_id}"
            )
        )

    # ---------------------------------------------------------
    # 10. Equilibrio del nodo interior
    # ---------------------------------------------------------

    # Los GDL del nodo 4 no tienen fuerzas externas.
    # Por lo tanto, deben satisfacer:
    #
    # (K @ U - f)_libres = 0

    free_dofs = mesh.node_global_dofs(4)

    np.testing.assert_allclose(
        (K @ U - f)[free_dofs],
        0.0,
        rtol=0.0,
        atol=1e-9
    )

    # ---------------------------------------------------------
    # 11. Verificacion de deformaciones y tensiones
    # ---------------------------------------------------------

    epsilon_exacta = np.array([
        c,
        -material.nu * c,
        0.0
    ])

    sigma_exacta = np.array([
        material.E * c,
        0.0,
        0.0
    ])

    # La deformacion de cada CST es constante.
    # Evaluamos en el baricentro natural del triangulo.
    xi = 1.0 / 3.0
    eta = 1.0 / 3.0

    for element in elements:

        epsilon_fem = element.strain(U, xi, eta)
        sigma_fem = element.stress(U, xi, eta)

        np.testing.assert_allclose(
            epsilon_fem,
            epsilon_exacta,
            rtol=1e-9,
            atol=1e-10,
            err_msg=(
                f"CST: deformaciones incorrectas "
                f"en elemento {element.global_id}"
            )
        )

        np.testing.assert_allclose(
            sigma_fem,
            sigma_exacta,
            rtol=1e-9,
            atol=1e-8,
            err_msg=(
                f"CST: tensiones incorrectas "
                f"en elemento {element.global_id}"
            )
        )

    # ---------------------------------------------------------
    # 12. Resultados
    # ---------------------------------------------------------

    print(
        f"CST patch: {len(mesh.nodes)} nodos, "
        f"{len(elements)} elementos, "
        f"{len(internal_nodes)} nodo interior libre."
    )

    print(
        f"Error maximo de desplazamientos: "
        f"{error_max_desplazamientos:.3e}"
    )

    print(
        "CST patch: equilibrio, desplazamientos, "
        "deformaciones y tensiones verificados."
    )

    # ---------------------------------------------------------
    # 13. Visualizacion opcional
    # ---------------------------------------------------------

    if plot:

        from source.output.plot import plot_deformed_mesh

        plot_deformed_mesh(
            mesh,
            elements,
            U,
            scale_factor=300.0,
            save_filename="patch_test_cst.png"
        )

    print(
        "\n>>> PATCH TEST CST: SUPERADO EXITOSAMENTE <<<\n"
    )


# =============================================================
# EJECUCION
# =============================================================

if __name__ == "__main__":

    # Ejecutar el patch test y generar el grafico.
    run_patch_test_cst(plot=True)
