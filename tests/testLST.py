
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------
# Configuracion de rutas del proyecto
# ---------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from source.mesh.node import Node
from source.mesh.mesh import Mesh
from source.materials.LinearElastic import LinearElastic
from source.elements.lst import LST
from source.systems.assembly import assemble_stiffness
from source.systems.bc import solve_system_direct


# Puntos y pesos de Gauss de la regla triangular de tres puntos.
# Los pesos suman 1/2, area del triangulo natural de referencia.
GAUSS_TRIANGULO = [
    (1.0 / 6.0, 1.0 / 6.0),
    (2.0 / 3.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0),
]


# =============================================================
# PRUEBA 1: VERIFICACIONES DE UN ELEMENTO LST AISLADO
# =============================================================

def test_elemento_lst(material):
    """Verifica propiedades de interpolacion y de rigidez del LST."""

    # Triangulo rectangulo de lados 2 y 3, con nodos intermedios
    # ubicados exactamente en los puntos medios de sus aristas.
    # En este caso el mapeo isoparametrico es afin: x=2*xi, y=3*eta.
    mesh = Mesh()
    mesh.nodes = [
        Node(0, 0.0, 0.0),
        Node(1, 2.0, 0.0),
        Node(2, 0.0, 3.0),
        Node(3, 1.0, 0.0),   # Arista 1-2
        Node(4, 1.0, 1.5),   # Arista 2-3
        Node(5, 0.0, 1.5),   # Arista 3-1
    ]

    elemento = LST(0, [0, 1, 2, 3, 4, 5], mesh, material, 1.0)

    # 1.1. Interpolacion nodal: N_i vale 1 en el nodo i y 0 en los demas.
    nodos_naturales = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        (0.5, 0.0), (0.5, 0.5), (0.0, 0.5),
    ]
    for i, (xi, eta) in enumerate(nodos_naturales):
        esperado = np.eye(6)[i]
        np.testing.assert_allclose(
            elemento.shape_functions(xi, eta), esperado,
            rtol=0.0, atol=1e-12,
            err_msg=f"Interpolacion nodal incorrecta en nodo local {i+1}",
        )

    # 1.2. Particion de la unidad y derivadas de la particion.
    # Sum(N_i)=1 implica Sum(dN_i/dxi)=Sum(dN_i/deta)=0.
    for xi, eta in GAUSS_TRIANGULO + [(1.0 / 3.0, 1.0 / 3.0)]:
        N = elemento.shape_functions(xi, eta)
        dN_dxi, dN_deta = elemento._shape_derivatives(xi, eta)
        assert np.isclose(np.sum(N), 1.0, atol=1e-12)
        assert np.isclose(np.sum(dN_dxi), 0.0, atol=1e-12)
        assert np.isclose(np.sum(dN_deta), 0.0, atol=1e-12)

        # 1.3. Jacobiano teorico del mapeo x=2*xi, y=3*eta.
        J, detJ = elemento._jacobian(xi, eta)
        np.testing.assert_allclose(
            J, [[2.0, 0.0], [0.0, 3.0]],
            rtol=0.0, atol=1e-12,
        )
        assert np.isclose(detJ, 6.0, atol=1e-12)

    # 1.4. Propiedades de la rigidez de un LST: matriz 12x12 y simetrica.
    Ke = elemento.stiffness_matrix()
    assert Ke.shape == (12, 12)
    np.testing.assert_allclose(Ke, Ke.T, rtol=1e-12, atol=1e-10)

    # 1.5. Integracion completa: Ke debe tener exactamente tres
    # autovalores nulos (modos de cuerpo rigido), sin modos espurios.
    autovalores = np.linalg.eigvalsh(Ke)
    umbral = 1e-9 * np.max(np.abs(autovalores))
    assert np.count_nonzero(np.abs(autovalores) < umbral) == 3
    assert np.all(autovalores[3:] > umbral)

    # 1.6. Los tres movimientos rigidos deben producir deformacion nula.
    coords = elemento.coords
    traslacion_x = np.tile([1.0, 0.0], 6)
    traslacion_y = np.tile([0.0, 1.0], 6)
    rotacion_z = np.column_stack((-coords[:, 1], coords[:, 0])).reshape(-1)
    escala_K = np.linalg.norm(Ke, ord=np.inf)

    for movimiento in [traslacion_x, traslacion_y, rotacion_z]:
        for xi, eta in GAUSS_TRIANGULO:
            B, _ = elemento.B_matrix(xi, eta)
            np.testing.assert_allclose(B @ movimiento, 0.0, rtol=0.0, atol=1e-12)

        error_relativo = (
            np.linalg.norm(Ke @ movimiento, ord=np.inf)
            / (escala_K * np.linalg.norm(movimiento, ord=np.inf))
        )
        assert error_relativo < 1e-11

    # 1.7. Prueba propia del LST: reproduccion de un campo cuadratico.
    # Se prescribe analiticamente en los seis nodos, sin resolver un sistema.
    # El LST afin debe reproducir exactamente este campo y sus derivadas.
    def desplazamiento_cuadratico(x, y):
        ux = 0.001*x**2 + 0.002*x*y + 0.003*y
        uy = 0.002*y**2 - 0.001*x*y + 0.004*x
        return ux, uy

    U = np.zeros(mesh.number_of_dofs())
    for node in mesh.nodes:
        U[mesh.node_global_dofs(node.global_id)] = desplazamiento_cuadratico(
            node.x, node.y
        )

    for xi, eta in GAUSS_TRIANGULO + [(1.0 / 3.0, 1.0 / 3.0)]:
        # Las coordenadas fisicas se interpolan con las mismas N_i.
        x, y = elemento.shape_functions(xi, eta) @ elemento.coords

        # Derivadas exactas de ux(x,y) y uy(x,y):
        # ex = dux/dx, ey = duy/dy, gamma_xy = dux/dy + duy/dx.
        epsilon_exacta = np.array([
            0.002*x + 0.002*y,
            0.004*y - 0.001*x,
            0.002*x - 0.001*y + 0.007,
        ])
        np.testing.assert_allclose(
            elemento.strain(U, xi, eta), epsilon_exacta,
            rtol=1e-10, atol=1e-12,
            err_msg="LST: no reproduce las deformaciones del campo cuadratico",
        )

    print("LST aislado: interpolacion, Jacobiano, rigidez y campo cuadratico verificados.")


# =============================================================
# PRUEBA 2: PATCH TEST GLOBAL DE CUATRO ELEMENTOS LST
# =============================================================

def run_patch_test_lst(plot=False):
    """Verifica equilibrio y reproduccion de deformaciones constantes."""

    # 2.1. Material elastico lineal en estado plano de tension.
    # E=1000 en unidades consistentes, nu=0.3 y espesor unitario.
    material = LinearElastic(E=1000.0, nu=0.3, plane_state="stress")
    thickness = 1.0

    test_elemento_lst(material)

    # 2.2. Parche cuadrado de 2x2 con centro distorsionado (0.8, 1.2).
    # Cuatro vertices exteriores, cuatro nodos medios exteriores,
    # un nodo central y cuatro nodos medios en aristas internas.
    mesh = Mesh()
    mesh.nodes = [
        Node(0, 0.0, 0.0),
        Node(1, 2.0, 0.0),
        Node(2, 2.0, 2.0),
        Node(3, 0.0, 2.0),
        Node(4, 0.8, 1.2),  # Nodo central (distorsionado)
        Node(5, 1.0, 0.0),  # Arista exterior 0-1
        Node(6, 2.0, 1.0),  # Arista exterior 1-2
        Node(7, 1.0, 2.0),  # Arista exterior 2-3
        Node(8, 0.0, 1.0),  # Arista exterior 3-0
        Node(9, 0.4, 0.6),  # Arista interior 0-4
        Node(10, 1.4, 0.6), # Arista interior 1-4
        Node(11, 1.4, 1.6), # Arista interior 2-4
        Node(12, 0.4, 1.6), # Arista interior 3-4
    ]

    # Conectividad local LST:
    # [vertice_1, vertice_2, vertice_3, medio_12, medio_23, medio_31]
    # Los elementos se orientan en sentido antihorario y comparten
    # tanto los vertices como los nodos medios de aristas comunes.
    elements = [
        LST(0, [0, 1, 4, 5, 10, 9], mesh, material, thickness),
        LST(1, [1, 2, 4, 6, 11, 10], mesh, material, thickness),
        LST(2, [2, 3, 4, 7, 12, 11], mesh, material, thickness),
        LST(3, [3, 0, 4, 8, 9, 12], mesh, material, thickness),
    ]

    # 2.3. Ensamblaje de la rigidez global y ausencia de cargas externas.
    # Trece nodos y dos GDL por nodo -> matriz K de 26x26.
    K = assemble_stiffness(mesh, elements)
    f = np.zeros(mesh.number_of_dofs())
    assert K.shape == (26, 26)
    np.testing.assert_allclose(K, K.T, rtol=1e-12, atol=1e-10)

    # 2.4. Verificacion energetica adicional del ensamblaje.
    # La energia cuadratica global debe coincidir con la suma de
    # las contribuciones elementales para cualquier vector de U.
    U_prueba = np.linspace(-0.002, 0.003, mesh.number_of_dofs())
    energia_global = U_prueba @ K @ U_prueba
    energia_elemental = sum(
        U_prueba[e.global_dofs()] @ e.stiffness_matrix() @ U_prueba[e.global_dofs()]
        for e in elements
    )
    assert np.isclose(energia_global, energia_elemental, rtol=1e-11, atol=1e-12)

    # 2.5. Campo lineal analitico, compatible con tension uniaxial
    # en estado plano de tension:
    # ux=c*x, uy=-nu*c*y
    # epsilon=[c, -nu*c, 0], sigma=[E*c, 0, 0].
    c = 0.001

    def desplazamiento_exacto(node):
        return np.array([c*node.x, -material.nu*c*node.y])

    # Nodos del borde exterior (ambos GDL prescriptos).
    boundary_nodes = [0, 1, 2, 3, 5, 6, 7, 8]
    internal_nodes = [4, 9, 10, 11, 12]
    fixed_dofs = []
    fixed_values = []

    for nid in boundary_nodes:
        node = mesh.node(nid)
        fixed_dofs.extend(mesh.node_global_dofs(nid))
        fixed_values.extend(desplazamiento_exacto(node))

    # 2.6. Resolucion del sistema con desplazamientos prescriptos.
    # Los cinco nodos interiores, equivalentes a diez GDL, son libres.
    U, reacciones = solve_system_direct(K, f, fixed_dofs, fixed_values)
    U = np.asarray(U).reshape(-1)
    assert U.size == mesh.number_of_dofs()

    # 2.7. Comprobacion de los desplazamientos exactos.
    # La verificacion en nodos libres valida la reproduccion del campo
    # por los elementos ensamblados y el sistema de ecuaciones.
    error_max_desplazamientos = 0.0
    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)
        esperado = desplazamiento_exacto(node)
        calculado = U[dofs]
        error_max_desplazamientos = max(
            error_max_desplazamientos, np.max(np.abs(calculado - esperado))
        )
        np.testing.assert_allclose(
            calculado, esperado, rtol=1e-9, atol=1e-10,
            err_msg=f"LST: desplazamientos incorrectos en nodo {node.global_id}",
        )

    # 2.8. Comprobacion de deformaciones y tensiones en los tres
    # puntos de Gauss de cada elemento.
    epsilon_exacta = np.array([c, -material.nu*c, 0.0])
    sigma_exacta = np.array([material.E*c, 0.0, 0.0])

    for element in elements:
        for xi, eta in GAUSS_TRIANGULO:
            np.testing.assert_allclose(
                element.strain(U, xi, eta), epsilon_exacta,
                rtol=1e-9, atol=1e-10,
                err_msg=f"LST: deformaciones incorrectas en elemento {element.global_id}",
            )
            np.testing.assert_allclose(
                element.stress(U, xi, eta), sigma_exacta,
                rtol=1e-9, atol=1e-8,
                err_msg=f"LST: tensiones incorrectas en elemento {element.global_id}",
            )

    # 2.9. Visualizacion opcional: no debe condicionar el patch test.
    if plot:
        from source.output.plot import plot_deformed_mesh
        plot_deformed_mesh(
            mesh, elements, U,
            scale_factor=300.0,
            save_filename="patch_test_lst.png",
        )

    print(f"LST patch: {len(mesh.nodes)} nodos, {len(elements)} elementos, {len(internal_nodes)} nodos libres.")
    print(f"Error maximo de desplazamientos: {error_max_desplazamientos:.3e}")
    print("LST patch: desplazamientos, deformaciones y tensiones verificados.")
    print("\n>>> PATCH TEST LST: SUPERADO EXITOSAMENTE <<<\n")


if __name__ == "__main__":
    run_patch_test_lst(plot=True)
