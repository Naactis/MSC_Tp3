import sys
import os
import numpy as np

# Agregar la carpeta raíz al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.materials.LinearElastic import LinearElastic
from source.systems.assembly import assemble_stiffness
from source.systems.bc import solve_system_direct
from source.elements.q4 import Q4
from source.output.plot import plot_deformed_mesh


# --- 1. Malla simulada de 9 nodos (4 elementos Q4) ---
class MockNode:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class MockMesh:
    def __init__(self):
        # Parche de 2x2 compuesto por 4 elementos cuadriláteros
        # Nodos 0 a 7 en el contorno, Nodo 8 interno distorsionado en (0.8, 1.2)
        self.nodes = {
            0: MockNode(0, 0.0, 0.0),
            1: MockNode(1, 1.0, 0.0),
            2: MockNode(2, 2.0, 0.0),
            3: MockNode(3, 2.0, 1.0),
            4: MockNode(4, 2.0, 2.0),
            5: MockNode(5, 1.0, 2.0),
            6: MockNode(6, 0.0, 2.0),
            7: MockNode(7, 0.0, 1.0),
            8: MockNode(8, 0.8, 1.2)  # Nodo interno (distorsionado)
        }

    def node(self, nid):
        return self.nodes[nid]

    def number_of_dofs(self):
        return len(self.nodes) * 2

    def node_global_dofs(self, nid):
        return [2 * nid, 2 * nid + 1]


# --- 2. Función Principal del Patch Test ---
def run_patch_test_q4():
    mesh = MockMesh()
    mat = LinearElastic(E=1000.0, nu=0.3)
    thickness = 1.0

    # Definir los 4 elementos Q4 recorriendo nodos en sentido antihorario
    elements = [
        Q4(0, [0, 1, 8, 7], mesh, mat, thickness),
        Q4(1, [1, 2, 3, 8], mesh, mat, thickness),
        Q4(2, [8, 3, 4, 5], mesh, mat, thickness),
        Q4(3, [7, 8, 5, 6], mesh, mat, thickness),
    ]

    # Ensamblar matriz de rigidez global K y vector de fuerzas f
    K = assemble_stiffness(mesh, elements)
    f = np.zeros(mesh.number_of_dofs())

    # --- 3. Aplicar Campo Lineal de Desplazamientos ---
    # Campo analítico: u_x = c * x,  u_y = -nu * c * y
    c = 0.001
    fixed_dofs = []
    fixed_values = []

    # Imponer desplazamientos en TODOS los nodos del contorno (0 al 7)
    boundary_nodes = [0, 1, 2, 3, 4, 5, 6, 7]
    for nid in boundary_nodes:
        node = mesh.node(nid)
        ux_val = c * node.x
        uy_val = -mat.nu * c * node.y

        fixed_dofs.extend([2 * nid, 2 * nid + 1])
        fixed_values.extend([ux_val, uy_val])

    # Resolver el sistema (el nodo 8 es el único libre)
    u, R = solve_system_direct(K, f, fixed_dofs, fixed_values)

    #Graficar la malla deformada y guardar la imagen en 'output/patch_test_q4.png'
    plot_deformed_mesh(mesh, elements, u, scale_factor=300.0, save_filename="patch_test_q4.png")
    
    # --- 4. Verificación del Nodo Interno (Nodo 8) ---
    nodo_interno = mesh.node(8)
    ux_esperado = c * nodo_interno.x
    uy_esperado = -mat.nu * c * nodo_interno.y

    ux_calculado = u[2 * 8]
    uy_calculado = u[2 * 8 + 1]

    tolerancia = 1e-8

    print(f"ux Esperado: {ux_esperado:.8f} | ux Calculado: {ux_calculado:.8f}")
    print(f"uy Esperado: {uy_esperado:.8f} | uy Calculado: {uy_calculado:.8f}")

    assert abs(ux_calculado - ux_esperado) < tolerancia, f"Fallo en ux. Exp: {ux_esperado}, Calc: {ux_calculado}"
    assert abs(uy_calculado - uy_esperado) < tolerancia, f"Fallo en uy. Exp: {uy_esperado}, Calc: {uy_calculado}"

    print("\n>>> Patch Test Q4: SUPERADO EXITOSAMENTE <<<\n")


if __name__ == "__main__":
    run_patch_test_q4()