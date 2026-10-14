import sys
import os
import numpy as np

# Agregar la carpeta raíz al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.materials.LinearElastic import LinearElastic
from source.systems.assembly import assemble_stiffness
from source.systems.bc import solve_system_direct
from source.elements.q8 import Q8
from source.output.plot import plot_deformed_mesh

class MockNode:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class MockMesh:
    def __init__(self):
        # Parche 2x2 con 21 nodos continuos (IDs 0 a 20):
        # Nodos 0 a 8: Esquinas (Nodo 8 interno distorsionado)
        # Nodos 9 a 16: Nodos medios del contorno exterior
        # Nodos 17 a 20: Nodos medios de bordes internos
        self.nodes = {
            # Nodos de esquina
            0: MockNode(0, 0.0, 0.0),
            1: MockNode(1, 1.0, 0.0),
            2: MockNode(2, 2.0, 0.0),
            3: MockNode(3, 2.0, 1.0),
            4: MockNode(4, 2.0, 2.0),
            5: MockNode(5, 1.0, 2.0),
            6: MockNode(6, 0.0, 2.0),
            7: MockNode(7, 0.0, 1.0),
            8: MockNode(8, 0.8, 1.2),  # Esquina interna (distorsionada)

            # Nodos medios del contorno exterior
            9:  MockNode(9,  0.5, 0.0),  # Borde (0, 1)
            10: MockNode(10, 1.5, 0.0),  # Borde (1, 2)
            11: MockNode(11, 2.0, 0.5),  # Borde (2, 3)
            12: MockNode(12, 2.0, 1.5),  # Borde (3, 4)
            13: MockNode(13, 1.5, 2.0),  # Borde (4, 5)
            14: MockNode(14, 0.5, 2.0),  # Borde (5, 6)
            15: MockNode(15, 0.0, 1.5),  # Borde (6, 7)
            16: MockNode(16, 0.0, 0.5),  # Borde (7, 0)

            # Nodos medios internos
            17: MockNode(17, 0.9, 0.6),  # Arista (1, 8)
            18: MockNode(18, 1.4, 1.1),  # Arista (3, 8)
            19: MockNode(19, 0.9, 1.6),  # Arista (5, 8)
            20: MockNode(20, 0.4, 1.1),  # Arista (7, 8)
        }

    def node(self, nid):
        return self.nodes[nid]

    def number_of_dofs(self):
        return len(self.nodes) * 2

    def node_global_dofs(self, nid):
        return [2 * nid, 2 * nid + 1]


def run_patch_test_q8():
    mesh = MockMesh()
    mat = LinearElastic(E=1000.0, nu=0.3)
    thickness = 1.0

    # Formato Q8: [4 esquinas antihorario, 4 nodos medios antihorario]
    elements = [
        Q8(0, [0, 1, 8, 7, 9, 17, 20, 16], mesh, mat, thickness),
        Q8(1, [1, 2, 3, 8, 10, 11, 18, 17], mesh, mat, thickness),
        Q8(2, [8, 3, 4, 5, 18, 12, 13, 19], mesh, mat, thickness),
        Q8(3, [7, 8, 5, 6, 20, 19, 14, 15], mesh, mat, thickness),
    ]

    K = assemble_stiffness(mesh, elements)
    f = np.zeros(mesh.number_of_dofs())

    # Nodos del contorno exterior (restringidos)
    boundary_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
    # Nodos internos libres: [8, 17, 18, 19, 20]

    c = 0.001
    fixed_dofs = []
    fixed_values = []

    for nid in boundary_nodes:
        node = mesh.node(nid)
        ux_val = c * node.x
        uy_val = -mat.nu * c * node.y

        fixed_dofs.extend([2 * nid, 2 * nid + 1])
        fixed_values.extend([ux_val, uy_val])

    # Resolver el sistema para los nodos internos
    u, R = solve_system_direct(K, f, fixed_dofs, fixed_values)

    plot_deformed_mesh(mesh, elements, u, scale_factor=300.0, save_filename="patch_test_q8.png")
    # Verificación en los 5 nodos internos
    internal_nodes = [8, 17, 18, 19, 20]
    tolerancia = 1e-8

    for nid in internal_nodes:
        node = mesh.node(nid)
        ux_esp = c * node.x
        uy_esp = -mat.nu * c * node.y

        ux_calc = u[2 * nid]
        uy_calc = u[2 * nid + 1]

        assert abs(ux_calc - ux_esp) < tolerancia, f"Fallo Nodo {nid} ux. Exp: {ux_esp}, Calc: {ux_calc}"
        assert abs(uy_calc - uy_esp) < tolerancia, f"Fallo Nodo {nid} uy. Exp: {uy_esp}, Calc: {uy_calc}"

    print("\n>>> Patch Test Q8: SUPERADO EXITOSAMENTE <<<\n")


if __name__ == "__main__":
    run_patch_test_q8()