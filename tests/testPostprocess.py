import sys
from pathlib import Path
import numpy as np

# Permite ejecutar el test desde el boton Play de VS Code.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.mesh.mesh import Mesh
from source.mesh.node import Node
from source.output.postprocess import compute_displacements


def test_displacements():
    """
    Comprueba el calculo de desplazamientos nodales
    con un vector global de solucion conocido.
    """

    # ---------------------------------------------------------
    # 1. Malla ficticia de tres nodos
    # ---------------------------------------------------------

    mesh = Mesh()

    mesh.nodes = [
        Node(0, 0.0, 0.0),
        Node(1, 2.0, 0.0),
        Node(2, 2.0, 3.0)
    ]

    # ---------------------------------------------------------
    # 2. Vector global de desplazamientos conocido
    # ---------------------------------------------------------

    # Nodo 0: ux = 0,      uy = 0
    # Nodo 1: ux = 0.003,  uy = 0.004
    # Nodo 2: ux = -0.002, uy = 0.001
    U = np.array([
         0.0,    0.0,
         0.003,  0.004,
        -0.002,  0.001
    ])

    # ---------------------------------------------------------
    # 3. Ejecutar postprocesamiento
    # ---------------------------------------------------------

    results = compute_displacements(mesh, U)

    # ---------------------------------------------------------
    # 4. Verificar las componentes de desplazamiento
    # ---------------------------------------------------------

    np.testing.assert_allclose(
        results["ux"],
        [0.0, 0.003, -0.002]
    )

    np.testing.assert_allclose(
        results["uy"],
        [0.0, 0.004, 0.001]
    )

    # ---------------------------------------------------------
    # 5. Verificar los modulos
    # ---------------------------------------------------------

    # Nodo 1:
    # sqrt(0.003^2 + 0.004^2) = 0.005 m
    np.testing.assert_allclose(
        results["magnitude"],
        [
            0.0,
            0.005,
            np.sqrt(0.002**2 + 0.001**2)
        ]
    )

    # ---------------------------------------------------------
    # 6. Verificar el desplazamiento maximo
    # ---------------------------------------------------------

    assert results["max_magnitude"]["node_id"] == 1

    np.testing.assert_allclose(
        results["max_magnitude"]["value"],
        0.005
    )

    assert results["max_magnitude"]["x"] == 2.0
    assert results["max_magnitude"]["y"] == 0.0

    print(
        "\n>>> POSTPROCESAMIENTO DE "
        "DESPLAZAMIENTOS: SUPERADO <<<"
    )


if __name__ == "__main__":
    test_displacements()
