"""Ejecutar con Play: tests para el mapa de sigma principal deformada."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # El test no abre ventanas.

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.materials.LinearElastic import LinearElastic
from source.mesh.generator import generate_dam_mesh
from source.problems.GravityDam import create_elements
from source.output.principalStressDeformed import (
    sample_principal_stress_deformed, plot_principal_stress_deformed
)


def test_principal_stress_deformed():
    """Verifica geometria deformada y principal 3D para las cuatro familias."""
    scale_factor = 100.0
    nu = 0.2

    for element_type in ("CST", "LST", "Q4", "Q8"):
        for plane_state in ("strain", "stress"):
            material = LinearElastic(21e9, nu, plane_state, rho=2400)
            mesh = generate_dam_mesh(
                element_type, nx=1, ny=1, material=material
            )
            mesh.dofs_per_node = 2
            elements = create_elements(mesh, thickness=1.0)

            # Campo afín con deformaciones compresivas conocidas.
            u = np.zeros(mesh.number_of_dofs())
            for node in mesh.nodes:
                dx, dy = mesh.node_global_dofs(node.global_id)
                u[dx] = -1e-4 * node.x
                u[dy] = -3e-4 * node.y

            fields = sample_principal_stress_deformed(
                mesh, elements, u, scale_factor=scale_factor,
                subdivisions=4, principal_mode="3d"
            )
            for field in fields:
                np.testing.assert_allclose(
                    field["x_deformed"],
                    field["x"] + scale_factor * field["ux"],
                    rtol=0.0, atol=1e-12
                )
                np.testing.assert_allclose(
                    field["y_deformed"],
                    field["y"] + scale_factor * field["uy"],
                    rtol=0.0, atol=1e-12
                )
                np.testing.assert_allclose(
                    field["sigma_plot"],
                    np.maximum(field["sigma_1_xy"], field["sigma_z"])
                )
                if plane_state == "strain":
                    # Recuperar sigma_x + sigma_y desde el campo lineal.
                    # Las tensiones sigma_z deben ser uniformes para esta prueba.
                    assert np.ptp(field["sigma_z"]) < 1e-5
                    assert np.all(field["sigma_z"] > field["sigma_1_xy"])
                else:
                    np.testing.assert_allclose(field["sigma_z"], 0)

            fig = plot_principal_stress_deformed(
                mesh, elements, u, scale_factor=scale_factor,
                subdivisions=4, save_filename=None, show=False
            )
            assert fig.axes

    print("\n>>> TEST PRINCIPAL DEFORMADA CST/LST/Q4/Q8: SUPERADO <<<")


if __name__ == "__main__":
    test_principal_stress_deformed()
