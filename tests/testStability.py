"""
Verificacion del analisis de estabilidad al vuelco.

Ejecutar desde la raiz del proyecto:
    python tests/testStability.py

Se comprueba:
    1. Las formulas analiticas contra valores calculados a mano
       (presa rectangular y presa trapecial del TP).
    2. El calculo de momentos de fuerzas nodales con un caso simple.
    3. Que, para otra geometria y nivel de agua, el modelo FEM con
       CST, LST, Q4 y Q8 reproduzca los momentos analiticos y que el
       momento de las reacciones equilibre a las cargas.
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.mesh.mesh import Mesh
from source.mesh.node import Node
from source.mesh.generator import write_dam_mesh
from source.materials.LinearElastic import LinearElastic
from source.output.stability import analytic_stability, moment_about_point
from source.problems.GravityDam import DamConfig, run_gravity_dam


G = 9.81


# =============================================================
# 1. FORMULAS ANALITICAS
# =============================================================

def test_analitico():

    # Presa rectangular B = C: el centroide esta en B/2.
    r = analytic_stability(
        height=10.0, base_width=4.0, crest_width=4.0, water_level=6.0,
        rho_concrete=2500.0, rho_water=1000.0, g=G
    )
    W = 2500.0 * G * 4.0 * 10.0
    P = 0.5 * 1000.0 * G * 6.0**2

    np.testing.assert_allclose(r["x_centroid"], 2.0)
    np.testing.assert_allclose(r["M_stabilizing"], W * 2.0)
    np.testing.assert_allclose(r["M_overturning"], P * 2.0)
    np.testing.assert_allclose(r["FS_overturning"], (W * 2.0) / (P * 2.0))

    # Presa del TP, valores calculados a mano:
    # A = 315 m2, x_G = 387/63 = 6.142857 m, brazo = 11.857143 m
    # W = 2400*9.81*315 = 7.41636 MN -> M_est = 87.93684 MN.m
    # P = 0.5*1000*9.81*28^2 = 3.84552 MN, h/3 = 9.3333 m -> M_des = 35.89152 MN.m
    r = analytic_stability(
        height=30.0, base_width=18.0, crest_width=3.0, water_level=28.0,
        rho_concrete=2400.0, rho_water=1000.0, g=G
    )
    np.testing.assert_allclose(r["area"], 315.0)
    np.testing.assert_allclose(r["x_centroid"], 387.0 / 63.0)
    np.testing.assert_allclose(r["M_stabilizing"], 7.41636e6 * (18.0 - 387.0 / 63.0))
    np.testing.assert_allclose(r["M_overturning"], 3.84552e6 * 28.0 / 3.0)
    np.testing.assert_allclose(r["FS_overturning"], 2.4501, atol=1e-4)

    print("Estabilidad: formulas analiticas verificadas.")


# =============================================================
# 2. MOMENTO DE FUERZAS NODALES
# =============================================================

def test_momento_nodal():

    mesh = Mesh()
    mesh.nodes = [Node(0, 0.0, 0.0), Node(1, 2.0, 3.0)]

    # Fuerza (Fx, Fy) = (5, 7) en el nodo (2, 3), momento respecto de (1, 1):
    # M = (2-1)*7 - (3-1)*5 = -3
    forces = np.array([0.0, 0.0, 5.0, 7.0])
    np.testing.assert_allclose(moment_about_point(mesh, forces, 1.0, 1.0), -3.0)

    print("Estabilidad: momento de fuerzas nodales verificado.")


# =============================================================
# 3. MODELO FEM COMPLETO
# =============================================================

def test_fem():

    # Geometria y nivel de agua distintos a los del TP
    H, B, C, h = 25.0, 20.0, 5.0, 20.0
    material = LinearElastic(30e9, 0.2, "strain", rho=2500.0)

    analitico = analytic_stability(
        height=H, base_width=B, crest_width=C, water_level=h,
        rho_concrete=2500.0, rho_water=1000.0, g=G
    )

    with tempfile.TemporaryDirectory() as carpeta:
        for element_type in ["CST", "LST", "Q4", "Q8"]:

            archivo = Path(carpeta) / f"{element_type}.mesh"
            write_dam_mesh(
                archivo, element_type, 4, 5, material,
                height=H, base_width=B, crest_width=C
            )

            config = DamConfig(
                mesh_path=archivo, water_level=h, height=H,
                base_width=B, crest_width=C, plot=False
            )

            # run_gravity_dam ya verifica internamente los momentos
            # (report_stability con check=True). Se silencia la salida.
            with contextlib.redirect_stdout(io.StringIO()):
                resultados = run_gravity_dam(config)

            fem = resultados["stability"]["fem"]

            np.testing.assert_allclose(fem["M_body"], analitico["M_stabilizing"], rtol=1e-8)
            np.testing.assert_allclose(fem["M_hydro"], -analitico["M_overturning"], rtol=1e-8)
            np.testing.assert_allclose(
                fem["M_reactions"],
                -(analitico["M_stabilizing"] - analitico["M_overturning"]),
                rtol=1e-8
            )
            np.testing.assert_allclose(fem["FS_overturning"], analitico["FS_overturning"], rtol=1e-8)

            # La resultante vertical en la base equilibra el peso
            np.testing.assert_allclose(fem["Ry"], analitico["weight"], rtol=1e-8)

    print("Estabilidad: momentos FEM de CST, LST, Q4 y Q8 verificados.")


if __name__ == "__main__":
    test_analitico()
    test_momento_nodal()
    test_fem()
    print("\n>>> TEST DE ESTABILIDAD: SUPERADO EXITOSAMENTE <<<\n")
