"""
Verificacion del estudio de geometria.

Ejecutar desde la raiz del proyecto:
    python tests/testGeometryStudy.py

Se comprueba:
    1. Las tensiones clasicas de la base (ley trapecial) con casos a mano.
    2. Que el ancho de base minimo devuelva exactamente el FS objetivo.
    3. Que el estudio corra con mallas chicas y que el FS del FEM
       coincida con el analitico para cada geometria.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.materials.LinearElastic import LinearElastic
from source.output.stability import analytic_stability
from source.studies.geometryStudy import (
    GeometryStudyConfig,
    classical_base_stresses,
    minimum_base_for_fs,
    run_geometry_study,
)


MATERIAL = LinearElastic(21e9, 0.2, "strain", rho=2400.0)


def test_tensiones_clasicas():

    # Resultante centrada: tension uniforme N/B en toda la base
    heel, toe = classical_base_stresses(1.0e6, 5.0, 10.0)
    np.testing.assert_allclose([heel, toe], [-1.0e5, -1.0e5])

    # Resultante en el borde del tercio central (e = B/6): talon en cero,
    # pie con el doble de la media
    heel, toe = classical_base_stresses(1.0e6, 5.0 + 10.0 / 6.0, 10.0)
    np.testing.assert_allclose([heel, toe], [0.0, -2.0e5], atol=1e-6)

    print("Geometria: tensiones clasicas de la base verificadas.")


def test_base_minima():

    cfg = GeometryStudyConfig(material=MATERIAL, results_dir=Path("."))

    for target in (1.5, 2.0, 3.0):
        for crest in (3.0, 6.0):
            base = minimum_base_for_fs(target, crest, cfg)
            fs = analytic_stability(
                height=cfg.height, base_width=base, crest_width=crest,
                water_level=cfg.water_level, rho_concrete=MATERIAL.rho,
                rho_water=cfg.rho_water, g=cfg.gravity
            )["FS_overturning"]
            np.testing.assert_allclose(fs, target, rtol=1e-8)

    print("Geometria: ancho de base minimo verificado.")


def test_estudio():

    with tempfile.TemporaryDirectory() as carpeta:
        cfg = GeometryStudyConfig(
            material=MATERIAL,
            results_dir=Path(carpeta),
            base_widths=(14.0, 18.0),
            crest_widths=(3.0,),
            nx=4, ny=6,
            corner_exclusion=2.0,
            target_fs=(2.0,),
            plot=True,
        )

        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            rows, minimum_rows = run_geometry_study(cfg)

        assert len(rows) == 2
        for row in rows:
            np.testing.assert_allclose(row["FS_fem"], row["FS_analytic"], rtol=1e-8)
            assert row["sigma2_far_MPa"] < 0.0           # compresion

        # Mas base -> mas FS
        assert rows[1]["FS_fem"] > rows[0]["FS_fem"]

        # (con una sola cresta no se genera el grafico de eficiencia)
        for name in ("geometryStudy.csv", "minimumBase.csv", "geometryFS.png",
                     "geometryStresses.png"):
            assert (Path(carpeta) / name).is_file(), name

    print("Geometria: estudio parametrico verificado.")


if __name__ == "__main__":
    test_tensiones_clasicas()
    test_base_minima()
    test_estudio()
    print("\n>>> TEST DEL ESTUDIO DE GEOMETRIA: SUPERADO EXITOSAMENTE <<<\n")
