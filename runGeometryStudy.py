"""
Programa principal del estudio de variaciones de geometria.

Ejecutar desde la raiz del proyecto:
    python runGeometryStudy.py

Para cada combinacion de ancho de base y de cresta resuelve la presa
con FEM y compara:
    - el factor de seguridad al vuelco (y el area de hormigon),
    - la compresion maxima y la traccion maxima lejos de las esquinas,
    - las tensiones clasicas en la base (ley trapecial).

Resultados en results/geometry/:
    geometryStudy.csv, minimumBase.csv,
    geometryFS.png, geometryEfficiency.png, geometryStresses.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.materials.hormigon import hormigon
from source.studies.geometryStudy import GeometryStudyConfig, run_geometry_study


def main():

    config = GeometryStudyConfig(
        material=hormigon,
        results_dir=PROJECT_ROOT / "results" / "geometry",

        # Anchos a estudiar [m]. La presa original es B = 18, C = 3.
        base_widths=(12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0),
        crest_widths=(3.0, 6.0, 9.0),

        # Malla recomendada por el estudio de convergencia
        element_type="Q8",
        nx=12,
        ny=20,

        # Las tensiones a menos de esta distancia del talon y del pie
        # no se usan: alli la solucion es singular y no converge.
        corner_exclusion=3.0,

        target_fs=(1.5, 2.0),
    )

    run_geometry_study(config)


if __name__ == "__main__":
    main()
