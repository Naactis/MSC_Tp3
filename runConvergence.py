"""
Programa principal del estudio de convergencia FEM.

Ejecuta:
    1. Convergencia CST.
    2. Convergencia LST.
    3. Convergencia Q4.
    4. Convergencia Q8.
    5. Comparacion conjunta de las cuatro formulaciones.

Incluye desplazamientos, tensiones y energia de deformacion.
"""

from pathlib import Path

from source.studies.meshConvergence import (
    ConvergenceConfig,
    run_mesh_convergence,
)

from compareConvergence import main as compare_main


PROJECT_ROOT = Path(__file__).resolve().parent


def main():

    # Tipos de elementos a estudiar.
    element_types = ("CST", "LST", "Q4", "Q8")

    # Refinamientos comunes a todas las formulaciones.
    levels = (
    (3, 5),
    (6, 10),
    (9, 15),
    (12, 20),
    (15, 25),
    )

    # Coordenadas del punto de evaluacion de tensiones.
    stress_point = (2.0, 10.3)

    # ---------------------------------------------------------
    # 1. Ejecutar estudios individuales
    # ---------------------------------------------------------

    for element_type in element_types:

        print("\n" + "=" * 65)
        print(f"INICIANDO CONVERGENCIA: {element_type}")
        print("=" * 65)

        config = ConvergenceConfig(
            element_type=element_type,
            material_mesh_path=(
                PROJECT_ROOT / "meshes" / f"{element_type}.mesh"
            ),
            results_dir=(
                PROJECT_ROOT / "results" / "convergence" / element_type
            ),
            levels=levels,
            stress_point=stress_point,
        )

        run_mesh_convergence(config)

    # ---------------------------------------------------------
    # 2. Comparar las cuatro formulaciones
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("GENERANDO COMPARACION GLOBAL")
    print("=" * 65)

    compare_main()

    print("\n" + "=" * 65)
    print("ESTUDIO DE CONVERGENCIA COMPLETADO")
    print("=" * 65)


if __name__ == "__main__":
    main()