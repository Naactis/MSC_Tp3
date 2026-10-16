

"""
Trabajo Practico 3 - Mecanica de Solidos Computacional

Analisis de una presa de gravedad mediante el
Metodo de los Elementos Finitos (MEF).

Elementos disponibles:
    - CST: triangulo lineal de 3 nodos.
    - LST: triangulo cuadratico de 6 nodos.
    - Q4: cuadrilatero bilineal de 4 nodos.
    - Q8: cuadrilatero cuadratico de 8 nodos.

El programa principal configura el problema y delega
su resolucion al modulo gravityDam.py.
"""

import sys
from pathlib import Path


# =============================================================
# 1. CONFIGURACION DE RUTAS
# =============================================================

# Se obtiene automaticamente la raiz del repositorio,
# independientemente de su ubicacion en la computadora.
PROJECT_ROOT = Path(__file__).resolve().parent

# Permite ejecutar el archivo directamente desde VS Code.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================
# 2. IMPORTACION DEL PROBLEMA
# =============================================================

from source.problems.GravityDam import (
    DamConfig,
    run_gravity_dam
)
from source.mesh.generator import write_dam_mesh
from source.materials.hormigon import hormigon


# =============================================================
# 3. CONFIGURACION Y EJECUCION
# =============================================================

def main():

    # ---------------------------------------------------------
    # Malla
    # ---------------------------------------------------------

    # Seleccionar el tipo de elemento:
    # "CST", "LST", "Q4" o "Q8".
    element_type = "Q8"

    # Cantidad de divisiones de la seccion de la presa.
    # nx: horizontal (a lo ancho), ny: vertical (en altura).
    # Q4/Q8: nx*ny elementos. CST/LST: 2*nx*ny elementos.
    nx = 6
    ny = 10

    # True: se genera (o regenera) el archivo de malla con nx, ny.
    # False: se usa un archivo existente, escrito a mano o
    #        generado antes (indicar su nombre en mesh_file).
    generate_mesh = True
    mesh_file = f"presa_{element_type}_{nx}x{ny}.mesh"

    # Geometria de la presa [m]
    height = 30.0
    base_width = 18.0
    crest_width = 3.0

    mesh_path = PROJECT_ROOT / "meshes" / mesh_file

    if generate_mesh:
        # El material (E, nu, rho, estado plano) se escribe
        # en el archivo de malla y se lee desde ahi.
        write_dam_mesh(
            mesh_path,
            element_type,
            nx,
            ny,
            material=hormigon,
            height=height,
            base_width=base_width,
            crest_width=crest_width
        )
        print(f"Malla generada en: {mesh_path}")

    # Configuracion geometrica, fisica y de visualizacion.
    config = DamConfig(

        # Archivo de malla (define elementos, orden y material)
        mesh_path=mesh_path,

        # Espesor considerado en el modelo bidimensional [m]
        thickness=1.0,

        # Densidad del agua [kg/m3]
        rho_water=1000.0,

        # Aceleracion de la gravedad [m/s2]
        gravity=9.81,

        # Nivel del agua medido desde la base [m]
        water_level=28.0,

        # Geometria de la presa [m]
        height=height,
        base_width=base_width,
        crest_width=crest_width,

        # Coordenada horizontal del paramento aguas arriba [m]
        x_upstream=0.0,

        # Parametros de visualizacion
        scale_factor=500.0,
        plot=True
    )

    # Ejecutar el analisis completo.
    results = run_gravity_dam(config)

    return results


# =============================================================
# 4. PUNTO DE ENTRADA
# =============================================================

if __name__ == "__main__":
    main()
