

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


# =============================================================
# 3. CONFIGURACION Y EJECUCION
# =============================================================

def main():

    # Seleccionar el tipo de elemento:
    # "CST", "LST", "Q4" o "Q8".
    element_type = "Q8"

    # Configuracion geometrica, fisica y de visualizacion.
    config = DamConfig(

        # Archivo de malla correspondiente al elemento
        mesh_path=PROJECT_ROOT / "meshes" / f"{element_type}.mesh",

        # Espesor considerado en el modelo bidimensional [m]
        thickness=1.0,

        # Densidad del agua [kg/m3]
        rho_water=1000.0,

        # Aceleracion de la gravedad [m/s2]
        gravity=9.81,

        # Nivel del agua medido desde la base [m]
        water_level=28.0,

        # Geometria de la presa [m]
        height=30.0,
        base_width=18.0,
        crest_width=3.0,

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
