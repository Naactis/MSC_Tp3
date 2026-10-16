"""
Verificacion del formato de malla, del lector y del generador
de mallas de la presa.

Ejecutar desde la raiz del proyecto:
    python tests/testMesh.py

Se comprueba:
    1. Que el generador produzca mallas validas para CST, LST, Q4 y Q8:
       cantidad de nodos y elementos, area total, Jacobiano positivo,
       paramento aguas arriba en x = 0 y base en y = 0.
    2. Que los nodos intermedios (LST, Q8) esten en el punto medio de
       su arista y sean compartidos por los elementos vecinos.
    3. Que escribir y volver a leer una malla no cambie nada.
    4. Que el lector rechace archivos con errores.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.mesh.mesh import Mesh
from source.mesh.generator import generate_dam_mesh, write_dam_mesh
from source.materials.LinearElastic import LinearElastic
from source.problems.GravityDam import create_elements
from source.mesh.metrics import compute_area


H, B, C = 30.0, 18.0, 3.0
AREA_EXACTA = 0.5 * (B + C) * H

MATERIAL = LinearElastic(21e9, 0.2, "strain", rho=2400.0)


def cantidades_esperadas(element_type, nx, ny):
    """Cantidad de nodos y de elementos de una malla nx x ny."""

    vertices = (nx + 1) * (ny + 1)
    aristas_h = nx * (ny + 1)
    aristas_v = (nx + 1) * ny
    diagonales = nx * ny

    if element_type == "Q4":
        return vertices, nx * ny
    if element_type == "Q8":
        return vertices + aristas_h + aristas_v, nx * ny
    if element_type == "CST":
        return vertices, 2 * nx * ny
    # LST
    return vertices + aristas_h + aristas_v + diagonales, 2 * nx * ny


# =============================================================
# 1 y 2. GENERADOR
# =============================================================

def test_generador():

    for element_type in ["CST", "LST", "Q4", "Q8"]:
        for nx, ny in [(1, 1), (3, 5), (4, 7)]:

            mesh = generate_dam_mesh(
                element_type, nx, ny, MATERIAL,
                height=H, base_width=B, crest_width=C
            )

            etiqueta = f"{element_type} {nx}x{ny}"

            # Cantidades
            n_nodos, n_elem = cantidades_esperadas(element_type, nx, ny)
            assert mesh.number_of_nodes() == n_nodos, etiqueta
            assert mesh.number_of_elements() == n_elem, etiqueta

            # No hay nodos duplicados (misma posicion)
            coords = np.array([[n.x, n.y] for n in mesh.nodes])
            assert len(np.unique(np.round(coords, 9), axis=0)) == n_nodos, etiqueta

            # Todos los nodos dentro del trapecio
            ancho = B - (B - C) * coords[:, 1] / H
            assert np.all(coords[:, 0] >= -1e-12), etiqueta
            assert np.all(coords[:, 0] <= ancho + 1e-9), etiqueta

            # Base y paramento aguas arriba exactamente en y=0 y x=0
            assert np.sum(coords[:, 1] == 0.0) >= nx + 1, etiqueta
            assert np.sum(coords[:, 0] == 0.0) >= ny + 1, etiqueta

            # Los elementos se construyen sin error: esto verifica
            # Jacobiano positivo (orientacion antihoraria).
            elements = create_elements(mesh, thickness=1.0)

            # Area total
            np.testing.assert_allclose(
                compute_area(elements), AREA_EXACTA, rtol=1e-12,
                err_msg=f"{etiqueta}: area incorrecta"
            )

            # Nodos intermedios en el punto medio de su arista
            if element_type in ("Q8", "LST"):
                aristas = (
                    [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]
                    if element_type == "Q8"
                    else [(0, 1, 3), (1, 2, 4), (2, 0, 5)]
                )

                for conn in mesh.connectivity:
                    for a, b, m in aristas:
                        np.testing.assert_allclose(
                            coords[conn[m]],
                            0.5 * (coords[conn[a]] + coords[conn[b]]),
                            atol=1e-12,
                            err_msg=f"{etiqueta}: nodo medio fuera de lugar"
                        )

    print("Generador: mallas CST, LST, Q4 y Q8 verificadas.")


# =============================================================
# 3. ESCRITURA Y LECTURA
# =============================================================

def test_escritura_lectura():

    with tempfile.TemporaryDirectory() as carpeta:
        for element_type in ["CST", "LST", "Q4", "Q8"]:

            archivo = Path(carpeta) / f"{element_type}.mesh"

            original = write_dam_mesh(archivo, element_type, 3, 4, MATERIAL)

            leida = Mesh()
            leida.read(archivo)

            assert leida.number_of_nodes() == original.number_of_nodes()
            assert leida.connectivity == original.connectivity
            assert leida.element_types == original.element_types
            assert leida.element_orders == original.element_orders
            assert leida.element_material_ids == original.element_material_ids

            for a, b in zip(leida.nodes, original.nodes):
                assert abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9

            mat = leida.materials[1]
            assert mat.E == MATERIAL.E
            assert mat.nu == MATERIAL.nu
            assert mat.rho == MATERIAL.rho
            assert mat.plane_state == MATERIAL.plane_state

    print("Formato: escritura y lectura consistentes.")


# =============================================================
# 4. ARCHIVOS CON ERRORES
# =============================================================

ARCHIVO_VALIDO = """
MATERIALS 1
1 21e9 0.2 2400 strain

NODES 4
1 0 0
2 1 0
3 1 1
4 0 1

ELEMENTS 1
1 Q4 1 1  1 2 3 4
"""


def debe_fallar(texto, descripcion):
    with tempfile.NamedTemporaryFile("w", suffix=".mesh", delete=False) as f:
        f.write(texto)
        nombre = f.name

    try:
        Mesh().read(nombre)
    except ValueError:
        return
    finally:
        Path(nombre).unlink()

    raise AssertionError(f"El lector acepto un archivo invalido: {descripcion}")


def test_errores():

    # El archivo base es valido
    with tempfile.NamedTemporaryFile("w", suffix=".mesh", delete=False) as f:
        f.write(ARCHIVO_VALIDO)
        nombre = f.name
    mesh = Mesh()
    mesh.read(nombre)
    Path(nombre).unlink()
    assert mesh.element_types == ["Q4"]

    debe_fallar(ARCHIVO_VALIDO.replace("Q4 1 1", "Q4 2 1"), "orden incoherente con el tipo")
    debe_fallar(ARCHIVO_VALIDO.replace("Q4 1 1", "Q4 1 7"), "material inexistente")
    debe_fallar(ARCHIVO_VALIDO.replace("1 2 3 4", "1 2 3"), "cantidad de nodos incorrecta")
    debe_fallar(ARCHIVO_VALIDO.replace("1 2 3 4", "1 2 3 9"), "nodo inexistente")
    debe_fallar(ARCHIVO_VALIDO.replace("Q4 1 1", "Q5 1 1"), "tipo desconocido")
    debe_fallar(ARCHIVO_VALIDO.replace("strain", "plano"), "estado plano invalido")
    debe_fallar(ARCHIVO_VALIDO.replace("NODES 4", "NODES 5"), "cantidad de nodos mal declarada")
    debe_fallar(ARCHIVO_VALIDO.replace("MATERIALS 1\n1 21e9 0.2 2400 strain\n", ""), "sin materiales")

    print("Lector: archivos con errores rechazados.")


if __name__ == "__main__":
    test_generador()
    test_escritura_lectura()
    test_errores()
    print("\n>>> TEST DE MALLAS: SUPERADO EXITOSAMENTE <<<\n")
