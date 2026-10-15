
"""
Pruebas de verificacion del elemento CST y del ensamblaje de rigidez.

Ejecutar desde la raiz del proyecto (Marto para tu compu cambia el path a donde sea que tengas alojado esto): 
    cd A:\\Git\\MSC_Tp3
    python -m source.elements.testeo.testCST

Las pruebas comprueban:
    1. Reproduccion de un campo de deformaciones constantes.
    2. Dimensiones y simetria de la matriz de rigidez elemental.
    3. Ausencia de deformaciones ante movimientos de cuerpo rigido.
    4. Dimensiones y simetria de la matriz de rigidez global.
    5. Modos de cuerpo rigido de la malla ensamblada.

"""

import numpy as np

from source.mesh.node import Node
from source.mesh.mesh import Mesh
from source.elements.cst import CST
from source.materials.hormigon import hormigon
from source.systems.assembly import assemble_stiffness


# ============================================================
# PRUEBA 1: COMPORTAMIENTO DE UN ELEMENTO CST AISLADO
# ============================================================

# 1. Material utilizado en las verificaciones
material = hormigon

# 2. Malla de prueba con un unico triangulo de orientacion antihoraria
mesh = Mesh()

mesh.nodes = [
    Node(0, 0.0, 0.0),
    Node(1, 2.0, 0.0),
    Node(2, 0.0, 3.0)
]

# 3. Construccion del elemento CST.
# El constructor calcula su geometria, matriz B y rigidez elemental Ke.
elemento = CST(
    global_id=0,
    node_global_ids=[0, 1, 2],
    mesh=mesh,
    material=material,
    thickness=1.0
)


# 4. Campo analitico de desplazamientos lineales.
# El CST debe reproducir exactamente este tipo de interpolacion.
# Sus derivadas producen deformaciones constantes conocidas.
def desplazamientos(x, y):
    u = 0.001*x + 0.002*y
    v = 0.003*x - 0.001*y
    return u, v


# 5. Construccion del vector global de desplazamientos U.
# Se evalua el campo analitico en cada nodo y se almacenan sus
# componentes segun la numeracion global de GDL.
U = np.zeros(mesh.number_of_dofs())

for node in mesh.nodes:
    dofs = mesh.node_global_dofs(node.global_id)
    U[dofs] = desplazamientos(node.x, node.y)


# 6. Verificacion de deformaciones.
# El elemento calcula epsilon = B @ Ue.
# Se compara contra las derivadas analiticas del campo impuesto.
epsilon_fem = elemento.strain(U)

epsilon_exacta = np.array([
    0.001,
    -0.001,
    0.005
])

print("Deformaciones FEM:")
print(epsilon_fem)

print("Deformaciones exactas:")
print(epsilon_exacta)

print("Error:")
print(epsilon_fem - epsilon_exacta)

assert np.allclose(epsilon_fem, epsilon_exacta)


# 7. Verificacion de la matriz de rigidez elemental.
# Ke debe tener dimension 6x6 y ser simetrica para elasticidad lineal.
Ke = elemento.stiffness_matrix()

print("Dimensiones de Ke:", Ke.shape)
print("Ke simetrica:", np.allclose(Ke, Ke.T))

assert Ke.shape == (6, 6)
assert np.allclose(Ke, Ke.T)


# 8. Verificacion de los movimientos de cuerpo rigido.
# En un continuo 2D existen tres: dos traslaciones y una rotacion.
# Ninguno debe generar deformaciones: B @ U_rig = 0.

coords = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [0.0, 3.0]
])

traslacion_x = np.array([1, 0, 1, 0, 1, 0])
traslacion_y = np.array([0, 1, 0, 1, 0, 1])

rotacion_z = np.zeros(6)

# Construccion del campo de rotacion infinitesimal:
# u = -y, v = x
for i in range(3):
    x, y = coords[i]
    rotacion_z[2*i] = -y
    rotacion_z[2*i + 1] = x

# Comprobacion de deformaciones nulas para los tres movimientos
for movimiento in [traslacion_x, traslacion_y, rotacion_z]:
    assert np.allclose(elemento.B @ movimiento, np.zeros(3))

print("Los tres movimientos de cuerpo rigido verifican.")


# 9. Inspeccion de la matriz constitutiva y las tensiones.
# Se utiliza sigma = C @ epsilon.
# Conversiones: Pa a GPa (1e9) y Pa a MPa (1e6).
print("Estado plano:", material.plane_state)

print("Matriz constitutiva [GPa]:")
print(material.matriz_constitutiva() / 1e9)

print("Tensiones CST [MPa]:")
print(elemento.stress(U) / 1e6)


# ============================================================
# PRUEBA 2: ENSAMBLAJE GLOBAL DE DOS ELEMENTOS CST
# ============================================================

# 10. Malla cuadrada de cuatro nodos, dividida en dos triangulos.
# Los elementos comparten los nodos 0 y 2.
mesh_assembly = Mesh()

mesh_assembly.nodes = [
    Node(0, 0.0, 0.0),
    Node(1, 2.0, 0.0),
    Node(2, 2.0, 2.0),
    Node(3, 0.0, 2.0)
]

# 11. Construccion de los dos CST y sus conectividades
elements = [
    CST(0, [0, 1, 2], mesh_assembly, material, 1.0),
    CST(1, [0, 2, 3], mesh_assembly, material, 1.0)
]

# 12. Ensamblaje de las matrices elementales en la matriz global K
K_global = assemble_stiffness(mesh_assembly, elements)

# 13. Verificacion de dimensiones y simetria.
# Cuatro nodos con dos GDL por nodo producen una matriz 8x8.
print("\nVerificacion del ensamblaje global:")
print("GDL totales:", mesh_assembly.number_of_dofs())
print("Dimensiones de K:", K_global.shape)
print("K simetrica:", np.allclose(K_global, K_global.T))

assert K_global.shape == (8, 8)
assert np.allclose(K_global, K_global.T)

print("Ensamblaje global: dimensiones y simetria verificadas.")


# 14. Verificacion de los modos de cuerpo rigido globales.
# Sin condiciones de borde, la matriz global debe satisfacer:
# K @ U_rig = 0 para las dos traslaciones y la rotacion.

n_dofs = mesh_assembly.number_of_dofs()

traslacion_x = np.zeros(n_dofs)
traslacion_y = np.zeros(n_dofs)
rotacion_z = np.zeros(n_dofs)

for node in mesh_assembly.nodes:
    dofs = mesh_assembly.node_global_dofs(node.global_id)

    # Traslaciones uniformes en x e y
    traslacion_x[dofs[0]] = 1.0
    traslacion_y[dofs[1]] = 1.0

    # Rotacion infinitesimal alrededor del origen: u = -y, v = x
    rotacion_z[dofs[0]] = -node.y
    rotacion_z[dofs[1]] = node.x

# Escala de rigidez utilizada para calcular los residuos relativos
escala_K = np.linalg.norm(K_global, ord=np.inf)

for movimiento in [traslacion_x, traslacion_y, rotacion_z]:
    residuo = K_global @ movimiento

    # Se normaliza el residuo para evaluar el error numerico
    error_relativo = np.linalg.norm(residuo, ord=np.inf) / (
        escala_K * np.linalg.norm(movimiento, ord=np.inf)
    )

    assert error_relativo < 1e-12

print("Modos de cuerpo rigido globales verificados.")
