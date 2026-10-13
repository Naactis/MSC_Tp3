#NOTA: ejecutar usando la siguiente cadena de comandos en la terminal (Marto si queres correrla vos tenes que coregir el path a donde vos tengas alojadas las carpetas)
# cd A:\Git\MSC_Tp3
# python -m source.elements.testeo.testCST



import numpy as np

from source.mesh.node import Node
from source.mesh.mesh import Mesh
from source.elements.cst import CST
from source.materials.hormigon import hormigon
from source.systems.assembly import assemble_stiffness

# 1. Material
material = hormigon 

# 2. Malla
mesh = Mesh()

mesh.nodes = [
    Node(0, 0.0, 0.0),
    Node(1, 2.0, 0.0),
    Node(2, 0.0, 3.0)
]

# 3. Elemento CST
elemento = CST(
    global_id=0,
    node_global_ids=[0, 1, 2],
    mesh=mesh,
    material=material,
    thickness=1.0
)

# 4. Campo de desplazamientos conocido
def desplazamientos(x, y):
    u = 0.001*x + 0.002*y
    v = 0.003*x - 0.001*y
    return u, v

# 5. Vector global de desplazamientos
U = np.zeros(mesh.number_of_dofs())

for node in mesh.nodes:
    dofs = mesh.node_global_dofs(node.global_id)
    U[dofs] = desplazamientos(node.x, node.y)

# 6. Deformaciones obtenidas por FEM
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

# 7. Verificación de la matriz de rigidez
Ke = elemento.stiffness_matrix()

print("Dimensiones de Ke:", Ke.shape)
print("Ke simetrica:", np.allclose(Ke, Ke.T))

assert Ke.shape == (6, 6)
assert np.allclose(Ke, Ke.T)

# 8. Verificacion de movimientos de cuerpo rigido

coords = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [0.0, 3.0]
])

traslacion_x = np.array([1, 0, 1, 0, 1, 0])
traslacion_y = np.array([0, 1, 0, 1, 0, 1])

rotacion_z = np.zeros(6)

for i in range(3):
    x, y = coords[i]
    rotacion_z[2*i] = -y
    rotacion_z[2*i + 1] = x

for movimiento in [traslacion_x, traslacion_y, rotacion_z]:
    assert np.allclose(elemento.B @ movimiento, np.zeros(3))

print("Los tres movimientos de cuerpo rigido verifican.")
print("Estado plano:", material.plane_state)

print("Matriz constitutiva [GPa]:")
print(material.matriz_constitutiva() / 1e9)

print("Tensiones CST [MPa]:")
print(elemento.stress(U) / 1e6)


# 9. Verificacion del ensamblaje global de dos elementos CST

# Malla cuadrada de cuatro nodos, dividida en dos triangulos
mesh_assembly = Mesh()

mesh_assembly.nodes = [
    Node(0, 0.0, 0.0),
    Node(1, 2.0, 0.0),
    Node(2, 2.0, 2.0),
    Node(3, 0.0, 2.0)
]

# Definicion de los dos elementos y sus conectividades
elements = [
    CST(0, [0, 1, 2], mesh_assembly, hormigon, 1.0),
    CST(1, [0, 2, 3], mesh_assembly, hormigon, 1.0)
]

# Ensamblaje de la matriz de rigidez global
K_global = assemble_stiffness(mesh_assembly, elements)

# Verificacion de dimensiones y simetria
print("\nVerificacion del ensamblaje global:")
print("GDL totales:", mesh_assembly.number_of_dofs())
print("Dimensiones de K:", K_global.shape)
print("K simetrica:", np.allclose(K_global, K_global.T))

assert K_global.shape == (8, 8)
assert np.allclose(K_global, K_global.T)

print("Ensamblaje global: dimensiones y simetria verificadas.")


# 10. Verificacion de los modos de cuerpo rigido de la malla ensamblada

n_dofs = mesh_assembly.number_of_dofs()

traslacion_x = np.zeros(n_dofs)
traslacion_y = np.zeros(n_dofs)
rotacion_z = np.zeros(n_dofs)

for node in mesh_assembly.nodes:
    dofs = mesh_assembly.node_global_dofs(node.global_id)

    # Traslacion uniforme en las direcciones x e y
    traslacion_x[dofs[0]] = 1.0
    traslacion_y[dofs[1]] = 1.0

    # Rotacion infinitesimal alrededor del origen
    rotacion_z[dofs[0]] = -node.y
    rotacion_z[dofs[1]] = node.x

# Escala de referencia para evaluar el residuo relativo
escala_K = np.linalg.norm(K_global, ord=np.inf)

for movimiento in [traslacion_x, traslacion_y, rotacion_z]:
    residuo = K_global @ movimiento

    error_relativo = np.linalg.norm(residuo, ord=np.inf) / (
        escala_K * np.linalg.norm(movimiento, ord=np.inf)
    )

    assert error_relativo < 1e-12

print("Modos de cuerpo rigido globales verificados.")
