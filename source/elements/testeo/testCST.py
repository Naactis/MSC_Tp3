
import numpy as np

from source.mesh.node import Node
from source.mesh.mesh import Mesh
from source.elements.cst import CST
from source.materials.LinearElastic import LinearElastic


# 1. Material
material = LinearElastic(21e9, 0.20)

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