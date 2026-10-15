
import numpy as np


class CST:
    """
    Elemento triangular lineal de deformacion constante (CST).

    El elemento posee 3 nodos y 2 grados de libertad por nodo
    (desplazamientos u y v), resultando en 6 GDL elementales.

    La matriz de rigidez se obtiene mediante:
    Ke = t * Ae * B.T @ C @ B

    donde B es constante dentro del elemento.
    """

    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids, mesh, material, thickness):

        # Identificacion del elemento y almacenamiento de sus propiedades
        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness

        # ---------------------------------------------------------
        # 1. Geometria del elemento
        # ---------------------------------------------------------

        # Se recuperan los objetos Node mediante sus identificadores globales
        node_1 = mesh.node(node_global_ids[0])
        node_2 = mesh.node(node_global_ids[1])
        node_3 = mesh.node(node_global_ids[2])

        # Coordenadas de los tres nodos del elemento
        x1, y1 = node_1.x, node_1.y
        x2, y2 = node_2.x, node_2.y
        x3, y3 = node_3.x, node_3.y

        # Matriz geometrica de interpolacion:
        # [u1, u2, u3].T = A @ [a1, a2, a3].T
        # para el campo lineal u(x,y) = a1 + a2*x + a3*y
        A = np.array([
            [1.0, x1, y1],
            [1.0, x2, y2],
            [1.0, x3, y3]
        ])

        # El determinante de A es igual al doble del area orientada
        detA = np.linalg.det(A)

        # Se exige orientacion antihoraria y un triangulo no degenerado
        if detA <= 0.0:
            raise ValueError("CST: area no positiva u orientacion incorrecta")

        self.area = detA / 2.0

        # ---------------------------------------------------------
        # 2. Derivadas de las funciones de forma
        # ---------------------------------------------------------

        # Las funciones de forma satisfacen:
        # N(x,y) = [1, x, y] @ A_inv
        A_inv = np.linalg.inv(A)

        # Sus derivadas son las filas 2 y 3 de A_inv.
        # Son constantes porque la interpolacion es lineal.
        self.dN_dx = A_inv[1, :]
        self.dN_dy = A_inv[2, :]

        # ---------------------------------------------------------
        # 3. Matriz cinematica B
        # ---------------------------------------------------------

        # Relacion entre deformaciones y desplazamientos nodales:
        # epsilon = B @ ue
        #
        # epsilon = [epsilon_x, epsilon_y, gamma_xy]
        # ue = [u1, v1, u2, v2, u3, v3]

        B = np.zeros((3, 6))

        # Cada nodo aporta dos columnas: desplazamiento en x y en y
        for i in range(3):
            B[0, 2*i]     = self.dN_dx[i]
            B[1, 2*i + 1] = self.dN_dy[i]

            B[2, 2*i]     = self.dN_dy[i]
            B[2, 2*i + 1] = self.dN_dx[i]

        self.B = B

        # ---------------------------------------------------------
        # 4. Matriz de rigidez elemental
        # ---------------------------------------------------------

        # Matriz constitutiva del material (Plane Stress o Plane Strain)
        C = self.material.matriz_constitutiva()

        # Como B y C son constantes, la integral sobre el elemento
        # se reduce al producto por su area y espesor
        self.Ke = self.thickness * self.area * (self.B.T @ C @ self.B)

    def stiffness_matrix(self):
        """Devuelve la matriz de rigidez elemental de dimension 6x6."""
        return self.Ke

    def global_dofs(self):
        """
        Obtiene los indices globales de los 6 GDL del elemento,
        respetando el orden local [u1, v1, u2, v2, u3, v3].
        """

        dofs = []

        # Cada nodo tiene dos GDL, identificados mediante la clase Mesh
        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    def strain(self, u):
        """
        Calcula las deformaciones constantes del elemento a partir
        del vector global de desplazamientos.

        Retorna [epsilon_x, epsilon_y, gamma_xy].
        """

        # Correspondencia entre GDL locales y globales
        dofs = self.global_dofs()

        # Extraccion del vector de desplazamientos del elemento
        ue = np.zeros(len(dofs))

        for i in range(len(dofs)):
            ue[i] = u[dofs[i]]

        # Relacion cinematica: epsilon = B @ ue
        return self.B @ ue

    def stress(self, u):
        """
        Calcula las tensiones del elemento mediante la ley constitutiva.

        Retorna [sigma_x, sigma_y, tau_xy].
        """

        C = self.material.matriz_constitutiva()

        # Ley de Hooke: sigma = C @ epsilon
        return C @ self.strain(u)
