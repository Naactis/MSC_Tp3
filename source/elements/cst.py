
import numpy as np


class CST:
    """
    Elemento triangular lineal de deformacion constante (CST).

    Posee 3 nodos y 2 grados de libertad por nodo:
        ue = [u1,v1,u2,v2,u3,v3]

    Matriz de rigidez:
        Ke = t * Ae * B.T @ C @ B

    La matriz B y el Jacobiano son constantes.

    Numeracion local:

            3
           / \
          /   \
         1-----2

    Coordenadas naturales:
        Nodo 1: (0,0)
        Nodo 2: (1,0)
        Nodo 3: (0,1)
    """

    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids,
                 mesh, material, thickness):

        # ---------------------------------------------------------
        # 1. Propiedades del elemento
        # ---------------------------------------------------------

        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness

        # Validacion de los datos de entrada
        if len(node_global_ids) != 3:
            raise ValueError(
                "CST: se requieren exactamente 3 nodos"
            )

        if len(set(node_global_ids)) != 3:
            raise ValueError(
                "CST: los nodos deben ser distintos"
            )

        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                "CST: el espesor debe ser positivo y finito"
            )

        # ---------------------------------------------------------
        # 2. Geometria del elemento
        # ---------------------------------------------------------

        # Coordenadas fisicas de los tres nodos.
        # Cada fila almacena [x_i, y_i].
        self.coords = np.zeros((3, 2))

        for i in range(3):
            node = self.mesh.node(self.node_global_ids[i])

            self.coords[i, 0] = node.x
            self.coords[i, 1] = node.y

        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        x3, y3 = self.coords[2]

        # Matriz geometrica utilizada para determinar
        # los coeficientes de interpolacion lineal.
        A = np.array([
            [1.0, x1, y1],
            [1.0, x2, y2],
            [1.0, x3, y3]
        ])

        # El determinante es dos veces el area orientada.
        detA = np.linalg.det(A)

        if not np.isfinite(detA) or detA <= 0.0:
            raise ValueError(
                "CST: area no positiva o no finita. "
                "Verificar geometria y orden nodal."
            )

        self.area = detA / 2.0

        # CAMBIO:
        # Guardamos el determinante del Jacobiano.
        # Es constante para el triangulo lineal.
        self.detJ = detA

        # ---------------------------------------------------------
        # 3. Derivadas de las funciones de forma
        # ---------------------------------------------------------

        # Las funciones de forma lineales satisfacen:
        #
        # N(x,y) = [1,x,y] @ A_inv
        #
        # Sus derivadas son constantes.
        A_inv = np.linalg.inv(A)

        self.dN_dx = A_inv[1, :]
        self.dN_dy = A_inv[2, :]

        # ---------------------------------------------------------
        # 4. Matriz cinematica B
        # ---------------------------------------------------------

        # epsilon = B @ ue
        #
        # epsilon = [epsilon_x, epsilon_y, gamma_xy]
        # ue = [u1,v1,u2,v2,u3,v3]

        self.B = np.zeros((3, 6))

        for i in range(3):

            self.B[0, 2*i] = self.dN_dx[i]

            self.B[1, 2*i + 1] = self.dN_dy[i]

            self.B[2, 2*i] = self.dN_dy[i]

            self.B[2, 2*i + 1] = self.dN_dx[i]

        # ---------------------------------------------------------
        # 5. Matriz de rigidez elemental
        # ---------------------------------------------------------

        C = self.material.matriz_constitutiva()

        # La matriz B es constante, por lo tanto
        # la integracion de rigidez es analitica.
        self.Ke = (
            self.thickness
            * self.area
            * (self.B.T @ C @ self.B)
        )

    # -------------------------------------------------------------
    # 6. Funciones de forma
    # -------------------------------------------------------------

    def shape_functions(self, xi, eta):
        """
        Evalua las tres funciones de forma del CST
        en las coordenadas naturales (xi, eta).

        Retorna:
            N: arreglo de dimension 3.
        """

        N1 = 1.0 - xi - eta
        N2 = xi
        N3 = eta

        return np.array([N1, N2, N3])

    # -------------------------------------------------------------
    # 7. Matriz cinematica B
    # -------------------------------------------------------------

    def B_matrix(self, xi=0.0, eta=0.0):
        """
        Devuelve la matriz cinematica B y el determinante
        del Jacobiano.

        Se mantienen los argumentos xi y eta para utilizar
        la misma interfaz que LST, Q4 y Q8.

        Como el CST tiene deformacion constante, los
        resultados son independientes del punto evaluado.

        Retorna:
            B: matriz cinematica 3x6.
            detJ: determinante del Jacobiano.
        """

        return self.B, self.detJ

    # -------------------------------------------------------------
    # 8. Matriz de rigidez elemental
    # -------------------------------------------------------------

    def stiffness_matrix(self):
        """
        Devuelve la matriz de rigidez elemental 6x6.
        """

        return self.Ke

    # -------------------------------------------------------------
    # 9. Correspondencia de grados de libertad
    # -------------------------------------------------------------

    def global_dofs(self):
        """
        Devuelve los indices globales de los seis GDL.

        Orden local:
            [u1,v1,u2,v2,u3,v3]
        """

        dofs = []

        for node_global_id in self.node_global_ids:

            for dof in self.mesh.node_global_dofs(node_global_id):

                dofs.append(dof)

        return dofs

    # -------------------------------------------------------------
    # 10. Calculo de deformaciones
    # -------------------------------------------------------------

    def strain(self, u, xi=0.0, eta=0.0):
        """
        Calcula las deformaciones del CST.

        Parametros:
            u: vector global de desplazamientos.
            xi, eta: coordenadas naturales opcionales.

        Las deformaciones son constantes, por lo que
        no dependen de xi ni de eta.

        Retorna:
            [epsilon_x, epsilon_y, gamma_xy]
        """

        # Recuperar la matriz cinematica
        B, _ = self.B_matrix(xi, eta)

        # Obtener los GDL globales del elemento
        dofs = self.global_dofs()

        # Extraer los desplazamientos elementales
        ue = np.asarray(u).reshape(-1)[dofs]

        # Relacion deformacion-desplazamiento
        return B @ ue

    # -------------------------------------------------------------
    # 11. Calculo de tensiones
    # -------------------------------------------------------------

    def stress(self, u, xi=0.0, eta=0.0):
        """
        Calcula las tensiones constantes del CST.

        Parametros:
            u: vector global de desplazamientos.
            xi, eta: coordenadas naturales opcionales.

        Retorna:
            [sigma_x, sigma_y, tau_xy]
        """

        # Matriz constitutiva del material
        C = self.material.matriz_constitutiva()

        # Ley de Hooke: sigma = C @ epsilon
        return C @ self.strain(u, xi, eta)
