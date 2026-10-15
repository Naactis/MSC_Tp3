import numpy as np


class Q4:
    """
    Elemento cuadrilatero bilineal de cuatro nodos (Q4).

    Cada nodo posee dos grados de libertad: desplazamientos u y v.
    La matriz de rigidez se calcula mediante integracion de Gauss 2x2
    sobre el elemento de referencia isoparametrico.

    Numeracion local:
        4 -------- 3
        |          |
        |          |
        1 -------- 2

    Se requiere una geometria valida con Jacobiano positivo.
    """

    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids, mesh, material, thickness):

        # Identificacion del elemento y almacenamiento de sus propiedades
        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness

        # Se comprueba que el elemento tenga cuatro nodos y espesor positivo.
        if len(node_global_ids) != 4:
            raise ValueError("Q4: se requieren exactamente 4 nodos")

        if thickness <= 0.0:
            raise ValueError("Q4: el espesor debe ser positivo")

        # ---------------------------------------------------------
        # 1. Geometria del elemento
        # ---------------------------------------------------------

        # Coordenadas fisicas de los cuatro nodos.
        # Cada fila contiene las coordenadas (x, y) de un nodo.
        self.coords = np.zeros((4, 2))

        for i in range(4):
            node = mesh.node(node_global_ids[i])
            self.coords[i, 0] = node.x
            self.coords[i, 1] = node.y


        # Para un Q4 bilineal, detJ es una funcion afin de xi y eta.
        # Se exige que sea positivo en los cuatro vertices naturales.
        # Esto evita inversiones del mapeo dentro del elemento.
        for xi, eta in [
            (-1.0, -1.0),
            ( 1.0, -1.0),
            ( 1.0,  1.0),
            (-1.0,  1.0)
        ]:
            self.B_matrix(xi, eta)

        # ---------------------------------------------------------
        # 2. Matriz de rigidez elemental
        # ---------------------------------------------------------

        # Matriz constitutiva del material.
        # Depende del estado plano seleccionado: tension o deformacion.
        C = self.material.matriz_constitutiva()

        # Matriz de rigidez elemental de dimension 8x8.
        self.Ke = np.zeros((8, 8))

        # CAMBIO 3: Comentario corregido sobre la cuadratura.
        # Regla de Gauss 2x2: integracion completa estandar para el Q4.
        # Es exacta para la rigidez de elementos con mapeo afin.
        # Para cuadrilateros distorsionados, la integracion es aproximada.
        gauss_pts = [
            -1.0 / np.sqrt(3.0),
             1.0 / np.sqrt(3.0)
        ]

        weight = 1.0

        # Integracion numerica de la matriz de rigidez:
        #
        # Ke = t * integral(B.T @ C @ B * detJ) dxi deta
        #
        # La integral se aproxima mediante los cuatro puntos de Gauss.
        for xi in gauss_pts:
            for eta in gauss_pts:

                # CAMBIO 4: El Jacobiano y la matriz B se calculan
                # mediante un metodo auxiliar, evitando duplicar codigo.
                B, detJ = self.B_matrix(xi, eta)

                # Contribucion del punto de integracion a Ke.
                self.Ke += (
                    self.thickness
                    * (B.T @ C @ B)
                    * detJ
                    * weight * weight
                )

    # -------------------------------------------------------------
    # Funciones de forma
    # -------------------------------------------------------------

    # Se usan para interpolar cargas de volumen (peso propio).
    def shape_functions(self, xi, eta):
        """
        Evalua las cuatro funciones de forma bilineales
        en las coordenadas naturales (xi, eta).

        Retorna:
            N: vector de cuatro funciones de forma.
        """

        return 0.25 * np.array([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta)
        ])

    # -------------------------------------------------------------
    # 3. Matriz cinematica B y Jacobiano
    # -------------------------------------------------------------

    # CAMBIO 5: Nuevo metodo auxiliar utilizado tanto para calcular
    # la rigidez elemental como para evaluar las deformaciones.
    def B_matrix(self, xi, eta):
        """
        Calcula la matriz cinematica B y el determinante del Jacobiano
        en un punto de coordenadas naturales (xi, eta).

        Retorna:
            B: matriz cinematica de dimension 3x8.
            detJ: determinante de la matriz Jacobiana.
        """

        # Derivadas de las funciones de forma bilineales respecto
        # de las coordenadas naturales xi y eta.
        #
        # N1 = (1-xi)(1-eta)/4
        # N2 = (1+xi)(1-eta)/4
        # N3 = (1+xi)(1+eta)/4
        # N4 = (1-xi)(1+eta)/4

        dN_dxi = 0.25 * np.array([
            -(1.0 - eta),
             (1.0 - eta),
             (1.0 + eta),
            -(1.0 + eta)
        ])

        dN_deta = 0.25 * np.array([
            -(1.0 - xi),
            -(1.0 + xi),
             (1.0 + xi),
             (1.0 - xi)
        ])

        # Jacobiano del mapeo isoparametrico:
        #
        # J = [ dx/dxi    dy/dxi  ]
        #     [ dx/deta   dy/deta ]
        #
        # Esta convencion determina como se transforman las derivadas.

        J = np.zeros((2, 2))

        J[0, 0] = np.dot(dN_dxi, self.coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, self.coords[:, 1])
        J[1, 0] = np.dot(dN_deta, self.coords[:, 0])
        J[1, 1] = np.dot(dN_deta, self.coords[:, 1])

        detJ = np.linalg.det(J)

        # CAMBIO 6: Esta validacion tambien se aplica cuando se
        # calculan deformaciones o tensiones mediante B_matrix().
        if detJ <= 0.0:
            raise ValueError(
                "Q4: Jacobiano no positivo. "
                "Verificar geometria y orientacion de los nodos."
            )

        # Inversa del Jacobiano para transformar las derivadas
        # desde coordenadas naturales a coordenadas fisicas.
        J_inv = np.linalg.inv(J)

        # [dN/dx, dN/dy].T = J_inv @ [dN/dxi, dN/deta].T
        dN_dx = (
            J_inv[0, 0] * dN_dxi
            + J_inv[0, 1] * dN_deta
        )

        dN_dy = (
            J_inv[1, 0] * dN_dxi
            + J_inv[1, 1] * dN_deta
        )

        # Construccion de la matriz cinematica B:
        #
        # epsilon = B @ ue
        #
        # epsilon = [epsilon_x, epsilon_y, gamma_xy]
        # ue = [u1, v1, u2, v2, u3, v3, u4, v4]

        B = np.zeros((3, 8))

        for i in range(4):
            B[0, 2*i]     = dN_dx[i]
            B[1, 2*i + 1] = dN_dy[i]

            B[2, 2*i]     = dN_dy[i]
            B[2, 2*i + 1] = dN_dx[i]

        return B, detJ

    # -------------------------------------------------------------
    # 4. Matriz de rigidez y correspondencia de GDL
    # -------------------------------------------------------------

    def stiffness_matrix(self):
        """Devuelve la matriz de rigidez elemental de dimension 8x8."""
        return self.Ke

    def global_dofs(self):
        """
        Obtiene los indices globales de los ocho GDL del elemento,
        respetando el orden local [u1,v1,u2,v2,u3,v3,u4,v4].
        """

        dofs = []

        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    # -------------------------------------------------------------
    # 5. Calculo de deformaciones
    # -------------------------------------------------------------

    # CAMBIO 7: Se reutiliza B_matrix() y se evita repetir todo el
    # calculo de derivadas, Jacobiano y matriz cinematica.
    def strain(self, u, xi=0.0, eta=0.0):
        """
        Calcula las deformaciones del Q4 en un punto natural.

        Por defecto se evalua en el centro del elemento de referencia
        (xi=0, eta=0), que no necesariamente coincide con el
        centroide geometrico del cuadrilatero fisico.

        Retorna:
            [epsilon_x, epsilon_y, gamma_xy]
        """

        B, _ = self.B_matrix(xi, eta)

        # Se extraen los desplazamientos elementales desde el
        # vector global, respetando la correspondencia de GDL.
        dofs = self.global_dofs()
        ue = np.asarray(u)[dofs]

        # Relacion cinematica
        return B @ ue

    # -------------------------------------------------------------
    # 6. Calculo de tensiones
    # -------------------------------------------------------------

    # CAMBIO 8: Nuevo metodo para obtener las tensiones del elemento.
    def stress(self, u, xi=0.0, eta=0.0):
        """
        Calcula las tensiones en un punto natural del Q4.

        Retorna:
            [sigma_x, sigma_y, tau_xy]
        """

        C = self.material.matriz_constitutiva()

        # Ley de Hooke: sigma = C @ epsilon
        return C @ self.strain(u, xi, eta)