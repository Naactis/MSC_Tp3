import numpy as np


class Q8:
    """
    Elemento cuadrilatero serendipito cuadratico de ocho nodos (Q8).

    Cada nodo posee dos grados de libertad: desplazamientos u y v.
    La matriz de rigidez se calcula mediante integracion de Gauss 3x3
    sobre el elemento de referencia isoparametrico.

    Numeracion local:
        4 ---- 7 ---- 3
        |             |
        8             6
        |             |
        1 ---- 5 ---- 2

    Nodos 1 a 4: vertices, en sentido antihorario.
    Nodos 5 a 8: puntos medios de las aristas 1-2, 2-3, 3-4 y 4-1.

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

        # CAMBIO 1: Se comprueba que el elemento tenga ocho nodos
        # distintos y espesor positivo.
        if len(node_global_ids) != 8:
            raise ValueError("Q8: se requieren exactamente 8 nodos")

        if len(set(node_global_ids)) != 8:
            raise ValueError("Q8: los nodos deben ser distintos")

        if thickness <= 0.0:
            raise ValueError("Q8: el espesor debe ser positivo")

        # ---------------------------------------------------------
        # 1. Geometria del elemento
        # ---------------------------------------------------------

        # Coordenadas fisicas de los ocho nodos.
        # Cada fila contiene las coordenadas (x, y) de un nodo.
        self.coords = np.zeros((8, 2))

        for i in range(8):
            node = mesh.node(node_global_ids[i])
            self.coords[i, 0] = node.x
            self.coords[i, 1] = node.y

        # ---------------------------------------------------------
        # 2. Matriz de rigidez elemental
        # ---------------------------------------------------------

        # Matriz constitutiva del material.
        # Depende del estado plano seleccionado: tension o deformacion.
        C = self.material.matriz_constitutiva()

        # Matriz de rigidez elemental de dimension 16x16.
        self.Ke = np.zeros((16, 16))

        # Regla de Gauss 3x3: integracion completa estandar para el Q8.
        # Es exacta para la rigidez de elementos con mapeo afin
        # (nodos medios en el centro de aristas rectas).
        gauss_pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
        gauss_weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

        # Integracion numerica de la matriz de rigidez:
        #
        # Ke = t * integral(B.T @ C @ B * detJ) dxi deta
        #
        # La integral se aproxima mediante los nueve puntos de Gauss.
        # CAMBIO 2: el Jacobiano positivo se verifica en cada punto
        # de Gauss dentro de B_matrix().
        for i, xi in enumerate(gauss_pts):
            for j, eta in enumerate(gauss_pts):

                weight = gauss_weights[i] * gauss_weights[j]

                # CAMBIO 3: El Jacobiano y la matriz B se calculan
                # mediante un metodo auxiliar, evitando duplicar codigo.
                B, detJ = self.B_matrix(xi, eta)

                # Contribucion del punto de integracion a Ke.
                self.Ke += (
                    self.thickness
                    * (B.T @ C @ B)
                    * detJ
                    * weight
                )

    # -------------------------------------------------------------
    # 3. Funciones de forma y sus derivadas
    # -------------------------------------------------------------

    # CAMBIO 4: Nuevo metodo con las funciones de forma.
    # Sirve para interpolar cargas de volumen o de superficie.
    def shape_functions(self, xi, eta):
        """
        Evalua las ocho funciones de forma serendipitas
        en las coordenadas naturales (xi, eta).

        Retorna:
            N: vector de ocho funciones de forma.
        """

        return np.array([
            # Vertices
            0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1),
            0.25 * (1 + xi) * (1 - eta) * ( xi - eta - 1),
            0.25 * (1 + xi) * (1 + eta) * ( xi + eta - 1),
            0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1),

            # Puntos medios de las aristas
            0.5 * (1 - xi**2) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta**2),
            0.5 * (1 - xi**2) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta**2)
        ])

    def _shape_derivatives(self, xi, eta):
        """
        Calcula las derivadas de las ocho funciones de forma
        respecto de las coordenadas naturales xi y eta.

        Retorna:
            dN_dxi: vector de 8 derivadas respecto de xi.
            dN_deta: vector de 8 derivadas respecto de eta.
        """

        dN_dxi = np.zeros(8)
        dN_deta = np.zeros(8)

        # Nodos de los vertices (1 a 4)
        dN_dxi[0] = 0.25 * (1 - eta) * (2*xi + eta)
        dN_deta[0] = 0.25 * (1 - xi) * (xi + 2*eta)

        dN_dxi[1] = 0.25 * (1 - eta) * (2*xi - eta)
        dN_deta[1] = 0.25 * (1 + xi) * (2*eta - xi)

        dN_dxi[2] = 0.25 * (1 + eta) * (2*xi + eta)
        dN_deta[2] = 0.25 * (1 + xi) * (2*eta + xi)

        dN_dxi[3] = 0.25 * (1 + eta) * (2*xi - eta)
        dN_deta[3] = 0.25 * (1 - xi) * (2*eta - xi)

        # Nodos en los puntos medios de las aristas (5 a 8)
        dN_dxi[4] = -xi * (1 - eta)
        dN_deta[4] = -0.5 * (1 - xi**2)

        dN_dxi[5] = 0.5 * (1 - eta**2)
        dN_deta[5] = -eta * (1 + xi)

        dN_dxi[6] = -xi * (1 + eta)
        dN_deta[6] = 0.5 * (1 - xi**2)

        dN_dxi[7] = -0.5 * (1 - eta**2)
        dN_deta[7] = -eta * (1 - xi)

        return dN_dxi, dN_deta

    # -------------------------------------------------------------
    # 4. Matriz cinematica B y Jacobiano
    # -------------------------------------------------------------

    # CAMBIO 5: Nuevo metodo auxiliar utilizado tanto para calcular
    # la rigidez elemental como para evaluar las deformaciones.
    def B_matrix(self, xi, eta):
        """
        Calcula la matriz cinematica B y el determinante del Jacobiano
        en un punto de coordenadas naturales (xi, eta).

        Retorna:
            B: matriz cinematica de dimension 3x16.
            detJ: determinante de la matriz Jacobiana.
        """

        dN_dxi, dN_deta = self._shape_derivatives(xi, eta)

        # Jacobiano del mapeo isoparametrico:
        #
        # J = [ dx/dxi    dy/dxi  ]
        #     [ dx/deta   dy/deta ]

        J = np.zeros((2, 2))

        J[0, 0] = np.dot(dN_dxi, self.coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, self.coords[:, 1])
        J[1, 0] = np.dot(dN_deta, self.coords[:, 0])
        J[1, 1] = np.dot(dN_deta, self.coords[:, 1])

        detJ = np.linalg.det(J)

        # A diferencia del Q4, en el Q8 detJ no es lineal en xi y eta,
        # por lo que no alcanza con revisar los vertices: se verifica
        # en cada punto donde se evalua B.
        if detJ <= 0.0:
            raise ValueError(
                "Q8: Jacobiano no positivo. "
                "Verificar geometria, orientacion y posicion de nodos medios."
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
        # ue = [u1, v1, u2, v2, ..., u8, v8]

        B = np.zeros((3, 16))

        for k in range(8):
            B[0, 2*k]     = dN_dx[k]
            B[1, 2*k + 1] = dN_dy[k]

            B[2, 2*k]     = dN_dy[k]
            B[2, 2*k + 1] = dN_dx[k]

        return B, detJ

    # -------------------------------------------------------------
    # 5. Matriz de rigidez y correspondencia de GDL
    # -------------------------------------------------------------

    def stiffness_matrix(self):
        """Devuelve la matriz de rigidez elemental de dimension 16x16."""
        return self.Ke

    def global_dofs(self):
        """
        Obtiene los indices globales de los dieciseis GDL del elemento,
        respetando el orden local [u1,v1,u2,v2,...,u8,v8].
        """

        dofs = []

        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    # -------------------------------------------------------------
    # 6. Calculo de deformaciones
    # -------------------------------------------------------------

    # CAMBIO 6: Se reutiliza B_matrix() y se evita repetir todo el
    # calculo de derivadas, Jacobiano y matriz cinematica.
    def strain(self, u, xi=0.0, eta=0.0):
        """
        Calcula las deformaciones del Q8 en un punto natural.

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
    # 7. Calculo de tensiones
    # -------------------------------------------------------------

    # CAMBIO 7: Nuevo metodo para obtener las tensiones del elemento.
    def stress(self, u, xi=0.0, eta=0.0):
        """
        Calcula las tensiones en un punto natural del Q8.

        Retorna:
            [sigma_x, sigma_y, tau_xy]
        """

        C = self.material.matriz_constitutiva()

        # Ley de Hooke: sigma = C @ epsilon
        return C @ self.strain(u, xi, eta)