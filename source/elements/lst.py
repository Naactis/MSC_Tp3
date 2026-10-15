
import numpy as np


class LST:
    """
    Elemento triangular cuadratico de seis nodos (LST).

    Posee dos grados de libertad por nodo: desplazamientos u y v.
    Utiliza funciones de forma cuadraticas y permite representar
    deformaciones lineales cuando el mapeo geometrico es afin.

    Numeracion local:

                3
               / \
              6   5
             /     \
            1---4---2

    Nodos 1, 2 y 3: vertices.
    Nodo 4: arista 1-2.
    Nodo 5: arista 2-3.
    Nodo 6: arista 3-1.

    La matriz de rigidez se calcula mediante cuadratura
    triangular de Gauss de tres puntos.
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
        # 1. Validacion de los datos de entrada
        # ---------------------------------------------------------

        # El LST posee seis nodos, todos con identificadores distintos
        if len(node_global_ids) != 6:
            raise ValueError("LST: se requieren exactamente 6 nodos")

        if len(set(node_global_ids)) != 6:
            raise ValueError("LST: los nodos deben ser distintos")

        # Se exige un espesor positivo y finito
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("LST: el espesor debe ser positivo y finito")

        # ---------------------------------------------------------
        # 2. Geometria del elemento
        # ---------------------------------------------------------

        # Se almacenan las coordenadas fisicas de los seis nodos.
        # Cada fila contiene las coordenadas (x, y) de un nodo,
        # respetando la numeracion local del LST.
        self.coords = np.zeros((6, 2))

        for i in range(6):
            node = self.mesh.node(self.node_global_ids[i])

            self.coords[i, 0] = node.x
            self.coords[i, 1] = node.y

        # ---------------------------------------------------------
        # 3. Matriz constitutiva y rigidez elemental
        # ---------------------------------------------------------

        # Matriz constitutiva del material.
        # Depende del estado plano seleccionado: tension o deformacion.
        C = self.material.matriz_constitutiva()

        # Se inicializa la matriz de rigidez elemental.
        # El LST posee 6 nodos con 2 GDL por nodo: Ke es de 12x12.
        self.Ke = np.zeros((12, 12))

        # Cuadratura triangular de Gauss de tres puntos.
        # Integra exactamente polinomios de grado total <= 2
        # sobre el triangulo de referencia de area 1/2.
        gauss_pts = [
            (1.0/6.0, 1.0/6.0),
            (2.0/3.0, 1.0/6.0),
            (1.0/6.0, 2.0/3.0)
        ]

        # Los pesos suman 1/2, el area del triangulo natural.
        gauss_weights = [1.0/6.0] * 3

        # Integracion numerica de la matriz de rigidez:
        #
        # Ke = t * integral(B.T @ C @ B * detJ) dxi deta
        #
        # Se acumula la contribucion de cada punto de Gauss.
        for (xi, eta), weight in zip(gauss_pts, gauss_weights):

            # Matriz cinematica y determinante del Jacobiano
            B, detJ = self.B_matrix(xi, eta)

            # Contribucion del punto de integracion a Ke
            self.Ke += (
                self.thickness
                * (B.T @ C @ B)
                * detJ
                * weight
            )

    # -------------------------------------------------------------
    # 4. Funciones de forma cuadraticas
    # -------------------------------------------------------------

    def shape_functions(self, xi, eta):
        """
        Evalua las seis funciones de forma cuadraticas del LST
        en las coordenadas naturales (xi, eta).

        Retorna:
            N: vector de seis funciones de forma.
        """

        # Coordenadas de area del triangulo de referencia
        # L1 + L2 + L3 = 1
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta

        # Funciones de forma de los tres vertices
        N1 = L1 * (2.0*L1 - 1.0)
        N2 = L2 * (2.0*L2 - 1.0)
        N3 = L3 * (2.0*L3 - 1.0)

        # Funciones de forma de los tres nodos intermedios
        N4 = 4.0 * L1 * L2
        N5 = 4.0 * L2 * L3
        N6 = 4.0 * L3 * L1

        return np.array([N1, N2, N3, N4, N5, N6])

    # -------------------------------------------------------------
    # 5. Derivadas de las funciones de forma
    # -------------------------------------------------------------

    def _shape_derivatives(self, xi, eta):
        """
        Calcula las derivadas de las funciones de forma respecto
        de las coordenadas naturales xi y eta.

        Retorna:
            dN_dxi: vector de 6 derivadas respecto de xi.
            dN_deta: vector de 6 derivadas respecto de eta.
        """

        # Coordenadas de area
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta

        # Derivadas respecto de xi
        dN_dxi = np.array([
            1.0 - 4.0*L1,
            4.0*L2 - 1.0,
            0.0,
            4.0*(L1 - L2),
            4.0*L3,
            -4.0*L3
        ])

        # Derivadas respecto de eta
        dN_deta = np.array([
            1.0 - 4.0*L1,
            0.0,
            4.0*L3 - 1.0,
            -4.0*L2,
            4.0*L2,
            4.0*(L1 - L3)
        ])

        return dN_dxi, dN_deta

    # -------------------------------------------------------------
    # 6. Jacobiano del mapeo isoparametrico
    # -------------------------------------------------------------

    def _jacobian(self, xi, eta):
        """
        Calcula el Jacobiano del mapeo del triangulo natural
        al triangulo fisico, junto con su determinante.

        Retorna:
            J: matriz Jacobiana de dimension 2x2.
            detJ: determinante del Jacobiano.
        """

        # Derivadas de las funciones de forma en coordenadas naturales
        dN_dxi, dN_deta = self._shape_derivatives(xi, eta)

        # Jacobiano:
        #
        # J = [ dx/dxi    dy/dxi  ]
        #     [ dx/deta   dy/deta ]
        #
        # Las derivadas geometricas se calculan mediante
        # las coordenadas nodales y las derivadas de N.
        J = np.zeros((2, 2))

        J[0, 0] = np.dot(dN_dxi, self.coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, self.coords[:, 1])

        J[1, 0] = np.dot(dN_deta, self.coords[:, 0])
        J[1, 1] = np.dot(dN_deta, self.coords[:, 1])

        # Factor de transformacion de areas entre ambos dominios
        detJ = np.linalg.det(J)

        # Se rechazan puntos con Jacobiano no positivo o no finito.
        # Esta comprobacion es local al punto evaluado.
        if not np.isfinite(detJ) or detJ <= 0.0:
            raise ValueError(
                "LST: Jacobiano no positivo o no finito. "
                "Verificar geometria y orden nodal."
            )

        return J, detJ

    # -------------------------------------------------------------
    # 7. Matriz cinematica B
    # -------------------------------------------------------------

    def B_matrix(self, xi, eta):
        """
        Calcula la matriz cinematica B y el determinante del
        Jacobiano en un punto natural (xi, eta).

        Retorna:
            B: matriz cinematica de dimension 3x12.
            detJ: determinante del Jacobiano.
        """

        # Derivadas de las funciones de forma respecto
        # de las coordenadas naturales
        dN_dxi, dN_deta = self._shape_derivatives(xi, eta)

        # Jacobiano del mapeo isoparametrico
        J, detJ = self._jacobian(xi, eta)

        # Inversa del Jacobiano para transformar las derivadas
        J_inv = np.linalg.inv(J)

        # Transformacion de derivadas a coordenadas fisicas:
        #
        # [dN/dx]   [       ] [dN/dxi ]
        # [dN/dy] = [ J_inv ] [dN/deta]

        dN_dx = (
            J_inv[0, 0]*dN_dxi
            + J_inv[0, 1]*dN_deta
        )

        dN_dy = (
            J_inv[1, 0]*dN_dxi
            + J_inv[1, 1]*dN_deta
        )

        # Matriz cinematica:
        #
        # epsilon = B @ ue
        #
        # epsilon = [epsilon_x, epsilon_y, gamma_xy]
        # ue = [u1,v1,u2,v2,...,u6,v6]

        B = np.zeros((3, 12))

        # Cada nodo aporta dos columnas:
        # una para su desplazamiento u y otra para v.
        for i in range(6):
            B[0, 2*i]     = dN_dx[i]
            B[1, 2*i + 1] = dN_dy[i]

            B[2, 2*i]     = dN_dy[i]
            B[2, 2*i + 1] = dN_dx[i]

        return B, detJ

    # -------------------------------------------------------------
    # 8. Matriz de rigidez elemental
    # -------------------------------------------------------------

    def stiffness_matrix(self):
        """
        Devuelve la matriz de rigidez elemental del LST.

        Retorna:
            Ke: matriz de rigidez de dimension 12x12.
        """
        return self.Ke

    # -------------------------------------------------------------
    # 9. Correspondencia de grados de libertad
    # -------------------------------------------------------------

    def global_dofs(self):
        """
        Obtiene los indices globales de los doce GDL del elemento,
        respetando el orden local [u1,v1,u2,v2,...,u6,v6].

        Retorna:
            dofs: lista de indices globales del elemento.
        """

        dofs = []

        # Se recorren los seis nodos del elemento y se recuperan
        # sus GDL globales mediante la clase Mesh.
        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    # -------------------------------------------------------------
    # 10. Calculo de deformaciones
    # -------------------------------------------------------------

    def strain(self, u, xi=0.0, eta=0.0):
        """
        Calcula las deformaciones del LST en un punto natural.

        Parametros:
            u: vector global de desplazamientos.
            xi, eta: coordenadas naturales del punto de evaluacion.

        Por defecto, se evalua en (xi=0, eta=0), que corresponde
        al vertice local 1 del elemento de referencia.

        Retorna:
            [epsilon_x, epsilon_y, gamma_xy]
        """

        # Matriz cinematica evaluada en el punto solicitado
        B, _ = self.B_matrix(xi, eta)

        # Correspondencia entre GDL locales y globales
        dofs = self.global_dofs()

        # Extraccion de los desplazamientos del elemento
        # desde el vector global de desplazamientos
        ue = np.asarray(u)[dofs]

        # Relacion cinematica epsilon = B @ ue
        return B @ ue

    # -------------------------------------------------------------
    # 11. Calculo de tensiones
    # -------------------------------------------------------------

    def stress(self, u, xi=0.0, eta=0.0):
        """
        Calcula las tensiones del LST en un punto natural.

        Parametros:
            u: vector global de desplazamientos.
            xi, eta: coordenadas naturales del punto de evaluacion.

        Retorna:
            [sigma_x, sigma_y, tau_xy]
        """

        # Matriz constitutiva del material
        C = self.material.matriz_constitutiva()

        # Ley de Hooke: sigma = C @ epsilon
        return C @ self.strain(u, xi, eta)
