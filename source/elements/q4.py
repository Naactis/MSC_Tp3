import numpy as np

class Q4:
    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids, mesh, material, thickness):
        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness

        # 1. Extraer las coordenadas físicas de los 4 nodos
        self.coords = np.zeros((4, 2))
        for i in range(4):
            node = mesh.node(node_global_ids[i])
            self.coords[i, 0] = node.x
            self.coords[i, 1] = node.y

        self.Ke = np.zeros((8, 8))
        C = self.material.matriz_constitutiva()

        # 2. Puntos y pesos de Cuadratura de Gauss (Regla 2x2)
        # Necesario porque el integrando contiene términos x e y a la primera y segunda potencia
        gauss_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
        weight = 1.0

        # 3. Integración numérica sobre el elemento de referencia isoparamétrico
        for xi in gauss_pts:
            for eta in gauss_pts:
                # Derivadas de las funciones de forma respecto a las coordenadas naturales (xi, eta)
                # N1 = 0.25(1-xi)(1-eta), N2 = 0.25(1+xi)(1-eta), N3 = 0.25(1+xi)(1+eta), N4 = 0.25(1-xi)(1+eta)
                dN_dxi = 0.25 * np.array([-(1-eta),  (1-eta), (1+eta), -(1+eta)])
                dN_deta = 0.25 * np.array([-(1-xi), -(1+xi),  (1+xi),  (1-xi)])

                # Matriz Jacobiana J (Mapeo de xi/eta a x/y)
                J = np.zeros((2, 2))
                J[0, 0] = np.dot(dN_dxi, self.coords[:, 0])  # dx/dxi
                J[0, 1] = np.dot(dN_dxi, self.coords[:, 1])  # dy/dxi
                J[1, 0] = np.dot(dN_deta, self.coords[:, 0]) # dx/deta
                J[1, 1] = np.dot(dN_deta, self.coords[:, 1]) # dy/deta

                detJ = np.linalg.det(J)
                if detJ <= 0.0:
                    raise ValueError("Q4: Elemento distorsionado u orientación incorrecta (Jacobiano <= 0)")

                J_inv = np.linalg.inv(J)

                # Transformación de las derivadas a coordenadas físicas x e y usando la inversa de J
                dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
                dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta

                # Ensamblaje de la matriz B (3x8) evaluada en el punto de Gauss actual
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i]     = dN_dx[i]
                    B[1, 2*i + 1] = dN_dy[i]
                    B[2, 2*i]     = dN_dy[i]
                    B[2, 2*i + 1] = dN_dx[i]

                # 4. Sumar la contribución del punto de Gauss a la matriz de rigidez
                # Ke = sum( B^T * C * B * detJ * thickness * W_xi * W_eta )
                self.Ke += self.thickness * (B.T @ C @ B) * detJ * (weight * weight)

    def stiffness_matrix(self):
        return self.Ke

    def global_dofs(self):
        dofs = []
        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)
        return dofs

    def strain(self, u, xi=0.0, eta=0.0):
        # A diferencia del CST, en el Q4 la deformación no es constante.
        # Por defecto, se evalúa la deformación en el centroide del elemento (xi=0, eta=0).
        dN_dxi = 0.25 * np.array([-(1-eta),  (1-eta), (1+eta), -(1+eta)])
        dN_deta = 0.25 * np.array([-(1-xi), -(1+xi),  (1+xi),  (1-xi)])

        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, self.coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, self.coords[:, 1])
        J[1, 0] = np.dot(dN_deta, self.coords[:, 0])
        J[1, 1] = np.dot(dN_deta, self.coords[:, 1])

        J_inv = np.linalg.inv(J)

        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta

        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]     = dN_dx[i]
            B[1, 2*i + 1] = dN_dy[i]
            B[2, 2*i]     = dN_dy[i]
            B[2, 2*i + 1] = dN_dx[i]

        dofs = self.global_dofs()
        ue = np.zeros(len(dofs))
        for i in range(len(dofs)):
            ue[i] = u[dofs[i]]

        return B @ ue