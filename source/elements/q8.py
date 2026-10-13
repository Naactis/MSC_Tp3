import numpy as np

class Q8:
    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids, mesh, material, thickness):
        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness
        
        # 1. Extraer las coordenadas físicas de los 8 nodos
        # El orden típico es: 4 esquinas (antihorario) y 4 nodos medios (antihorario)
        self.coords = np.zeros((8, 2))
        for i in range(8):
            node = mesh.node(node_global_ids[i])
            self.coords[i, 0] = node.x
            self.coords[i, 1] = node.y

        self.Ke = np.zeros((16, 16))
        C = self.material.matriz_constitutiva()

        # 2. Puntos y pesos de Cuadratura de Gauss (Regla 3x3 para integración exacta del Q8)
        gauss_pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
        gauss_weights = [5.0/9.0, 8.0/9.0, 5.0/9.0]

        # 3. Integración numérica sobre el elemento de referencia isoparamétrico
        for i, xi in enumerate(gauss_pts):
            for j, eta in enumerate(gauss_pts):
                weight = gauss_weights[i] * gauss_weights[j]
                
                # Obtener derivadas de funciones de forma en el punto de Gauss
                dN_dxi, dN_deta = self._shape_derivatives(xi, eta)

                # Matriz Jacobiana J
                J = np.zeros((2, 2))
                J[0, 0] = np.dot(dN_dxi, self.coords[:, 0])  
                J[0, 1] = np.dot(dN_dxi, self.coords[:, 1])  
                J[1, 0] = np.dot(dN_deta, self.coords[:, 0]) 
                J[1, 1] = np.dot(dN_deta, self.coords[:, 1]) 

                detJ = np.linalg.det(J)
                if detJ <= 0.0:
                    raise ValueError("Q8: Elemento distorsionado u orientación incorrecta (Jacobiano <= 0)")

                J_inv = np.linalg.inv(J)

                # Derivadas respecto a x e y
                dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
                dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta

                # Ensamblaje de la matriz B (3x16)
                B = np.zeros((3, 16))
                for k in range(8):
                    B[0, 2*k]     = dN_dx[k]
                    B[1, 2*k + 1] = dN_dy[k]
                    B[2, 2*k]     = dN_dy[k]
                    B[2, 2*k + 1] = dN_dx[k]

                # 4. Sumar contribución a la matriz de rigidez
                self.Ke += self.thickness * (B.T @ C @ B) * detJ * weight

    def _shape_derivatives(self, xi, eta):
        """Calcula las derivadas de las 8 funciones de forma respecto a xi y eta."""
        dN_dxi = np.zeros(8)
        dN_deta = np.zeros(8)
        
        # Nodos en las esquinas (1 a 4)
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

    def stiffness_matrix(self):
        #Devuelve la matriz de rigidez del elemento Q8.
        return self.Ke

    def global_dofs(self):
        #Devuelve la lista de grados de libertad globales asociados a este elemento.
        dofs = []
        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)
        return dofs

    def strain(self, u, xi=0.0, eta=0.0):    
        #Devuelve la deformación en el punto (xi, eta) del elemento. Por defecto, se evalúa en el centroide (xi=0, eta=0).
        dN_dxi, dN_deta = self._shape_derivatives(xi, eta)

        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, self.coords[:, 0])
        J[0, 1] = np.dot(dN_dxi, self.coords[:, 1])
        J[1, 0] = np.dot(dN_deta, self.coords[:, 0])
        J[1, 1] = np.dot(dN_deta, self.coords[:, 1])

        J_inv = np.linalg.inv(J)

        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta

        B = np.zeros((3, 16))
        for k in range(8):
            B[0, 2*k]     = dN_dx[k]
            B[1, 2*k + 1] = dN_dy[k]
            B[2, 2*k]     = dN_dy[k]
            B[2, 2*k + 1] = dN_dx[k]

        dofs = self.global_dofs()
        ue = np.zeros(len(dofs))
        for i in range(len(dofs)):
            ue[i] = u[dofs[i]]

        return B @ ue