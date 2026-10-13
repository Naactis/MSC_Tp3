
import numpy as np

class CST:
    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids, mesh, material, thickness):
        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness

        node_1 = mesh.node(node_global_ids[0])
        node_2 = mesh.node(node_global_ids[1])
        node_3 = mesh.node(node_global_ids[2])

        x1, y1 = node_1.x, node_1.y
        x2, y2 = node_2.x, node_2.y
        x3, y3 = node_3.x, node_3.y

        A = np.array([
            [1.0, x1, y1],
            [1.0, x2, y2],
            [1.0, x3, y3]])

        detA = np.linalg.det(A)

        if detA <= 0.0:
            raise ValueError("CST: area no positiva u orientacion incorrecta")

        self.area = detA / 2.0

        
        A_inv = np.linalg.inv(A)

        self.dN_dx = A_inv[1, :]
        self.dN_dy = A_inv[2, :]

        B = np.zeros((3, 6))

        for i in range(3):
            B[0, 2*i]     = self.dN_dx[i]
            B[1, 2*i + 1] = self.dN_dy[i]

            B[2, 2*i]     = self.dN_dy[i]
            B[2, 2*i + 1] = self.dN_dx[i]

        self.B = B
        
        C = self.material.matriz_constitutiva()
        self.Ke = self.thickness * self.area * (self.B.T @ C @ self.B)

    def stiffness_matrix(self):
        return self.Ke

    def global_dofs(self):
        dofs = []

        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    def strain(self, u):
        dofs = self.global_dofs()

        ue = np.zeros(len(dofs))

        for i in range(len(dofs)):
            ue[i] = u[dofs[i]]

        return self.B @ ue
    
    def stress(self, u):
        C = self.material.matriz_constitutiva()

        return C @ self.strain(u)
