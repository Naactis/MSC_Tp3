import numpy as np

class LinearElastic:
    def __init__(self, E, nu, plane_state="stress"):
        self.E = E
        self.nu = nu
        self.plane_state = plane_state

        if plane_state not in ("stress", "strain"):
            raise ValueError("El estado plano debe ser 'stress' o 'strain'")

    def matriz_constitutiva(self):

        E = self.E
        nu = self.nu

        if self.plane_state == "stress":
            # Estado plano de tension
            C = E / (1 - nu**2) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu)/2]
            ])

        else:
            # Estado plano de deformacion
            C = E / ((1 + nu) * (1 - 2*nu)) * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2*nu)/2]
            ])

        return C
