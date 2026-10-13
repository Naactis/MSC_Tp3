import numpy as np

class LinearElastic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def matriz_constitutiva(self):
        """
        Calcula la matriz constitutiva [C] para Estado Plano de Deformación (Plane Strain).
        """
        coef = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        C = coef * np.array([
            [1 - self.nu, self.nu, 0],
            [self.nu, 1 - self.nu, 0],
            [0, 0, (1 - 2 * self.nu) / 2]
        ])
            
        return C
