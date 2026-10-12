import numpy as np

class LinearElastic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def matriz_constitutiva(self):
        C = self.E / (1 - self.nu**2) * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu)/2]
        ])

        return C
