import numpy as np


def assemble_stiffness(mesh, elements):
    n_dofs = mesh.number_of_dofs()

    K = np.zeros((n_dofs, n_dofs))

    for element in elements:
        Ke = element.stiffness_matrix()
        dofs = element.global_dofs()

        for i in range(len(dofs)):
            for j in range(len(dofs)):
                K[dofs[i], dofs[j]] += Ke[i, j]

    return K
