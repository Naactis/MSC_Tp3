import numpy as np

def apply_dirichlet_bc(K, f, dof, value):
    """
    Aplica una condición de borde de Dirichlet (desplazamiento prescrito)
    modificando la matriz de rigidez global K y el vector de fuerzas f para un solo GDL.
    """
    f -= K[:, dof] * value
    K[dof, :] = 0.0
    K[:, dof] = 0.0
    K[dof, dof] = 1.0
    f[dof] = value

    return K, f


def apply_dirichlet_bcs(K, f, fixed_dofs, fixed_values=None):
    """
    Aplica condiciones de borde de Dirichlet para una lista de GDLs fijos.
    Crea copias de K y f para no alterar las matrices originales.
    """
    K_mod = K.copy()
    f_mod = f.copy()

    if fixed_values is None:
        fixed_values = [0.0] * len(fixed_dofs)

    for dof, val in zip(fixed_dofs, fixed_values):
        apply_dirichlet_bc(K_mod, f_mod, dof, val)

    return K_mod, f_mod


def solve_system_direct(K, f, fixed_dofs, fixed_values=None):
    """
    Aplica las condiciones de borde sobre copias de K y f, resuelve el sistema completo
    y retorna el vector de desplazamientos u y el vector de reacciones R.
    """
    K_mod, f_mod = apply_dirichlet_bcs(K, f, fixed_dofs, fixed_values)
    u = np.linalg.solve(K_mod, f_mod)
    R = K @ u - f
    return u, R