import numpy as np

def apply_dirichlet_bc(K, f, dof, value):
    """
    Aplica una condición de borde de Dirichlet (desplazamiento prescrito)
    modificando la matriz de rigidez global K y el vector de fuerzas f.
    
    Parámetros:
        K (np.ndarray): Matriz de rigidez global (n_dofs x n_dofs).
        f (np.ndarray): Vector de fuerzas global (n_dofs,).
        dof (int): Índice del grado de libertad (GDL) a restringir.
        value (float): Valor del desplazamiento prescrito (típicamente 0.0 para apoyos).
    """
    # 1. Modificar el vector de fuerzas restando la contribución del desplazamiento conocido
    # f = f - K[:, dof] * value
    f -= K[:, dof] * value

    # 2. Hacer cero la fila y la columna correspondientes al GDL prescrito
    K[dof, :] = 0.0
    K[:, dof] = 0.0

    # 3. Colocar un 1 en la diagonal principal y el valor prescrito en el vector de fuerzas
    K[dof, dof] = 1.0
    f[dof] = value

    return K, f