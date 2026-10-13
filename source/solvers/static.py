import numpy as np

def solve(K, f):
    """
    Resuelve el sistema de ecuaciones lineales K * u = f.
    
    Parámetros:
        K (np.ndarray): Matriz de rigidez global (ya modificada con las condiciones de borde).
        f (np.ndarray): Vector de fuerzas global (modificado).
        
    Retorna:
        u (np.ndarray): Vector solución con los desplazamientos nodales.
    """
    return np.linalg.solve(K, f)