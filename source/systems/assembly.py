
import numpy as np


def assemble_stiffness(mesh, elements):
    """
    Los programas de formulación de los elementos ya me entregan su matriz K elemental entonces esto 
    ensambla la matriz de rigidez global a partir de ellas.

    Parametros:
        mesh: objeto Mesh que contiene la malla y su numeracion de GDL.
        elements: lista de objetos de elementos finitos (CST, LST, Q4, Q8).

    Retorno:
        K: matriz de rigidez global.
    """

    # Cantidad total de GDL: numero de nodos multiplicado por los GDL por nodo
    n_dofs = mesh.number_of_dofs()

    # Inicializacion de la matriz de rigidez global: creo la matriz cuadrada de ceros
    K = np.zeros((n_dofs, n_dofs))

    # Se recorre cada elemento para incorporar su contribucion a la matriz global
    for element in elements:

        # Matriz de rigidez elemental y correspondencia local-global de GDL
        # todo lo saco directamente de cada elemento
        Ke = element.stiffness_matrix()
        dofs = element.global_dofs()

        # Se recorren las filas y columnas de Ke y se suman sus entradas
        # en las posiciones de K correspondientes a los GDL globales.
        # Usamos += porque varios elementos pueden compartir GDL
        # entonces sumamos en vez de reemplazar para acumular las contribuciones
        # de cada elemento
        
        for i in range(len(dofs)):
            for j in range(len(dofs)):
                K[dofs[i], dofs[j]] += Ke[i, j]

    return K
