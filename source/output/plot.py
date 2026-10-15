
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_deformed_mesh(
    mesh,
    elements,
    u,
    scale_factor=200.0,
    title="Malla Deformada",
    save_filename=None
):
    """
    Grafica las mallas original y deformada para elementos:
        - CST: triangulo lineal de 3 nodos.
        - LST: triangulo cuadratico de 6 nodos.
        - Q4: cuadrilatero bilineal de 4 nodos.
        - Q8: cuadrilatero cuadratico de 8 nodos.

    Parametros:
        mesh: objeto Mesh.
        elements: lista de elementos finitos.
        u: vector global de desplazamientos.
        scale_factor: factor de amplificacion visual.
        title: titulo del grafico.
        save_filename: nombre del archivo para guardar la figura.

    La deformacion representada esta amplificada:
        x_def = x + scale_factor * ux
        y_def = y + scale_factor * uy

    Las aristas se representan mediante segmentos rectos
    entre nodos. Para elementos geometricamente curvos,
    seria necesario interpolar puntos adicionales.
    """

    # ---------------------------------------------------------
    # 1. Preparacion de la figura
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 7))

    # Convertimos el vector de desplazamientos a un arreglo
    # unidimensional para utilizar indices globales.
    u = np.asarray(u).reshape(-1)

    # ---------------------------------------------------------
    # 2. Recorrido de los elementos
    # ---------------------------------------------------------

    for i, elem in enumerate(elements):

        node_ids = elem.node_global_ids

        # -----------------------------------------------------
        # 3. Orden de recorrido del contorno elemental
        # -----------------------------------------------------

        if len(node_ids) == 3:

            # CST:
            # Tres vertices recorridos en sentido antihorario.
            order = [0, 1, 2, 0]

        elif len(node_ids) == 6:

            # LST:
            # Tres vertices y tres nodos intermedios.
            #
            # Nodos 0, 1, 2: vertices.
            # Nodo 3: arista 0-1.
            # Nodo 4: arista 1-2.
            # Nodo 5: arista 2-0.
            #
            # IMPORTANTE:
            # No se debe recorrer [0,1,2,3,4,5,0]
            # porque eso conecta nodos que no comparten arista.
            order = [0, 3, 1, 4, 2, 5, 0]

        elif len(node_ids) == 4:

            # Q4:
            # Cuatro vertices antihorarios.
            order = [0, 1, 2, 3, 0]

        elif len(node_ids) == 8:

            # Q8:
            # Cuatro vertices y cuatro nodos intermedios.
            #
            # Vertices: 0,1,2,3.
            # Medios: 4,5,6,7.
            order = [0, 4, 1, 5, 2, 6, 3, 7, 0]

        else:

            # Evitamos graficar conectividades desconocidas
            # con un orden posiblemente incorrecto.
            raise ValueError(
                f"Tipo de elemento no soportado: "
                f"{len(node_ids)} nodos."
            )

        # -----------------------------------------------------
        # 4. Coordenadas originales y deformadas
        # -----------------------------------------------------

        x_orig = []
        y_orig = []

        x_def = []
        y_def = []

        for idx in order:

            # Identificador global del nodo
            nid = node_ids[idx]

            # Recuperamos las coordenadas originales
            node = mesh.node(nid)

            x0 = node.x
            y0 = node.y

            # CAMBIO:
            # Se usa el metodo de la malla para recuperar
            # los GDL globales, en lugar de asumirlos.
            dofs = mesh.node_global_dofs(nid)

            ux = u[dofs[0]]
            uy = u[dofs[1]]

            # Coordenadas originales
            x_orig.append(x0)
            y_orig.append(y0)

            # Coordenadas deformadas amplificadas
            x_def.append(x0 + scale_factor * ux)
            y_def.append(y0 + scale_factor * uy)

        # -----------------------------------------------------
        # 5. Dibujo de la malla original
        # -----------------------------------------------------

        # Dibujamos cada contorno original con una linea
        # gris discontinua.
        #
        # Solo el primer elemento agrega la etiqueta a
        # la leyenda para evitar entradas repetidas.
        ax.plot(
            x_orig,
            y_orig,
            linestyle="--",
            color="gray",
            linewidth=1.2,
            alpha=0.65,
            label="Original" if i == 0 else None
        )

        # -----------------------------------------------------
        # 6. Dibujo de la malla deformada
        # -----------------------------------------------------

        # La linea azul une los nodos en el orden correcto.
        # Los circulos representan los nodos de cada elemento,
        # incluyendo los nodos intermedios del LST y del Q8.
        ax.plot(
            x_def,
            y_def,
            "b-o",
            linewidth=1.5,
            markersize=4,
            label="Deformada" if i == 0 else None
        )

    # ---------------------------------------------------------
    # 7. Formato del grafico
    # ---------------------------------------------------------

    ax.axhline(
        0,
        color="gray",
        linestyle=":",
        linewidth=0.8
    )

    ax.axvline(
        0,
        color="gray",
        linestyle=":",
        linewidth=0.8
    )

    # Misma escala en ambos ejes para no alterar
    # visualmente las proporciones de la geometria.
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"{title} (Escala: {scale_factor}x)"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    ax.legend(loc="upper right")

    fig.tight_layout()

    # ---------------------------------------------------------
    # 8. Guardado opcional de la figura
    # ---------------------------------------------------------

    if save_filename:

        # Se crea la carpeta output si no existe.
        os.makedirs("output", exist_ok=True)

        filepath = os.path.join(
            "output",
            save_filename
        )

        fig.savefig(
            filepath,
            dpi=300
        )

        print(
            f"Grafico guardado exitosamente en: {filepath}"
        )

    # ---------------------------------------------------------
    # 9. Visualizacion
    # ---------------------------------------------------------

    plt.show()
