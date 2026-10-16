"""
Generador de mallas estructuradas para la seccion trapezoidal de la presa.

Geometria (unidades en metros):

          (0,H) ___ (C,H)
               |    \\
               |     \\        paramento aguas arriba: x = 0 (vertical)
               |      \\       paramento aguas abajo: recta (B,0)-(C,H)
               |       \\
          (0,0) -------- (B,0)

Se divide la seccion en nx columnas y ny filas de celdas. Cada celda
se convierte en un Q4/Q8, o en dos triangulos CST/LST.

Mapeo de la grilla:
    y = H * j / ny
    x = (i / nx) * ancho(y),   con ancho(y) = B - (B - C) * y / H

Las celdas resultantes tienen aristas rectas. Los nodos intermedios
(LST y Q8) se ubican en el punto medio geometrico de cada arista,
y se comparten entre los elementos vecinos.
"""

from source.mesh.mesh import Mesh, ELEMENT_INFO, normalize_element_type
from source.mesh.node import Node


def generate_dam_mesh(
    element_type,
    nx,
    ny,
    material,
    height=30.0,
    base_width=18.0,
    crest_width=3.0,
    material_id=1
):
    """
    Genera la malla de la presa y devuelve un objeto Mesh.

    Parametros:
        element_type: "CST", "LST", "Q4" o "Q8".
        nx: cantidad de divisiones en la direccion horizontal.
        ny: cantidad de divisiones en la direccion vertical.
        material: objeto LinearElastic (con E, nu, rho y plane_state).
        height, base_width, crest_width: geometria de la presa [m].
        material_id: identificador con el que se guarda el material.

    Total de elementos: nx*ny (Q4, Q8) o 2*nx*ny (CST, LST).
    """

    element_type = normalize_element_type(element_type)

    if not (isinstance(nx, int) and isinstance(ny, int) and nx >= 1 and ny >= 1):
        raise ValueError("nx y ny deben ser enteros mayores o iguales a 1")

    if not (height > 0 and base_width > 0 and crest_width > 0):
        raise ValueError("La geometria de la presa debe ser positiva")

    mesh = Mesh()
    mesh.materials[material_id] = material

    # ---------------------------------------------------------
    # 1. Nodos de vertice (grilla de (nx+1) x (ny+1))
    # ---------------------------------------------------------

    def add_node(x, y):
        node_id = len(mesh.nodes)
        mesh.nodes.append(Node(node_id, x, y))
        return node_id

    vertex = {}

    for j in range(ny + 1):
        y = height * j / ny
        width = base_width - (base_width - crest_width) * y / height

        for i in range(nx + 1):
            x = width * i / nx
            vertex[(i, j)] = add_node(x, y)

    # ---------------------------------------------------------
    # 2. Nodos intermedios de arista (solo LST y Q8)
    # ---------------------------------------------------------

    # Cada arista se identifica por el par ordenado de sus vertices,
    # para que dos elementos vecinos compartan el mismo nodo medio.
    midside = {}

    def mid_node(a, b):
        key = (min(a, b), max(a, b))

        if key not in midside:
            na = mesh.nodes[a]
            nb = mesh.nodes[b]
            midside[key] = add_node(0.5 * (na.x + nb.x), 0.5 * (na.y + nb.y))

        return midside[key]

    # ---------------------------------------------------------
    # 3. Elementos (vertices en sentido antihorario)
    # ---------------------------------------------------------

    order = ELEMENT_INFO[element_type]["order"]

    def add_element(connectivity):
        mesh.element_global_ids.append(len(mesh.connectivity))
        mesh.element_types.append(element_type)
        mesh.element_orders.append(order)
        mesh.element_material_ids.append(material_id)
        mesh.connectivity.append(connectivity)

    for j in range(ny):
        for i in range(nx):

            # Vertices de la celda, antihorario desde abajo-izquierda
            v1 = vertex[(i, j)]
            v2 = vertex[(i + 1, j)]
            v3 = vertex[(i + 1, j + 1)]
            v4 = vertex[(i, j + 1)]

            if element_type == "Q4":
                add_element([v1, v2, v3, v4])

            elif element_type == "Q8":
                # [4 vertices, nodos medios de 1-2, 2-3, 3-4, 4-1]
                add_element([
                    v1, v2, v3, v4,
                    mid_node(v1, v2), mid_node(v2, v3),
                    mid_node(v3, v4), mid_node(v4, v1)
                ])

            else:
                # Dos triangulos separados por la diagonal v1-v3
                for a, b, c in [(v1, v2, v3), (v1, v3, v4)]:

                    if element_type == "CST":
                        add_element([a, b, c])

                    else:  # LST
                        # [3 vertices, nodos medios de 1-2, 2-3, 3-1]
                        add_element([
                            a, b, c,
                            mid_node(a, b), mid_node(b, c), mid_node(c, a)
                        ])

    return mesh


def write_dam_mesh(
    filename,
    element_type,
    nx,
    ny,
    material,
    height=30.0,
    base_width=18.0,
    crest_width=3.0
):
    """
    Genera la malla de la presa y la escribe en un archivo de texto.

    Devuelve el objeto Mesh generado.
    """

    mesh = generate_dam_mesh(
        element_type, nx, ny, material,
        height=height, base_width=base_width, crest_width=crest_width
    )

    header = (
        f"Malla de la presa generada automaticamente\n"
        f"Elemento {normalize_element_type(element_type)}, "
        f"{nx} x {ny} divisiones (horizontal x vertical)\n"
        f"Geometria: H = {height} m, base = {base_width} m, "
        f"cresta = {crest_width} m"
    )

    mesh.write(filename, header=header)

    return mesh
