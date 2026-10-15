
from source.mesh.node import Node


class Mesh:
    """
    Administra los nodos, las conectividades y la numeracion global de
    los grados de libertad de una malla de elementos finitos 2D.
    """

    def __init__(self):

        # Dimension espacial y cantidad de grados de libertad por nodo
        self.dim = 2               #representa un dominio bidimensional
        self.dofs_per_node = 2     #dos desplazamientos independientes

        # Nodos de la malla almacenados como objetos Node
        self.nodes = []

        # Identificadores y tipos de los elementos
        self.element_global_ids = []
        self.element_types = []

        # Conectividad: nodos que componen cada elemento y su orden local ej [0 1 2]
        self.connectivity = []

    def read(self, filename):
        """
        Lee un archivo de texto con la geometria y conectividad de la malla.

        Formato esperado:
            n_nodes n_elements dim
            node_id x y
            ...
            element_id element_type node_id_1 node_id_2 ...

        Los identificadores del archivo comienzan en 1 y se convierten
        a indices internos que comienzan en 0.
        """

        file = open(filename, "r")

        # Se eliminan comentarios y lineas vacias del archivo
        lines = []

        for raw_line in file:
            line = raw_line.split("#")[0]
            line = line.strip()

            if line != "":
                lines.append(line)

        file.close()

        # Lectura del encabezado del archivo
        header = lines[0].split()

        n_nodes = int(header[0])
        n_elements = int(header[1])
        dim = int(header[2])

        self.dim = dim

        # Lectura de los nodos
        line_number = 1

        for i in range(n_nodes):
            values = lines[line_number].split()
            line_number = line_number + 1

            # Conversion de numeracion externa a interna
            global_id = int(values[0]) - 1

            # Se exige numeracion consecutiva y ordenada
            if global_id != i:
                raise ValueError("Unexpected node numbering")

            # Coordenadas y creacion del objeto Node
            x = float(values[1])
            y = float(values[2])

            self.nodes.append(Node(global_id, x, y))

        # Lectura de los elementos y sus conectividades
        for i in range(n_elements):
            values = lines[line_number].split()
            line_number = line_number + 1

            element_global_id = int(values[0]) - 1 #importante acordarse que python arranca a contar desde el 0 y nosotros no tenemos nodo 0, arrancamos desde el 1
            element_type = values[1]

            # Conectividad del elemento con indices desde cero
            element_conn = []

            for j in range(2, len(values)):
                element_conn.append(int(values[j]) - 1)

            # Almacenamiento de los datos del elemento
            self.element_global_ids.append(element_global_id)
            self.element_types.append(element_type)
            self.connectivity.append(element_conn)

    #auxiliares
    def number_of_nodes(self):
        """Devuelve la cantidad total de nodos de la malla."""
        return len(self.nodes)

    def number_of_elements(self):
        """Devuelve la cantidad total de elementos de la malla."""
        return len(self.connectivity)

    def node(self, node_global_id):
        """Devuelve el objeto Node correspondiente al identificador global."""
        return self.nodes[node_global_id]

    def number_of_dofs(self):
        """Devuelve la cantidad total de grados de libertad del sistema."""
        return self.number_of_nodes() * self.dofs_per_node

    def node_global_dofs(self, node_global_id):
        """
        Devuelve los indices globales de los GDL de un nodo.
        En 2D: nodo n -> [2*n, 2*n + 1].
        """

        dofs = []

        for i in range(self.dofs_per_node):
            dofs.append(node_global_id * self.dofs_per_node + i)

        return dofs
