
from source.mesh.node import Node
from source.materials.LinearElastic import LinearElastic


# =============================================================
# TIPOS DE ELEMENTO SOPORTADOS
# =============================================================

# Para cada tipo: cantidad de nodos y orden de interpolacion.
# Se aceptan tambien los nombres alternativos TRI3, TRI6, QUAD4, QUAD8.
ELEMENT_INFO = {
    "CST": {"n_nodes": 3, "order": 1},
    "LST": {"n_nodes": 6, "order": 2},
    "Q4":  {"n_nodes": 4, "order": 1},
    "Q8":  {"n_nodes": 8, "order": 2},
}

ELEMENT_ALIASES = {
    "TRI3": "CST",
    "TRI6": "LST",
    "QUAD4": "Q4",
    "QUAD8": "Q8",
}


def normalize_element_type(element_type):
    """Devuelve el nombre canonico del tipo de elemento (CST, LST, Q4, Q8)."""

    name = element_type.upper()
    name = ELEMENT_ALIASES.get(name, name)

    if name not in ELEMENT_INFO:
        raise ValueError(f"Tipo de elemento desconocido: {element_type}")

    return name


class Mesh:
    """
    Administra los nodos, las conectividades, los materiales y la
    numeracion global de los grados de libertad de una malla 2D.
    """

    def __init__(self):

        # Dimension espacial y cantidad de grados de libertad por nodo
        self.dim = 2               #representa un dominio bidimensional
        self.dofs_per_node = 2     #dos desplazamientos independientes

        # Nodos de la malla almacenados como objetos Node
        self.nodes = []

        # Materiales definidos en el archivo: {id: LinearElastic}
        self.materials = {}

        # Datos de cada elemento (una entrada por elemento)
        self.element_global_ids = []
        self.element_types = []
        self.element_orders = []         # orden de interpolacion (1 o 2)
        self.element_material_ids = []   # id del material de cada elemento

        # Conectividad: nodos que componen cada elemento y su orden local ej [0 1 2]
        self.connectivity = []

    # -------------------------------------------------------------
    # LECTURA DEL ARCHIVO DE MALLA
    # -------------------------------------------------------------

    def read(self, filename):
        """
        Lee un archivo de texto con materiales, nodos y elementos.

        Formato (las lineas que empiezan con # son comentarios):

            MATERIALS n_materiales
            # id  E[Pa]   nu   rho[kg/m3]  estado_plano
            1     21e9    0.2  2400        strain

            NODES n_nodos
            # id  x  y
            1     0.0  0.0
            ...

            ELEMENTS n_elementos
            # id  tipo  orden  material  nodos...
            1     Q8    2      1         1 3 17 15 2 10 16 9
            ...

        - tipo: CST, LST, Q4 o Q8.
        - orden: orden de interpolacion (1 lineal, 2 cuadratico);
          debe ser coherente con el tipo.
        - estado_plano: stress (tension plana) o strain (deformacion plana).

        Los identificadores del archivo comienzan en 1 y se convierten
        a indices internos que comienzan en 0.
        """

        # Se eliminan comentarios y lineas vacias del archivo
        lines = []

        with open(filename, "r") as file:
            for raw_line in file:
                line = raw_line.split("#")[0].strip()

                if line != "":
                    lines.append(line)

        if not lines or lines[0].split()[0].upper() not in ("MATERIALS", "NODES", "ELEMENTS"):
            raise ValueError(
                f"{filename}: formato de malla no reconocido. "
                "Se esperan las secciones MATERIALS, NODES y ELEMENTS."
            )

        # Se separan las lineas de cada seccion
        sections = {}
        line_number = 0

        while line_number < len(lines):
            values = lines[line_number].split()
            keyword = values[0].upper()

            if keyword not in ("MATERIALS", "NODES", "ELEMENTS") or len(values) != 2:
                raise ValueError(
                    f"{filename}: se esperaba un encabezado de seccion, "
                    f"se encontro '{lines[line_number]}'"
                )

            if keyword in sections:
                raise ValueError(f"{filename}: seccion {keyword} repetida")

            count = int(values[1])
            start = line_number + 1
            sections[keyword] = lines[start:start + count]

            if len(sections[keyword]) != count:
                raise ValueError(
                    f"{filename}: la seccion {keyword} declara {count} "
                    f"lineas pero tiene {len(sections[keyword])}"
                )

            line_number = start + count

        for keyword in ("MATERIALS", "NODES", "ELEMENTS"):
            if keyword not in sections:
                raise ValueError(f"{filename}: falta la seccion {keyword}")

        self._read_materials(sections["MATERIALS"])
        self._read_nodes(sections["NODES"])
        self._read_elements(sections["ELEMENTS"])

    def _read_materials(self, lines):
        """Lee: id E nu rho estado_plano"""

        for line in lines:
            values = line.split()

            if len(values) != 5:
                raise ValueError(f"Linea de material invalida: '{line}'")

            material_id = int(values[0])
            E = float(values[1])
            nu = float(values[2])
            rho = float(values[3])
            plane_state = values[4].lower()

            if material_id in self.materials:
                raise ValueError(f"Material {material_id} repetido")

            self.materials[material_id] = LinearElastic(E, nu, plane_state, rho=rho)

    def _read_nodes(self, lines):
        """Lee: id x y"""

        for i, line in enumerate(lines):
            values = line.split()

            # Conversion de numeracion externa a interna
            global_id = int(values[0]) - 1

            # Se exige numeracion consecutiva y ordenada
            if global_id != i:
                raise ValueError(
                    f"Numeracion de nodos no consecutiva: se esperaba "
                    f"el nodo {i + 1} y se encontro el {values[0]}"
                )

            # Coordenadas y creacion del objeto Node
            x = float(values[1])
            y = float(values[2])

            self.nodes.append(Node(global_id, x, y))

    def _read_elements(self, lines):
        """Lee: id tipo orden material nodos..."""

        for line in lines:
            values = line.split()

            element_global_id = int(values[0]) - 1 #importante acordarse que python arranca a contar desde el 0 y nosotros no tenemos nodo 0, arrancamos desde el 1
            element_type = normalize_element_type(values[1])
            order = int(values[2])
            material_id = int(values[3])

            info = ELEMENT_INFO[element_type]

            # El orden declarado debe coincidir con el del tipo
            if order != info["order"]:
                raise ValueError(
                    f"Elemento {values[0]}: el tipo {element_type} es de "
                    f"orden {info['order']}, pero se declaro orden {order}"
                )

            if material_id not in self.materials:
                raise ValueError(
                    f"Elemento {values[0]}: material {material_id} no definido"
                )

            # Conectividad del elemento con indices desde cero
            element_conn = [int(v) - 1 for v in values[4:]]

            if len(element_conn) != info["n_nodes"]:
                raise ValueError(
                    f"Elemento {values[0]}: {element_type} requiere "
                    f"{info['n_nodes']} nodos y se dieron {len(element_conn)}"
                )

            for nid in element_conn:
                if nid < 0 or nid >= len(self.nodes):
                    raise ValueError(
                        f"Elemento {values[0]}: nodo {nid + 1} inexistente"
                    )

            # Almacenamiento de los datos del elemento
            self.element_global_ids.append(element_global_id)
            self.element_types.append(element_type)
            self.element_orders.append(order)
            self.element_material_ids.append(material_id)
            self.connectivity.append(element_conn)

    # -------------------------------------------------------------
    # ESCRITURA DEL ARCHIVO DE MALLA
    # -------------------------------------------------------------

    def write(self, filename, header=None):
        """
        Escribe la malla en el mismo formato que lee read().

        header: texto opcional que se agrega como comentario al inicio.
        """

        lines = []

        if header:
            for text in header.splitlines():
                lines.append(f"# {text}")
            lines.append("")

        lines.append(f"MATERIALS {len(self.materials)}")
        lines.append("# id  E[Pa]  nu  rho[kg/m3]  estado_plano")

        for material_id, mat in sorted(self.materials.items()):
            lines.append(
                f"{material_id} {mat.E:.6e} {mat.nu:g} {mat.rho:g} {mat.plane_state}"
            )

        lines.append("")
        lines.append(f"NODES {len(self.nodes)}")
        lines.append("# id  x[m]  y[m]")

        for node in self.nodes:
            lines.append(f"{node.global_id + 1} {node.x:.10f} {node.y:.10f}")

        lines.append("")
        lines.append(f"ELEMENTS {len(self.connectivity)}")
        lines.append("# id  tipo  orden  material  nodos")

        for global_id, element_type, order, material_id, conn in zip(
            self.element_global_ids,
            self.element_types,
            self.element_orders,
            self.element_material_ids,
            self.connectivity
        ):
            nodes_text = " ".join(str(nid + 1) for nid in conn)
            lines.append(
                f"{global_id + 1} {element_type} {order} {material_id} {nodes_text}"
            )

        with open(filename, "w") as file:
            file.write("\n".join(lines) + "\n")

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

    def element_material(self, element_index):
        """Devuelve el material (LinearElastic) del elemento indicado."""
        return self.materials[self.element_material_ids[element_index]]

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
