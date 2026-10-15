import sys
import os
import numpy as np

# Agregar la carpeta raíz al path para las importaciones
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from source.mesh.mesh import Mesh  
from source.materials.hormigon import hormigon, rho
from source.elements.q4 import Q4
from source.elements.q8 import Q8
from source.systems.assembly import assemble_stiffness
from source.systems.bc import solve_system_direct
from source.output.plot import plot_deformed_mesh


def gauss_points_2d(elem):
    """
    Devuelve los puntos y pesos de Gauss 1D adecuados al elemento:
    regla 3x3 para Q8 y 2x2 para Q4.
    """
    if len(elem.node_global_ids) == 8:  # Q8
        return [-np.sqrt(0.6), 0.0, np.sqrt(0.6)], [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    else:  # Q4
        return [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], [1.0, 1.0]


def compute_body_forces(mesh, elements, rho_concrete=2400.0, g=9.81, thickness=1.0):
    """
    Calcula e integra el peso propio (fuerzas de cuerpo en -Y) sobre el dominio 2D.

    f_i = integral( N_i * b_y ) dA,  con b_y = -rho * g

    Las funciones de forma y el Jacobiano se toman de cada elemento
    (shape_functions y B_matrix), para no repetir sus formulas aca.
    """
    number_of_dofs = mesh.number_of_dofs()
    f_body = np.zeros(number_of_dofs)
    b_y = -rho_concrete * g  # Carga vertical descendente N/m3

    for elem in elements:
        node_ids = elem.node_global_ids
        pts_1d, w_1d = gauss_points_2d(elem)

        for xi, w_xi in zip(pts_1d, w_1d):
            for eta, w_eta in zip(pts_1d, w_1d):
                N = elem.shape_functions(xi, eta)
                _, detJ = elem.B_matrix(xi, eta)

                weight = w_xi * w_eta * detJ * thickness

                # Sumar contribuciones a los GDL verticales (Y -> índices impares)
                for i, nid in enumerate(node_ids):
                    f_body[2 * nid + 1] += N[i] * b_y * weight

    return f_body


def compute_area(elements):
    """
    Integra el area del dominio sumando detJ en los puntos de Gauss
    de cada elemento.
    """
    area = 0.0

    for elem in elements:
        pts_1d, w_1d = gauss_points_2d(elem)

        for xi, w_xi in zip(pts_1d, w_1d):
            for eta, w_eta in zip(pts_1d, w_1d):
                _, detJ = elem.B_matrix(xi, eta)
                area += w_xi * w_eta * detJ

    return area


def compute_hydrostatic_forces(mesh, elements, rho_water=1000.0, g=9.81, y_water_level=28.0):
    """
    Calcula e integra la presión hidrostática en el paramento de aguas arriba (x = 0).
    """
    number_of_dofs = mesh.number_of_dofs()
    f_hydro = np.zeros(number_of_dofs)

    for elem in elements:
        node_ids = elem.node_global_ids
        
        upstream_nodes = []
        for local_idx, nid in enumerate(node_ids):
            node = mesh.node(nid)
            if abs(node.x - 0.0) < 1e-5:
                upstream_nodes.append((local_idx, nid, node.y))

        upstream_nodes.sort(key=lambda item: item[2])

        if len(upstream_nodes) == 3: # Q8
            n1_id, y1 = upstream_nodes[0][1], upstream_nodes[0][2]
            n2_id, y2 = upstream_nodes[1][1], upstream_nodes[1][2]
            n3_id, y3 = upstream_nodes[2][1], upstream_nodes[2][2]

            if y1 < y_water_level:
                L_face = abs(y3 - y1)
                J_1d = L_face / 2.0
                gauss_pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
                gauss_w = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

                for xi, w in zip(gauss_pts, gauss_w):
                    N1 = 0.5 * xi * (xi - 1.0)
                    N2 = 1.0 - xi**2
                    N3 = 0.5 * xi * (xi + 1.0)

                    y_gauss = N1 * y1 + N2 * y2 + N3 * y3
                    depth = max(0.0, y_water_level - y_gauss)
                    pressure = rho_water * g * depth

                    f_hydro[2 * n1_id] += w * N1 * pressure * J_1d
                    f_hydro[2 * n2_id] += w * N2 * pressure * J_1d
                    f_hydro[2 * n3_id] += w * N3 * pressure * J_1d

        elif len(upstream_nodes) == 2: # Q4
            n1_id, y1 = upstream_nodes[0][1], upstream_nodes[0][2]
            n2_id, y2 = upstream_nodes[1][1], upstream_nodes[1][2]

            if y1 < y_water_level:
                L_face = abs(y2 - y1)
                J_1d = L_face / 2.0
                gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
                gauss_w = [1.0, 1.0]

                for xi, w in zip(gauss_pts, gauss_w):
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)

                    y_gauss = N1 * y1 + N2 * y2
                    depth = max(0.0, y_water_level - y_gauss)
                    pressure = rho_water * g * depth

                    f_hydro[2 * n1_id] += w * N1 * pressure * J_1d
                    f_hydro[2 * n2_id] += w * N2 * pressure * J_1d

    return f_hydro


def main():
    # 1. Cargar la malla
    print("Cargando la malla...")
    mesh = Mesh()
    mesh.read("meshes/Q8.mesh")  
    mesh.dofs_per_node = 2

    # 2. Material y Propiedades
    material = hormigon
    thickness = 1.0
    g = 9.81
    rho_agua = 1000.0
    y_agua = 28.0

    # 3. Instanciar elementos
    print("Instanciando elementos...")
    elements = []
    
    for e in range(mesh.number_of_elements()):
        element_type = mesh.element_types[e]
        
        if element_type in ["QUAD4", "Q4"]:
            elements.append(Q4(mesh.element_global_ids[e], mesh.connectivity[e], mesh, material, thickness))
        elif element_type in ["QUAD8", "Q8"]:
            elements.append(Q8(mesh.element_global_ids[e], mesh.connectivity[e], mesh, material, thickness))

    # 4. Ensamblar Matriz de Rigidez Global
    print("Ensamblando matriz de rigidez global...")
    K = assemble_stiffness(mesh, elements)

    # 5. Vector de Cargas Totales
    number_of_dofs = mesh.number_of_dofs()
    f = np.zeros(number_of_dofs)

    print("Calculando e integrando fuerzas hidrostáticas y peso propio...")
    f_hydro = compute_hydrostatic_forces(mesh, elements, rho_water=rho_agua, g=g, y_water_level=y_agua)
    f_weight = compute_body_forces(mesh, elements, rho_concrete=rho, g=g, thickness=thickness)
    
    f += f_hydro + f_weight

    # --- VERIFICACIÓN DE CARGAS INTEGRADAS Y ÁREA ---
    Fx_integrada = np.sum(f_hydro[0::2])
    Fx_teorica = 0.5 * rho_agua * g * (y_agua ** 2)

    # Area integrada por el FEM vs area analitica del trapecio.
    # El peso teorico usa el area analitica, asi la comparacion
    # es independiente de la malla.
    area_total_fem = compute_area(elements)
    area_teorica = 0.5 * (18.0 + 3.0) * 30.0   # trapecio: (base + cresta)/2 * altura

    W_teorico = area_teorica * rho * g * thickness
    W_integrado = abs(np.sum(f_weight[1::2]))

    print("-" * 65)
    print(" VERIFICACIÓN DE CARGAS Y GEOMETRÍA (FEM):")
    print(f"  • Resultante Hidrostática (FEM):  {Fx_integrada / 1e6:.4f} MN")
    print(f"  • Resultante Hidrostática (Teórica): {Fx_teorica / 1e6:.4f} MN")
    print(f"  • Área de la Presa (FEM):         {area_total_fem:.2f} m²")
    print(f"  • Área de la Presa (Teórica):     {area_teorica:.2f} m²")
    print(f"  • Peso Propio (FEM):              {W_integrado / 1e6:.4f} MN")
    print(f"  • Peso Propio (Teórico):          {W_teorico / 1e6:.4f} MN")
    print("-" * 65)

    # 6. Condiciones de Borde (Empotramiento en y = 0)
    fixed_dofs = []
    fixed_values = []
    base_nodes = [nid for nid in range(mesh.number_of_nodes()) if abs(mesh.node(nid).y - 0.0) < 1e-6]
    
    for nid in base_nodes:
        dofs = mesh.node_global_dofs(nid)
        fixed_dofs.extend(dofs)
        fixed_values.extend([0.0, 0.0])

    print(f"Aplicando empotramiento en {len(base_nodes)} nodos de la base...")

    # 7. Resolver el Sistema
    print("Resolviendo el sistema...")
    u, R = solve_system_direct(K, f, fixed_dofs, fixed_values)
    print(">>> Cálculo finalizado con éxito <<<")

    # --- VERIFICACIÓN DE EQUILIBRIO DE REACCIONES ---
    Ry_reaccion = np.sum([R[dof] for dof in fixed_dofs if dof % 2 != 0])
    Fx_reaccion = np.sum([R[dof] for dof in fixed_dofs if dof % 2 == 0])

    print("-" * 65)
    print(" EQUILIBRIO DE REACCIONES EN LOS APOYOS:")
    print(f"  • Empuje Hidrostático Aplicado (+X): {Fx_integrada / 1e6:.4f} MN")
    print(f"  • Reacción Horizontal Base (Rx):     {Fx_reaccion / 1e6:.4f} MN")
    print(f"  • Peso Propio Aplicado (-Y):         {-W_integrado / 1e6:.4f} MN")
    print(f"  • Reacción Vertical Base (Ry):       {Ry_reaccion / 1e6:.4f} MN")
    print("-" * 65)

    # 8. Guardar resultado
    print("Generando gráfico...")
    plot_deformed_mesh(
        mesh, 
        elements, 
        u, 
        scale_factor=500.0, 
        title="Presa de Gravedad - Deformada (Peso + Presión Agua)", 
        save_filename="presa_completa.png"  # <-- Solo el nombre del archivo
    )

if __name__ == "__main__":
    main()