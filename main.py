import sys
import os
import numpy as np

# Agregar la carpeta raíz al path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from source.mesh.mesh import Mesh  
from source.materials.hormigon import hormigon, rho
from source.elements.q4 import Q4
from source.elements.q8 import Q8
from source.systems.assembly import assemble_stiffness
from source.systems.bc import solve_system_direct
from source.output.plot import plot_deformed_mesh

def compute_body_forces(mesh, elements, rho_concrete=2400.0, g=9.81, thickness=1.0):
    """
    Calcula e integra el peso propio (fuerzas de cuerpo en -Y) sobre todo el dominio 2D.
    Soporta integración de Gauss 2D para elementos Q4 (2x2) y Q8 (3x3).
    """
    number_of_dofs = mesh.number_of_dofs()
    f_body = np.zeros(number_of_dofs)

    b_y = -rho_concrete * g  # Carga vertical descendente N/m3

    for elem in elements:
        node_ids = elem.node_global_ids
        
        # Selección de esquema de integración de Gauss 2D
        if len(node_ids) == 8:  # Q8
            pts_1d = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
            w_1d = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        else:  # Q4
            pts_1d = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
            w_1d = [1.0, 1.0]

        # Cuadratura de Gauss 2D (producto cartesiano)
        for xi, w_xi in zip(pts_1d, w_1d):
            for eta, w_eta in zip(pts_1d, w_1d):
                # Usar los métodos del propio elemento para evaluar N y det(J)
                N = elem.shape_functions(xi, eta) if hasattr(elem, 'shape_functions') else None
                
                # Si el elemento expone shape_functions_and_derivatives
                if hasattr(elem, 'shape_functions_and_derivatives'):
                    N, dN_dxi, dN_deta = elem.shape_functions_and_derivatives(xi, eta)
                    coords = np.array([[mesh.node(nid).x, mesh.node(nid).y] for nid in node_ids])
                    J = np.dot(np.vstack((dN_dxi, dN_deta)), coords)
                    detJ = np.linalg.det(J)
                else:
                    # En su defecto, evaluar B / detJ desde el elemento
                    coords = np.array([[mesh.node(nid).x, mesh.node(nid).y] for nid in node_ids])
                    # N para Q8/Q4 en (xi, eta)
                    if len(node_ids) == 8:
                        N = np.array([
                            0.25*(1-xi)*(1-eta)*(-xi-eta-1),
                            0.25*(1+xi)*(1-eta)*(xi-eta-1),
                            0.25*(1+xi)*(1+eta)*(xi+eta-1),
                            0.25*(1-xi)*(1+eta)*(-xi+eta-1),
                            0.5*(1-xi**2)*(1-eta),
                            0.5*(1+xi)*(1-eta**2),
                            0.5*(1-xi**2)*(1+eta),
                            0.5*(1-xi)*(1-eta**2)
                        ])
                        dN_dxi = np.array([
                            0.25*(1-eta)*(2*xi+eta), 0.25*(1-eta)*(2*xi-eta),
                            0.25*(1+eta)*(2*xi+eta), 0.25*(1+eta)*(2*xi-eta),
                            -xi*(1-eta), 0.5*(1-eta**2), -xi*(1+eta), -0.5*(1-eta**2)
                        ])
                        dN_deta = np.array([
                            0.25*(1-xi)*(xi+2*eta), 0.25*(1+xi)*(-xi+2*eta),
                            0.25*(1+xi)*(xi+2*eta), 0.25*(1-xi)*(-xi+2*eta),
                            -0.5*(1-xi**2), -eta*(1+xi), 0.5*(1-xi**2), -eta*(1-xi)
                        ])
                    else: # Q4
                        N = np.array([
                            0.25*(1-xi)*(1-eta), 0.25*(1+xi)*(1-eta),
                            0.25*(1+xi)*(1+eta), 0.25*(1-xi)*(1+eta)
                        ])
                        dN_dxi = np.array([-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)])
                        dN_deta = np.array([-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)])

                    J = np.dot(np.vstack((dN_dxi, dN_deta)), coords)
                    detJ = np.linalg.det(J)

                weight = w_xi * w_eta * detJ * thickness

                # Sumar contribuciones a los GDL verticales (Y -> índices impares)
                for i, nid in enumerate(node_ids):
                    f_body[2 * nid + 1] += N[i] * b_y * weight

    return f_body


def compute_hydrostatic_forces(mesh, elements, rho_water=1000.0, g=9.81, y_water_level=28.0):
    """
    Calcula e integra la presión hidrostática en el paramento de aguas arriba (x = 0)
    Soporta elementos Q4 (2 nodos de borde) y Q8 (3 nodos de borde).
    """
    number_of_dofs = mesh.number_of_dofs()
    f_hydro = np.zeros(number_of_dofs)

    for elem in elements:
        node_ids = elem.node_global_ids
        
        # Filtrar nodos del elemento que están sobre el paramento vertical x = 0
        upstream_nodes = []
        for local_idx, nid in enumerate(node_ids):
            node = mesh.node(nid)
            if abs(node.x - 0.0) < 1e-5:
                upstream_nodes.append((local_idx, nid, node.y))

        # Ordenar nodos de la cara de abajo hacia arriba según coordenada Y
        upstream_nodes.sort(key=lambda item: item[2])

        # --- CASO Q8: Cara de 3 nodos ---
        if len(upstream_nodes) == 3:
            n1_id, y1 = upstream_nodes[0][1], upstream_nodes[0][2]
            n2_id, y2 = upstream_nodes[1][1], upstream_nodes[1][2]  # Nodo medio
            n3_id, y3 = upstream_nodes[2][1], upstream_nodes[2][2]

            # Si al menos parte de la cara está sumergida (por debajo de y_water_level)
            if y1 < y_water_level:
                L_face = abs(y3 - y1)
                J_1d = L_face / 2.0  # Jacobiano 1D

                # Puntos y pesos de Gauss para 3 puntos (exacto para polinomios de grado <= 5)
                gauss_pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
                gauss_w = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

                for xi, w in zip(gauss_pts, gauss_w):
                    # Funciones de forma cuadráticas 1D
                    N1 = 0.5 * xi * (xi - 1.0)
                    N2 = 1.0 - xi**2
                    N3 = 0.5 * xi * (xi + 1.0)

                    # Cota y profundidad en el punto de Gauss
                    y_gauss = N1 * y1 + N2 * y2 + N3 * y3
                    depth = max(0.0, y_water_level - y_gauss)
                    pressure = rho_water * g * depth

                    # Asignar fuerzas horizontales (+X) a los 3 nodos de la cara
                    f_hydro[2 * n1_id] += w * N1 * pressure * J_1d
                    f_hydro[2 * n2_id] += w * N2 * pressure * J_1d
                    f_hydro[2 * n3_id] += w * N3 * pressure * J_1d

        # --- CASO Q4: Cara de 2 nodos ---
        elif len(upstream_nodes) == 2:
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

    # 3. Instanciar elementos
    print("Instanciando elementos...")
    elements = []
    
    for e in range(mesh.number_of_elements()):
        element_type = mesh.element_types[e]
        
        if element_type in ["QUAD4", "Q4"]:
            element = Q4(
                mesh.element_global_ids[e],
                mesh.connectivity[e],
                mesh,
                material,
                thickness
            )
            elements.append(element)
            
        elif element_type in ["QUAD8", "Q8"]:
            element = Q8(
                mesh.element_global_ids[e],
                mesh.connectivity[e],
                mesh,
                material,
                thickness
            )
            elements.append(element)

    # 4. Ensamblar Matriz de Rigidez Global
    print("Ensamblando matriz de rigidez global...")
    K = assemble_stiffness(mesh, elements)

    # 5. Vector de Cargas (Presión Hidrostática)
    number_of_dofs = mesh.number_of_dofs()
    f = np.zeros(number_of_dofs)

    print("Calculando e integrando fuerzas hidrostáticas...")
    rho_agua = 1000.0   # kg/m3
    g = 9.81            # m/s2
    y_agua = 28.0       # Cota máxima real del agua (m)

    f_hydro = compute_hydrostatic_forces(
        mesh=mesh, 
        elements=elements, 
        rho_water=rho_agua, 
        g=g, 
        y_water_level=y_agua
    )
    
    f += f_hydro

    # --- VERIFICACIÓN DE CARGA HIDROSTÁTICA ---
    Fx_integrada = np.sum(f_hydro[0::2])  # Suma de fuerzas horizontales (GDL X)
    Fx_teorica = 0.5 * rho_agua * g * (y_agua ** 2)  # F = 0.5 * rho * g * H^2
    error_porcentual = abs(Fx_integrada - Fx_teorica) / Fx_teorica * 100.0

    print("-" * 55)
    print(" VERIFICACIÓN DE LA CARGA HIDROSTÁTICA INTEGRADA:")
    print(f"  • Resultante Integrada (FEM): {Fx_integrada / 1e6:.4f} MN")
    print(f"  • Resultante Teórica (Analítica): {Fx_teorica / 1e6:.4f} MN")
    print(f"  • Error de Integración: {error_porcentual:.2f}%")
    print("-" * 55)

    # 6. Condiciones de Borde (Empotramiento en y = 0)
    fixed_dofs = []
    fixed_values = []

    base_nodes = [nid for nid in range(mesh.number_of_nodes()) if abs(mesh.node(nid).y - 0.0) < 1e-6]
    
    for nid in base_nodes:
        # Acceso directo al método de tu Mesh
        dofs = mesh.node_global_dofs(nid)
        fixed_dofs.extend(dofs)
        fixed_values.extend([0.0, 0.0])

    print(f"Aplicando empotramiento en {len(base_nodes)} nodos de la base...")

    # 7. Resolver el Sistema
    print("Resolviendo el sistema...")
    u, R = solve_system_direct(K, f, fixed_dofs, fixed_values)
    print(">>> Cálculo finalizado con éxito <<<")

    # 8. Guardar resultado
    print("Generando gráfico...")
    plot_deformed_mesh(
        mesh, 
        elements, 
        u, 
        scale_factor=500.0, 
        title="Presa de Gravedad - Deformada por Presión Hidrostática", 
        save_filename="presa_hidrostatica.png"
    )

if __name__ == "__main__":
    main()