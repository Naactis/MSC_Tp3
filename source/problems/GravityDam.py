
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Malla (incluye los materiales definidos en el archivo)
from source.mesh.mesh import Mesh

# Elementos finitos
from source.elements.cst import CST
from source.elements.lst import LST
from source.elements.q4 import Q4
from source.elements.q8 import Q8

# Ensamblaje y resolucion
from source.systems.assembly import assemble_stiffness
from source.systems.bc import solve_system_direct

# Cargas
from source.loads.bodyforces import compute_body_forces
from source.loads.hydrostatic import compute_hydrostatic_forces

# Geometria y verificaciones
from source.mesh.metrics import compute_area
from source.output.reports import report_loads, report_equilibrium

# Visualizacion
from source.output.plot import plot_deformed_mesh

# Desplazamientos
from source.output.postprocess import compute_displacements
from source.output.reports import report_displacements
from source.output.displacementPlot import plot_displacement_fields

# Tensiones
from source.output.stressPostprocess import compute_strain_stress
from source.output.reports import report_strain_stress
from source.output.stressPlot import plot_stress_fields

# Estabilidad al vuelco
from source.output.stability import (
    analytic_stability,
    fem_stability,
    report_stability
)

# =============================================================
# 1. CONFIGURACION DEL PROBLEMA
# =============================================================

@dataclass
class DamConfig:
    """
    Parametros fisicos y de visualizacion de la presa.
    """

    mesh_path: Path

    thickness: float = 1.0

    # Densidad del hormigon [kg/m3]. Si es None, se toma
    # del material definido en el archivo de malla.
    rho_concrete: float = None
    rho_water: float = 1000.0

    gravity: float = 9.81
    water_level: float = 28.0

    height: float = 30.0
    base_width: float = 18.0
    crest_width: float = 3.0

    x_upstream: float = 0.0

    scale_factor: float = 500.0
    plot: bool = True


# =============================================================
# 2. CREACION DE ELEMENTOS
# =============================================================

def create_elements(mesh, thickness):
    """
    Instancia los elementos segun el tipo y el material
    definidos para cada uno en el archivo de malla.
    """

    # Mesh.read ya normaliza los nombres (TRI3 -> CST, etc.)
    # y verifica el orden de interpolacion y la cantidad de nodos.
    element_classes = {
        "CST": CST,
        "LST": LST,
        "Q4": Q4,
        "Q8": Q8
    }

    elements = []

    for e, (global_id, element_type, connectivity) in enumerate(zip(
        mesh.element_global_ids,
        mesh.element_types,
        mesh.connectivity
    )):

        element_class = element_classes[element_type]

        # Material asignado al elemento en el archivo de malla
        material = mesh.element_material(e)

        element = element_class(
            global_id,
            connectivity,
            mesh,
            material,
            thickness
        )

        elements.append(element)

    if not elements:
        raise ValueError(
            "La malla no contiene elementos reconocidos"
        )

    return elements


# =============================================================
# 3. EMPOTRAMIENTO DE LA BASE
# =============================================================

def base_boundary_conditions(mesh, y_base=0.0):
    """
    Fija desplazamientos horizontales y verticales
    de todos los nodos ubicados sobre la base.
    """

    fixed_dofs = []
    fixed_values = []

    for nid in range(len(mesh.nodes)):

        node = mesh.node(nid)

        if abs(node.y - y_base) < 1e-6:

            dofs = mesh.node_global_dofs(nid)

            fixed_dofs.extend(dofs)
            fixed_values.extend([0.0, 0.0])

    if not fixed_dofs:
        raise ValueError(
            "No se encontraron nodos de la base"
        )

    return fixed_dofs, fixed_values


# =============================================================
# 4. RESOLUCION DEL PROBLEMA DE LA PRESA
# =============================================================

def run_gravity_dam(config):
    """
    Ejecuta un analisis completo de la presa.

    Etapas:
        1. Lectura de malla.
        2. Instanciacion de elementos.
        3. Ensamblaje de rigidez.
        4. Integracion de cargas.
        5. Verificaciones fisicas.
        6. Condiciones de borde.
        7. Resolucion del sistema.
        8. Verificacion de equilibrio.
        9. Visualizacion.

    Retorna un diccionario con los resultados principales.
    """

    # ---------------------------------------------------------
    # 1. Leer la malla
    # ---------------------------------------------------------

    print("Cargando malla...")

    mesh = Mesh()
    mesh.read(str(config.mesh_path))
    mesh.dofs_per_node = 2

    # ---------------------------------------------------------
    # 2. Instanciar elementos
    # ---------------------------------------------------------

    elements = create_elements(
        mesh,
        config.thickness
    )

    print(
        f"Malla: {len(mesh.nodes)} nodos, "
        f"{len(elements)} elementos"
    )

    # Densidad del hormigon: la del archivo de malla, salvo que
    # se indique otra en la configuracion. La verificacion del
    # peso teorico supone una presa de un unico material.
    if config.rho_concrete is None:
        densities = {element.material.rho for element in elements}

        if len(densities) != 1:
            raise ValueError(
                "La verificacion del peso propio supone un unico "
                f"material, pero la malla tiene densidades {densities}"
            )

        rho_concrete = densities.pop()
    else:
        rho_concrete = config.rho_concrete

    if rho_concrete <= 0.0:
        raise ValueError("La densidad del hormigon debe ser positiva")

    # ---------------------------------------------------------
    # 3. Ensamblar matriz global de rigidez
    # ---------------------------------------------------------

    print("Ensamblando rigidez global...")

    K = assemble_stiffness(mesh, elements)

    # ---------------------------------------------------------
    # 4. Calcular fuerzas nodales equivalentes
    # ---------------------------------------------------------

    print("Calculando peso propio...")

    f_body = compute_body_forces(
        mesh,
        elements,
        rho_concrete=rho_concrete,
        g=config.gravity
    )

    print("Calculando presion hidrostatica...")

    f_hydro = compute_hydrostatic_forces(
        mesh,
        elements,
        rho_water=config.rho_water,
        g=config.gravity,
        y_water_level=config.water_level,
        x_upstream=config.x_upstream
    )

    # Vector total de cargas
    f = f_body + f_hydro

    # ---------------------------------------------------------
    # 5. Verificar geometria y cargas
    # ---------------------------------------------------------

    area = compute_area(elements)

    report_loads(
        mesh,
        area,
        f_body,
        f_hydro,
        rho_concrete=rho_concrete,
        rho_water=config.rho_water,
        g=config.gravity,
        water_level=config.water_level,
        thickness=config.thickness,
        base_width=config.base_width,
        crest_width=config.crest_width,
        dam_height=config.height
    )

    # ---------------------------------------------------------
    # 6. Aplicar condiciones de borde
    # ---------------------------------------------------------

    fixed_dofs, fixed_values = base_boundary_conditions(mesh)

    print(
        f"Grados de libertad restringidos: {len(fixed_dofs)}"
    )

    # ---------------------------------------------------------
    # 7. Resolver sistema
    # ---------------------------------------------------------

    print("Resolviendo el sistema...")

    U, _ = solve_system_direct(
        K,
        f,
        fixed_dofs,
        fixed_values
    )

    U = np.asarray(U).reshape(-1)

    # ---------------------------------------------------------
    # 8. Verificar equilibrio global
    # ---------------------------------------------------------

    reactions = report_equilibrium(
        mesh,
        K,
        U,
        f,
        fixed_dofs
    )

    # ---------------------------------------------------------
    # 8b. Estabilidad al vuelco
    # ---------------------------------------------------------

    # Momentos respecto del pie aguas abajo:
    #   - analiticos, a partir de la geometria y las densidades
    #   - del modelo FEM (cargas nodales y reacciones de la base)
    stability_analytic = analytic_stability(
        height=config.height,
        base_width=config.base_width,
        crest_width=config.crest_width,
        water_level=config.water_level,
        rho_concrete=rho_concrete,
        rho_water=config.rho_water,
        g=config.gravity,
        thickness=config.thickness
    )

    stability_fem = fem_stability(
        mesh,
        f_body,
        f_hydro,
        reactions,
        x_toe=config.x_upstream + config.base_width
    )

    stability_summary = report_stability(
        stability_analytic,
        stability_fem,
        base_width=config.base_width,
        x_upstream=config.x_upstream
    )

    # ---------------------------------------------------------
    # 9. Postprocesamiento de desplazamientos
    # ---------------------------------------------------------

    print("Procesando desplazamientos nodales...")

    # Se recuperan las componentes y magnitudes de los
    # desplazamientos calculados por el solver.
    displacements = compute_displacements(mesh, U)

    # Se muestran los resultados maximos en la terminal.
    report_displacements(displacements)


    # ---------------------------------------------------------
    # 10. Postprocesamiento de deformaciones y tensiones
    # ---------------------------------------------------------


    print("Calculando deformaciones y tensiones...")

    stress_results = compute_strain_stress(
        mesh,
        elements,
        U
    )

    print(
        f"Puntos de evaluacion: "
        f"{len(stress_results['stresses'])}"
    )

    report_strain_stress(stress_results)

    # ---------------------------------------------------------
    # 11. Visualizacion de resultados
    # ---------------------------------------------------------

    if config.plot:

        # Grafico de la malla original y deformada.
        # Los desplazamientos se amplifican solamente
        # para facilitar su visualizacion.
        plot_deformed_mesh(
            mesh,
            elements,
            U,
            scale_factor=config.scale_factor,
            title="Presa de gravedad - Peso propio + agua",
            save_filename=f"deformedMesh{config.mesh_path.stem}.png"
        )

        # Mapas de desplazamiento horizontal, vertical
        # y modulo del desplazamiento total.
        # Los valores se representan en milimetros,
        # sin amplificacion.
        plot_displacement_fields(
            mesh,
            elements,
            U,
            subdivisions=10,
            title=f"Presa de gravedad - {config.mesh_path.stem}",
            save_filename=f"displacements{config.mesh_path.stem}.png"
        )

        # Se representan las tensiones normales, cortantes
        # y la tension principal mayor del plano XY.
        #
        # Las tensiones se expresan en MPa.
        # La geometria no se amplifica.
        
        plot_stress_fields(
            mesh,
            elements,
            U,
            subdivisions=10,
            title=f"Presa de gravedad - {config.mesh_path.stem}",
            save_filename=f"stresses{config.mesh_path.stem}.png"
        )

    print("\n>>> ANALISIS DE LA PRESA COMPLETADO <<<")

    # ---------------------------------------------------------
    # 12. Devolver resultados
    # ---------------------------------------------------------

    return {
        "mesh": mesh,
        "elements": elements,
        "K": K,
        "F": f,
        "U": U,
        "reactions": reactions,
        "F_body": f_body,
        "F_hydro": f_hydro,
        "displacements": displacements,
        "stresses": stress_results,
        "stability": {
            "analytic": stability_analytic,
            "fem": stability_fem,
            **stability_summary
        }
    }



