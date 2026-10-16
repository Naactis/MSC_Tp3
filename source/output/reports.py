import numpy as np


def force_resultants(mesh, forces):
    """
    Calcula Fx, Fy y Mz de un vector de fuerzas nodales.

    El momento se calcula respecto del origen:
        Mz = suma(x*Fy - y*Fx)
    """

    Fx = 0.0
    Fy = 0.0
    Mz = 0.0

    for nid in range(len(mesh.nodes)):

        node = mesh.node(nid)
        dof_x, dof_y = mesh.node_global_dofs(nid)

        fx = forces[dof_x]
        fy = forces[dof_y]

        Fx += fx
        Fy += fy

        Mz += node.x * fy - node.y * fx

    return np.array([Fx, Fy, Mz])


def report_loads(
    mesh,
    area_fem,
    f_body,
    f_hydro,
    rho_concrete,
    rho_water,
    g,
    water_level,
    thickness,
    base_width,
    crest_width,
    dam_height
):
    """
    Compara las cargas integradas mediante FEM
    con las resultantes analiticas de la presa.
    """

    # Area teorica del trapecio
    area_theory = (
        0.5 * (base_width + crest_width) * dam_height
    )

    # Peso teorico
    weight_theory = (
        area_theory * rho_concrete * g * thickness
    )

    # Resultante hidrostatica teorica
    hydro_theory = (
        0.5 * rho_water * g * water_level**2 * thickness
    )

    # Resultantes calculadas mediante FEM
    body_result = force_resultants(mesh, f_body)
    hydro_result = force_resultants(mesh, f_hydro)

    weight_fem = -body_result[1]
    hydro_fem = hydro_result[0]

    # El empuje actua a h/3 desde la base.
    moment_hydro_theory = -hydro_theory * water_level / 3.0

    print("\n" + "-"*60)
    print("VERIFICACION DE CARGAS")

    print(
        f"Area FEM / teorica: "
        f"{area_fem:.4f} / {area_theory:.4f} m2"
    )

    print(
        f"Peso FEM / teorico: "
        f"{weight_fem/1e6:.6f} / "
        f"{weight_theory/1e6:.6f} MN"
    )

    print(
        f"Empuje FEM / teorico: "
        f"{hydro_fem/1e6:.6f} / "
        f"{hydro_theory/1e6:.6f} MN"
    )

    print(
        f"Momento hidrostatico FEM / teorico: "
        f"{hydro_result[2]/1e6:.6f} / "
        f"{moment_hydro_theory/1e6:.6f} MN.m"
    )

    # Comprobaciones automaticas de resultantes
    np.testing.assert_allclose(
        area_fem, area_theory, rtol=1e-5, atol=1e-7
    )

    np.testing.assert_allclose(
        weight_fem, weight_theory, rtol=1e-5, atol=1e-3
    )

    np.testing.assert_allclose(
        hydro_fem, hydro_theory, rtol=1e-5, atol=1e-3
    )

    np.testing.assert_allclose(
        hydro_result[2], moment_hydro_theory,
        rtol=1e-5, atol=1e-3
    )


def report_equilibrium(mesh, K, U, f, fixed_dofs):
    """
    Recupera reacciones y verifica equilibrio global.

    R = K @ U - f

    Las reacciones no nulas deben encontrarse
    en los grados de libertad restringidos.
    """

    U = np.asarray(U).reshape(-1)
    f = np.asarray(f).reshape(-1)

    # Vector residual de equilibrio
    R = K @ U - f

    fixed = np.array(fixed_dofs, dtype=int)

    free = np.setdiff1d(
        np.arange(U.size),
        fixed
    )

    # Comprobar equilibrio en grados de libertad libres
    force_scale = max(
        float(np.linalg.norm(f, ord=np.inf)),
        1.0
    )

    np.testing.assert_allclose(
        R[free],
        0.0,
        rtol=0.0,
        atol=1e-8 * force_scale
    )

    # Conservar solamente las reacciones de apoyo
    reactions = np.zeros_like(R)
    reactions[fixed] = R[fixed]

    # Resultantes de fuerzas externas y reacciones
    external = force_resultants(mesh, f)
    supports = force_resultants(mesh, reactions)

    residual = external + supports

    print("\n" + "-"*60)
    print("EQUILIBRIO GLOBAL")

    print(
        f"Fuerzas externas Fx, Fy: "
        f"{external[0]/1e6:+.6f}, "
        f"{external[1]/1e6:+.6f} MN"
    )

    print(
        f"Reacciones Rx, Ry: "
        f"{supports[0]/1e6:+.6f}, "
        f"{supports[1]/1e6:+.6f} MN"
    )

    print(
        f"Error de equilibrio Fx, Fy: "
        f"{residual[0]:+.3e}, "
        f"{residual[1]:+.3e} N"
    )

    print(
        f"Error de equilibrio Mz: "
        f"{residual[2]:+.3e} N.m"
    )

    # Control de equilibrio global de fuerzas y momentos
    scale_F = max(np.linalg.norm(external[:2]), 1.0)
    scale_M = max(abs(external[2]), 1.0)

    assert np.linalg.norm(residual[:2]) < 1e-7 * scale_F
    assert abs(residual[2]) < 1e-7 * scale_M

    return reactions

# =============================================================
# INFORME DE DESPLAZAMIENTOS
# =============================================================

def report_displacements(displacements):
    """
    Muestra los desplazamientos maximos calculados por el MEF.

    Los valores se expresan en metros y milimetros.
    Se informa el desplazamiento nodal sin amplificacion.
    """

    max_ux = displacements["max_ux"]
    max_uy = displacements["max_uy"]
    max_u = displacements["max_magnitude"]

    print("\n" + "-" * 60)
    print("RESULTADOS DE DESPLAZAMIENTOS")

    print(
        f"Maximo |ux|: {abs(max_ux['value']):.6e} m "
        f"({abs(max_ux['value'])*1000:.6f} mm)"
    )
    print(
        f"  Valor con signo: {max_ux['value']:+.6e} m"
    )
    print(
        f"  Nodo {max_ux['node_id']} "
        f"en ({max_ux['x']:.3f}, {max_ux['y']:.3f}) m"
    )

    print(
        f"Maximo |uy|: {abs(max_uy['value']):.6e} m "
        f"({abs(max_uy['value'])*1000:.6f} mm)"
    )
    print(
        f"  Valor con signo: {max_uy['value']:+.6e} m"
    )
    print(
        f"  Nodo {max_uy['node_id']} "
        f"en ({max_uy['x']:.3f}, {max_uy['y']:.3f}) m"
    )

    print(
        f"Maximo |u|:  {max_u['value']:.6e} m "
        f"({max_u['value']*1000:.6f} mm)"
    )
    print(
        f"  Nodo {max_u['node_id']} "
        f"en ({max_u['x']:.3f}, {max_u['y']:.3f}) m"
    )

    print("-" * 60)
