"""
Analisis de estabilidad al vuelco de la presa de gravedad.

Todos los momentos se toman respecto del PIE AGUAS ABAJO de la presa,
el punto (x_pie, y_base) = (x_upstream + B, 0), que es el punto
alrededor del cual la presa tenderia a volcar.

Convencion de signos: momento antihorario positivo.
    - El peso propio (actua a la izquierda del pie) genera un momento
      antihorario: ESTABILIZANTE (> 0).
    - El empuje del agua (hacia +x, por encima de la base) genera un
      momento horario: DESESTABILIZANTE (< 0).

Factor de seguridad al vuelco:
    FS = M_estabilizante / M_desestabilizante   (en valor absoluto)

Ademas se calcula la posicion de la resultante de las reacciones en la
base. Si cae dentro del tercio central, la base esta toda comprimida
(criterio clasico de presas de gravedad).

No se consideran subpresion ni otras cargas: solo peso propio y
empuje hidrostatico sobre el paramento aguas arriba.
"""

import numpy as np


# =============================================================
# 1. MOMENTOS ANALITICOS
# =============================================================

def analytic_stability(
    height,
    base_width,
    crest_width,
    water_level,
    rho_concrete,
    rho_water,
    g=9.81,
    thickness=1.0
):
    """
    Calcula, a partir de la geometria y las densidades, las fuerzas
    y los momentos respecto del pie aguas abajo.

    Geometria: trapecio con paramento aguas arriba vertical (x = 0),
    base B, cresta C y altura H.

    Peso propio:
        A  = (B + C) / 2 * H
        W  = rho_c * g * A * t
        x_G = (B^2 + B*C + C^2) / (3 (B + C))   (centroide, desde x = 0)
        M_est = W * (B - x_G)                   (brazo: distancia al pie)

    Empuje hidrostatico (h = nivel del agua):
        P = 1/2 * rho_w * g * h^2 * t
        actua a h/3 desde la base
        M_desest = P * h / 3
    """

    if not (height > 0 and base_width > 0 and crest_width > 0):
        raise ValueError("La geometria de la presa debe ser positiva")

    if not (0.0 <= water_level <= height):
        raise ValueError("El nivel del agua debe estar entre 0 y la altura")

    B, C, H, h = base_width, crest_width, height, water_level

    # Peso propio y su punto de aplicacion
    area = 0.5 * (B + C) * H
    weight = rho_concrete * g * area * thickness
    x_centroid = (B**2 + B * C + C**2) / (3.0 * (B + C))
    arm_weight = B - x_centroid

    # Empuje del agua y su punto de aplicacion
    thrust = 0.5 * rho_water * g * h**2 * thickness
    arm_thrust = h / 3.0

    m_stabilizing = weight * arm_weight
    m_overturning = thrust * arm_thrust

    return {
        "area": area,
        "weight": weight,
        "x_centroid": x_centroid,
        "arm_weight": arm_weight,
        "thrust": thrust,
        "arm_thrust": arm_thrust,
        "M_stabilizing": m_stabilizing,
        "M_overturning": m_overturning,
        "FS_overturning": m_stabilizing / m_overturning if m_overturning > 0 else np.inf,
    }


# =============================================================
# 2. MOMENTOS DEL MODELO FEM
# =============================================================

def moment_about_point(mesh, forces, x0, y0):
    """
    Momento (antihorario +) de un vector de fuerzas nodales
    respecto del punto (x0, y0):

        M = suma[ (x - x0) * Fy - (y - y0) * Fx ]
    """

    forces = np.asarray(forces, dtype=float).reshape(-1)
    moment = 0.0

    for nid in range(len(mesh.nodes)):
        node = mesh.node(nid)
        dof_x, dof_y = mesh.node_global_dofs(nid)
        moment += (node.x - x0) * forces[dof_y] - (node.y - y0) * forces[dof_x]

    return moment


def fem_stability(mesh, f_body, f_hydro, reactions, x_toe, y_base=0.0):
    """
    Calcula los momentos respecto del pie a partir del modelo FEM:

        - de las fuerzas nodales del peso propio  (estabilizante)
        - de las fuerzas nodales del agua         (desestabilizante)
        - de las reacciones de vinculo en la base

    Tambien ubica la resultante de las reacciones verticales en la base:
        x_R = suma(x_i * Ry_i) / suma(Ry_i)
    """

    reactions = np.asarray(reactions, dtype=float).reshape(-1)

    m_body = moment_about_point(mesh, f_body, x_toe, y_base)
    m_hydro = moment_about_point(mesh, f_hydro, x_toe, y_base)
    m_reactions = moment_about_point(mesh, reactions, x_toe, y_base)

    # Resultantes de las reacciones
    rx = float(np.sum(reactions[0::2]))
    ry = float(np.sum(reactions[1::2]))

    # Punto de paso de la resultante vertical sobre la base
    x_coords = np.array([node.x for node in mesh.nodes])
    if abs(ry) > 0.0:
        x_resultant = float(np.sum(x_coords * reactions[1::2]) / ry)
    else:
        x_resultant = np.nan

    return {
        "M_body": m_body,
        "M_hydro": m_hydro,
        "M_reactions": m_reactions,
        "Rx": rx,
        "Ry": ry,
        "x_resultant": x_resultant,
        "FS_overturning": m_body / -m_hydro if m_hydro < 0 else np.inf,
    }


# =============================================================
# 3. INFORME
# =============================================================

def report_stability(analytic, fem, base_width, x_upstream=0.0, check=True):
    """
    Muestra la comparacion entre los momentos analiticos y los del
    modelo FEM, el factor de seguridad al vuelco y la posicion de la
    resultante en la base.

    Si check=True, verifica automaticamente que:
        - el momento del peso FEM coincida con el analitico,
        - el momento del agua FEM coincida con el analitico,
        - el momento de las reacciones equilibre a ambos:
              M_reacciones = -(M_est - M_desest)
    """

    m_est = analytic["M_stabilizing"]
    m_des = analytic["M_overturning"]

    # Momento neto de las cargas respecto del pie (antihorario +)
    m_net_analytic = m_est - m_des

    # Las reacciones deben equilibrar el momento neto de las cargas
    m_reactions_expected = -m_net_analytic

    def rel_error(value, reference):
        return abs(value - reference) / abs(reference) * 100.0

    print("\n" + "-" * 60)
    print("ESTABILIDAD AL VUELCO (momentos respecto del pie aguas abajo)")
    print("Convencion: antihorario positivo")
    print("-" * 60)

    print(
        f"Peso propio W = {analytic['weight']/1e6:.4f} MN, "
        f"brazo = {analytic['arm_weight']:.3f} m "
        f"(centroide en x = {x_upstream + analytic['x_centroid']:.3f} m)"
    )
    print(
        f"Empuje      P = {analytic['thrust']/1e6:.4f} MN, "
        f"brazo = {analytic['arm_thrust']:.3f} m (h/3)"
    )

    print(f"\n{'':28s}{'Analitico':>14s}{'FEM':>14s}{'Error':>10s}")
    print(
        f"{'Momento estabilizante':28s}"
        f"{m_est/1e6:>+11.4f} MN.m"
        f"{fem['M_body']/1e6:>+11.4f} MN.m"
        f"{rel_error(fem['M_body'], m_est):>9.3f}%"
    )
    print(
        f"{'Momento desestabilizante':28s}"
        f"{-m_des/1e6:>+11.4f} MN.m"
        f"{fem['M_hydro']/1e6:>+11.4f} MN.m"
        f"{rel_error(fem['M_hydro'], -m_des):>9.3f}%"
    )
    print(
        f"{'Momento de las reacciones':28s}"
        f"{m_reactions_expected/1e6:>+11.4f} MN.m"
        f"{fem['M_reactions']/1e6:>+11.4f} MN.m"
        f"{rel_error(fem['M_reactions'], m_reactions_expected):>9.3f}%"
    )

    residual = fem["M_body"] + fem["M_hydro"] + fem["M_reactions"]
    print(
        f"\nSuma de momentos FEM (cargas + reacciones): "
        f"{residual:+.3e} N.m"
    )

    print(
        f"\nFactor de seguridad al vuelco: "
        f"FS = {analytic['FS_overturning']:.3f} (analitico), "
        f"{fem['FS_overturning']:.3f} (FEM)"
    )

    # Resultante de las reacciones en la base y tercio central
    x_r = fem["x_resultant"] - x_upstream
    eccentricity = x_r - base_width / 2.0
    in_middle_third = abs(eccentricity) <= base_width / 6.0

    print(
        f"Resultante de la reaccion vertical en x = {fem['x_resultant']:.3f} m "
        f"(excentricidad {eccentricity:+.3f} m respecto del centro de la base)"
    )
    print(
        f"Tercio central de la base: x entre "
        f"{x_upstream + base_width/3:.3f} y {x_upstream + 2*base_width/3:.3f} m -> "
        + ("DENTRO (base toda comprimida segun la teoria clasica)"
           if in_middle_third else
           "FUERA (aparecen tracciones en la base segun la teoria clasica)")
    )
    print("-" * 60)

    if check:
        np.testing.assert_allclose(fem["M_body"], m_est, rtol=1e-6)
        np.testing.assert_allclose(fem["M_hydro"], -m_des, rtol=1e-6)
        np.testing.assert_allclose(
            fem["M_reactions"], m_reactions_expected, rtol=1e-6
        )

    return {
        "M_net": m_net_analytic,
        "eccentricity": eccentricity,
        "in_middle_third": in_middle_third,
    }
