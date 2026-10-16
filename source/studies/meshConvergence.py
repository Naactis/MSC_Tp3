"""Estudio de convergencia h para la presa de gravedad.

Reutiliza el generador de mallas y el analisis FEM existentes. Compara
valores en coordenadas fisicas comunes y energia elastica, sin modificar el solver.

Convenciones:
- Desplazamientos guardados en mm.
- Tensiones guardadas en MPa.
- Energia de deformacion guardada en J (para el espesor configurado).
- h_nominal es una medida orientativa del tamano de malla, no el diametro
  geometrico exacto de cada elemento.
- La diferencia relativa entre refinamientos NO es el error exacto FEM.
"""

from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from source.mesh.generator import write_dam_mesh
from source.mesh.mesh import Mesh, normalize_element_type
from source.problems.GravityDam import DamConfig, run_gravity_dam


@dataclass(frozen=True)
class ConvergenceConfig:
    # El material se lee de una malla existente del mismo proyecto.
    material_mesh_path: Path
    results_dir: Path
    element_type: str = "Q8"
    levels: tuple[tuple[int, int], ...] = ((3, 5), (6, 10), (9, 15))

    # Punto interior donde se comparan las tensiones de todos los refinamientos.
    stress_point: tuple[float, float] = (2.0, 10.5)

    height: float = 30.0
    base_width: float = 18.0
    crest_width: float = 3.0
    water_level: float = 28.0
    rho_water: float = 1000.0
    gravity: float = 9.81
    thickness: float = 1.0
    x_upstream: float = 0.0


def _inverse_mapping(element, point, tol=1.0e-9, max_iterations=20):
    """Calcula (xi, eta) tal que x(xi, eta) = point mediante Newton.

    Usa las funciones de forma *reales* del elemento para el mapeo
    isoparametrico. Funciona con CST, LST, Q4 y Q8, incluso si el
    elemento cuadratico tuviese lados curvos. Devuelve None si no
    encuentra una coordenada natural valida dentro del elemento.
    """
    node_ids = element.node_global_ids
    coords = np.array([
        [element.mesh.node(node_id).x, element.mesh.node(node_id).y]
        for node_id in node_ids
    ], dtype=float)
    target = np.asarray(point, dtype=float)
    triangular = len(node_ids) in (3, 6)
    natural = np.array([1.0 / 3.0, 1.0 / 3.0] if triangular else [0.0, 0.0])
    delta = 1.0e-6

    for _ in range(max_iterations):
        xi, eta = natural
        shape = np.asarray(element.shape_functions(xi, eta), dtype=float).reshape(-1)
        current = shape @ coords
        residual = current - target

        if np.linalg.norm(residual, ord=np.inf) < tol:
            break

        # Jacobiano del mapeo de geometria: columnas dx/dxi y dx/deta.
        shape_x_plus = np.asarray(element.shape_functions(xi + delta, eta), dtype=float).reshape(-1)
        shape_x_minus = np.asarray(element.shape_functions(xi - delta, eta), dtype=float).reshape(-1)
        shape_y_plus = np.asarray(element.shape_functions(xi, eta + delta), dtype=float).reshape(-1)
        shape_y_minus = np.asarray(element.shape_functions(xi, eta - delta), dtype=float).reshape(-1)
        dshape_dxi = (shape_x_plus - shape_x_minus) / (2.0 * delta)
        dshape_deta = (shape_y_plus - shape_y_minus) / (2.0 * delta)
        jacobian = np.column_stack((dshape_dxi @ coords, dshape_deta @ coords))

        try:
            correction = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            return None

        natural -= correction
        if not np.all(np.isfinite(natural)):
            return None
    else:
        return None

    xi, eta = natural
    margin = 1.0e-7
    if triangular:
        inside = (xi >= -margin and eta >= -margin and xi + eta <= 1.0 + margin)
    else:
        inside = (-1.0 - margin <= xi <= 1.0 + margin and
                  -1.0 - margin <= eta <= 1.0 + margin)
    return (float(xi), float(eta)) if inside else None


def stress_at_point(elements, displacement_vector, point):
    """Evalua (sigma_x, sigma_y, tau_xy) en un punto fisico [Pa].

    Se busca el elemento que contiene al punto y se evalua directamente
    sigma = C B Ue. No hay extrapolacion ni promediado nodal.
    Elegir un punto estrictamente interior y alejado de las esquinas.
    """
    u = np.asarray(displacement_vector, dtype=float).reshape(-1)
    target = np.asarray(point, dtype=float)

    for element in elements:
        coords = np.array([
            [element.mesh.node(nid).x, element.mesh.node(nid).y]
            for nid in element.node_global_ids
        ], dtype=float)
        if np.any(target < coords.min(axis=0) - 1.0e-8):
            continue
        if np.any(target > coords.max(axis=0) + 1.0e-8):
            continue

        local = _inverse_mapping(element, point)
        if local is None:
            continue

        xi, eta = local
        b_matrix, det_j = element.B_matrix(xi, eta)
        if det_j <= 0.0:
            raise ValueError(f"Jacobiano no positivo en elemento {element.global_id}")
        dofs = np.asarray(element.global_dofs(), dtype=int)
        strain = np.asarray(b_matrix, dtype=float) @ u[dofs]
        constitutive = np.asarray(element.material.matriz_constitutiva(), dtype=float)
        return constitutive @ strain

    raise ValueError(f"No se encontro un elemento que contenga el punto {point}")


def displacement_at_node(mesh, displacement_vector, point, tol=1.0e-7):
    """Devuelve (ux, uy) [m] de un nodo situado en la coordenada pedida."""
    x, y = point
    for node_id, node in enumerate(mesh.nodes):
        if abs(node.x - x) < tol and abs(node.y - y) < tol:
            u = np.asarray(displacement_vector, dtype=float).reshape(-1)
            dofs = mesh.node_global_dofs(node_id)
            return float(u[dofs[0]]), float(u[dofs[1]])
    raise ValueError(f"No existe un nodo en {point}; elegir un punto comun a las mallas")


def compute_strain_energy(stiffness, displacement_vector, force_vector, rtol=1.0e-7):
    """Energia elastica total [J] y error de consistencia energetica [%].

    Se usa la matriz de rigidez GLOBAL ORIGINAL, antes de modificarla
    para imponer las condiciones de borde:

        E = 0.5 * U.T @ K @ U

    Para apoyos homogeneos (desplazamientos prescritos nulos), debe
    cumplirse 2*E = U.T @ F. Las reacciones no hacen trabajo porque
    los desplazamientos en los apoyos son cero.

    No es energia por unidad de espesor: K ya incluye el espesor.
    """
    u = np.asarray(displacement_vector, dtype=float).reshape(-1)
    f = np.asarray(force_vector, dtype=float).reshape(-1)
    if f.size != u.size or stiffness.shape != (u.size, u.size):
        raise ValueError("Dimensiones incompatibles entre K, U y F")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(f)):
        raise ValueError("U o F contienen valores no finitos")

    ku = np.asarray(stiffness @ u, dtype=float).reshape(-1)
    energy = 0.5 * float(np.dot(u, ku))
    external_work = float(np.dot(u, f))
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"Energia de deformacion no valida: {energy} J")

    scale = max(abs(2.0 * energy), abs(external_work), 1.0e-12)
    relative_error = abs(2.0 * energy - external_work) / scale
    if relative_error > rtol:
        raise ValueError(
            "La identidad 2*E = U.T@F no se verifica: "
            f"diferencia relativa {100.0 * relative_error:.3e}%. "
            "Comprobar condiciones de borde y matriz K original."
        )
    return energy, 100.0 * relative_error


def _relative_change_percent(previous, current):
    """Cambio relativo respecto del valor actual: no es el error verdadero."""
    if previous is None:
        return None
    scale = max(abs(current), 1.0e-30)
    return 100.0 * abs(current - previous) / scale


def _save_csv(rows, path):
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})


def _plot_convergence(rows, results_dir, element_type, stress_point):
    dofs = [r["dofs"] for r in rows]
    displacements = [r["ux_crest_mm"] for r in rows]
    stresses = [r["sigma_y_probe_mpa"] for r in rows]
    energies = [r["strain_energy_j"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(dofs, displacements, "o-", linewidth=1.6)
    ax.set_xlabel("Grados de libertad totales")
    ax.set_ylabel("Desplazamiento ux en (3, 30) [mm]")
    ax.set_title(f"Convergencia de desplazamientos - {element_type}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"convergenceDisplacement{element_type}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(dofs, stresses, "o-", linewidth=1.6)
    ax.set_xlabel("Grados de libertad totales")
    ax.set_ylabel("Tension sigma_y [MPa]")
    ax.set_title(f"Convergencia en ({stress_point[0]}, {stress_point[1]}) m - {element_type}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"convergenceStress{element_type}.png", dpi=180)
    plt.close(fig)

    # La energia es una magnitud integral sobre todo el dominio, menos
    # sensible que la tension puntual a la ubicacion de una arista.
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(dofs, energies, "o-", linewidth=1.6)
    ax.set_xlabel("Grados de libertad totales")
    ax.set_ylabel("Energia de deformacion [J]")
    ax.set_title(f"Convergencia de energia - {element_type}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"convergenceEnergy{element_type}.png", dpi=180)
    plt.close(fig)


def run_mesh_convergence(config: ConvergenceConfig):
    """Genera, resuelve y compara una secuencia de mallas de la presa.

    Genera un CSV y tres PNG en config.results_dir. No altera main.py ni
    cambia el ensamblaje y las condiciones de borde existentes.
    """
    element_type = normalize_element_type(config.element_type)
    if not config.levels:
        raise ValueError("Se necesita al menos un nivel de refinamiento")
    if any(nx < 1 or ny < 1 for nx, ny in config.levels):
        raise ValueError("nx y ny deben ser positivos")
    element_counts = [nx * ny for nx, ny in config.levels]
    if any(b <= a for a, b in zip(element_counts, element_counts[1:])):
        raise ValueError("Los niveles deben tener cantidad creciente de celdas")

    output_dir = Path(config.results_dir)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Reutiliza las propiedades E, nu, rho y plane_state de un material
    # existente. Asi no duplicamos la definicion del hormigon.
    reference_mesh = Mesh()
    reference_mesh.read(str(config.material_mesh_path))
    material = reference_mesh.element_material(0)

    crest_point = (config.crest_width, config.height)
    rows = []
    previous_ux = None
    previous_sigma_y = None
    previous_energy = None

    print(f"\nESTUDIO DE CONVERGENCIA - {element_type}")
    print(f"Punto de desplazamientos: {crest_point} m")
    print(f"Punto interior de tensiones: {config.stress_point} m\n")

    for nx, ny in config.levels:
        mesh_path = mesh_dir / f"dam_{element_type}_{nx}x{ny}.mesh"
        write_dam_mesh(
            filename=str(mesh_path),
            element_type=element_type,
            nx=nx,
            ny=ny,
            material=material,
            height=config.height,
            base_width=config.base_width,
            crest_width=config.crest_width,
        )

        dam_config = DamConfig(
            mesh_path=mesh_path,
            thickness=config.thickness,
            rho_water=config.rho_water,
            gravity=config.gravity,
            water_level=config.water_level,
            height=config.height,
            base_width=config.base_width,
            crest_width=config.crest_width,
            x_upstream=config.x_upstream,
            plot=False,  # No generar ni abrir tres figuras en cada iteracion.
        )

        # run_gravity_dam conserva sus verificaciones y reportes, pero
        # evitamos imprimirlos repetidamente durante el barrido de mallas.
        with redirect_stdout(io.StringIO()):
            result = run_gravity_dam(dam_config)

        mesh = result["mesh"]
        elements = result["elements"]
        u = np.asarray(result["U"], dtype=float).reshape(-1)
        ux_crest, uy_crest = displacement_at_node(mesh, u, crest_point)
        sigma_x, sigma_y, tau_xy = stress_at_point(elements, u, config.stress_point)

        # La energia se calcula con los K, U y F del mismo analisis.
        # run_gravity_dam devuelve K antes de la aplicacion de apoyos.
        energy_j, energy_balance_percent = compute_strain_energy(
            result["K"], u, result["F"]
        )

        nodal_norm = np.hypot(u[0::2], u[1::2])
        # h nominal: maxima division horizontal en la base o vertical.
        nominal_h = max(config.base_width / nx, config.height / ny)
        row = {
            "element_type": element_type,
            "nx": nx,
            "ny": ny,
            "nodes": len(mesh.nodes),
            "elements": len(elements),
            "dofs": mesh.number_of_dofs(),
            "ux_point_x_m": crest_point[0],
            "ux_point_y_m": crest_point[1],
            "stress_point_x_m": config.stress_point[0],
            "stress_point_y_m": config.stress_point[1],
            "h_nominal_m": nominal_h,
            "ux_crest_mm": ux_crest * 1.0e3,
            "uy_crest_mm": uy_crest * 1.0e3,
            "max_u_nodal_mm": float(nodal_norm.max()) * 1.0e3,
            "sigma_x_probe_mpa": float(sigma_x) / 1.0e6,
            "sigma_y_probe_mpa": float(sigma_y) / 1.0e6,
            "tau_xy_probe_mpa": float(tau_xy) / 1.0e6,
            "strain_energy_j": energy_j,
            "energy_balance_percent": energy_balance_percent,
            "delta_ux_percent": _relative_change_percent(previous_ux, ux_crest),
            "delta_sigma_y_percent": _relative_change_percent(previous_sigma_y, sigma_y),
            "delta_energy_percent": _relative_change_percent(previous_energy, energy_j),
        }
        rows.append(row)
        previous_ux = ux_crest
        previous_sigma_y = sigma_y
        previous_energy = energy_j

        delta_u = "--" if row["delta_ux_percent"] is None else f"{row['delta_ux_percent']:.3f}%"
        delta_s = "--" if row["delta_sigma_y_percent"] is None else f"{row['delta_sigma_y_percent']:.3f}%"
        delta_e = "--" if row["delta_energy_percent"] is None else f"{row['delta_energy_percent']:.3f}%"
        print(f"{nx:>2}x{ny:<2} | elem={len(elements):>4} | GDLat={row['dofs']:>5} | "
              f"ux={row['ux_crest_mm']:+.6f} mm (cambio {delta_u}) | "
              f"sigma_y={row['sigma_y_probe_mpa']:+.6f} MPa (cambio {delta_s}) | "
              f"Energia={energy_j:.6f} J (cambio {delta_e})")

    csv_path = output_dir / f"convergence{element_type}.csv"
    _save_csv(rows, csv_path)
    _plot_convergence(rows, output_dir, element_type, config.stress_point)
    print(f"\nTabla CSV: {csv_path}")
    print(f"Graficos de convergencia: {output_dir}\n")
    return rows
