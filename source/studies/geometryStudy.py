"""
Estudio de variaciones de geometria de la presa de gravedad.

Para cada combinacion de ancho de base B y ancho de cresta C:
    1. Estabilidad al vuelco (analitica y FEM): FS, area de hormigon,
       posicion de la resultante en la base.
    2. Tensiones del modelo FEM, usando solo indicadores que convergen
       al refinar la malla. Se excluyen los puntos a menos de
       `corner_exclusion` metros del talon (0,0) y del pie (B,0): cerca de
       esas esquinas la tension es singular (crece sin limite al refinar)
       por el empotramiento de la base. Fuera de esa zona:
         - compresion maxima: minimo de la tension principal sigma_2;
         - traccion maxima: maximo de la tension principal sigma_1.
       Se verifico con mallas Q8 12x20, 18x30 y 24x40 (B = 12, 14 y 18 m)
       que estos indicadores cambian menos de 1-2 % al refinar, mientras
       que los extremos globales siguen creciendo.
    3. Tensiones clasicas en la base (ley trapecial, teoria de vigas):
           sigma = -N/B * (1 -/+ 6 e / B)   en talon / pie
       con N la reaccion vertical y e la excentricidad de la resultante.

Ademas calcula, con la formula analitica, el ancho de base minimo que
alcanza un factor de seguridad objetivo para cada ancho de cresta.

Convencion: tensiones de traccion positivas, compresion negativa.
"""

from __future__ import annotations

import csv
import io
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from source.mesh.generator import write_dam_mesh
from source.output.stability import analytic_stability
from source.problems.GravityDam import DamConfig, run_gravity_dam


@dataclass
class GeometryStudyConfig:
    material: object                     # LinearElastic con rho
    results_dir: Path

    base_widths: tuple = (12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0)
    crest_widths: tuple = (3.0, 6.0, 9.0)

    # Geometria de referencia (la del enunciado)
    reference_base: float = 18.0
    reference_crest: float = 3.0

    height: float = 30.0
    water_level: float = 28.0
    rho_water: float = 1000.0
    gravity: float = 9.81
    thickness: float = 1.0

    # Malla: Q8 12x20 segun el estudio de convergencia
    element_type: str = "Q8"
    nx: int = 12
    ny: int = 20

    # Radio de exclusion alrededor de las esquinas de la base [m]
    corner_exclusion: float = 3.0

    # Factores de seguridad objetivo para el ancho de base minimo
    target_fs: tuple = (1.5, 2.0)

    plot: bool = True


# =============================================================
# 1. INDICADORES DE TENSION
# =============================================================

def stress_indicators(stress_results, base_width, corner_exclusion):
    """
    Extrae de los puntos de Gauss los indicadores de tension:
        sigma1_far: traccion maxima lejos de las esquinas [Pa]
        sigma2_far: compresion maxima lejos de las esquinas [Pa]
        sigma1_global, sigma2_global: extremos en todo el dominio, solo
            informativos (dominados por las singularidades) [Pa]
    """

    coords = stress_results["physical_coords"]
    principal = stress_results["principal_stresses"]

    dist_heel = np.hypot(coords[:, 0], coords[:, 1])
    dist_toe = np.hypot(coords[:, 0] - base_width, coords[:, 1])
    far = (dist_heel > corner_exclusion) & (dist_toe > corner_exclusion)

    if not np.any(far):
        raise ValueError("No hay puntos de Gauss fuera de la zona excluida")

    i_t = np.argmax(np.where(far, principal[:, 0], -np.inf))
    i_c = np.argmin(np.where(far, principal[:, 1], np.inf))

    return {
        "sigma1_far": float(principal[i_t, 0]),
        "sigma1_far_x": float(coords[i_t, 0]),
        "sigma1_far_y": float(coords[i_t, 1]),
        "sigma2_far": float(principal[i_c, 1]),
        "sigma2_far_x": float(coords[i_c, 0]),
        "sigma2_far_y": float(coords[i_c, 1]),
        "sigma1_global": float(principal[:, 0].max()),
        "sigma2_global": float(principal[:, 1].min()),
    }


def classical_base_stresses(vertical_force, x_resultant, base_width, thickness=1.0):
    """
    Tensiones verticales en los extremos de la base segun la ley
    trapecial (flexion compuesta):
        sigma = -N/(B t) * (1 -/+ 6 e / B)
    Talon: signo menos (la excentricidad hacia el pie lo descarga).
    Devuelve (sigma_talon, sigma_pie), compresion negativa.
    """

    e = x_resultant - base_width / 2.0
    mean = -vertical_force / (base_width * thickness)

    return (
        mean * (1.0 - 6.0 * e / base_width),
        mean * (1.0 + 6.0 * e / base_width),
    )


# =============================================================
# 2. ANCHO DE BASE MINIMO PARA UN FS OBJETIVO
# =============================================================

def minimum_base_for_fs(target_fs, crest_width, cfg, b_low=None, b_high=200.0):
    """
    Ancho de base minimo B tal que FS(B, C) = target_fs, por biseccion
    sobre la formula analitica (FS crece monotonamente con B).
    """

    def fs(base):
        return analytic_stability(
            height=cfg.height, base_width=base, crest_width=crest_width,
            water_level=cfg.water_level, rho_concrete=cfg.material.rho,
            rho_water=cfg.rho_water, g=cfg.gravity, thickness=cfg.thickness
        )["FS_overturning"]

    low = crest_width if b_low is None else b_low
    high = b_high

    if fs(low) >= target_fs:
        return low
    if fs(high) < target_fs:
        raise ValueError("No se alcanza el FS objetivo en el rango de bases")

    for _ in range(100):
        mid = 0.5 * (low + high)
        if fs(mid) >= target_fs:
            high = mid
        else:
            low = mid

    return high


# =============================================================
# 3. ESTUDIO PARAMETRICO
# =============================================================

def run_geometry_study(cfg: GeometryStudyConfig):
    """
    Resuelve la presa para cada (B, C), guarda una tabla CSV y,
    si cfg.plot es True, los graficos del estudio.

    Devuelve la lista de filas (diccionarios).
    """

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\nESTUDIO DE GEOMETRIA")
    print(f"Elemento {cfg.element_type} {cfg.nx}x{cfg.ny}, "
          f"exclusion de esquinas r = {cfg.corner_exclusion} m\n")
    print(f"{'B':>5}{'C':>5}{'Area':>8}{'FS':>7}{'x_R':>7}"
          f"{'s1 lejos':>10}{'s2 lejos':>10}{'s talon':>9}{'s pie':>8}   [m, m2, -, m, MPa]")

    with tempfile.TemporaryDirectory() as mesh_dir:
        for crest in cfg.crest_widths:
            for base in cfg.base_widths:

                if base <= crest:
                    continue

                mesh_path = Path(mesh_dir) / f"dam_B{base:g}_C{crest:g}.mesh"
                write_dam_mesh(
                    mesh_path, cfg.element_type, cfg.nx, cfg.ny, cfg.material,
                    height=cfg.height, base_width=base, crest_width=crest
                )

                dam_config = DamConfig(
                    mesh_path=mesh_path,
                    thickness=cfg.thickness,
                    rho_water=cfg.rho_water,
                    gravity=cfg.gravity,
                    water_level=cfg.water_level,
                    height=cfg.height,
                    base_width=base,
                    crest_width=crest,
                    plot=False
                )

                # run_gravity_dam conserva todas sus verificaciones
                # (cargas, equilibrio, momentos); se silencia su salida.
                with redirect_stdout(io.StringIO()):
                    result = run_gravity_dam(dam_config)

                analytic = result["stability"]["analytic"]
                fem = result["stability"]["fem"]
                stresses = stress_indicators(
                    result["stresses"], base, cfg.corner_exclusion
                )
                s_heel, s_toe = classical_base_stresses(
                    fem["Ry"], fem["x_resultant"], base, cfg.thickness
                )

                row = {
                    "base_m": base,
                    "crest_m": crest,
                    "area_m2": analytic["area"],
                    "weight_MN": analytic["weight"] / 1e6,
                    "M_stab_MNm": analytic["M_stabilizing"] / 1e6,
                    "M_overt_MNm": analytic["M_overturning"] / 1e6,
                    "FS_analytic": analytic["FS_overturning"],
                    "FS_fem": fem["FS_overturning"],
                    "x_resultant_m": fem["x_resultant"],
                    "in_middle_third": abs(fem["x_resultant"] - base / 2) <= base / 6,
                    "sigma1_far_MPa": stresses["sigma1_far"] / 1e6,
                    "sigma1_far_x_m": stresses["sigma1_far_x"],
                    "sigma1_far_y_m": stresses["sigma1_far_y"],
                    "sigma2_far_MPa": stresses["sigma2_far"] / 1e6,
                    "sigma2_far_x_m": stresses["sigma2_far_x"],
                    "sigma2_far_y_m": stresses["sigma2_far_y"],
                    "sigma1_global_MPa": stresses["sigma1_global"] / 1e6,
                    "sigma2_global_MPa": stresses["sigma2_global"] / 1e6,
                    "sigma_heel_classic_MPa": s_heel / 1e6,
                    "sigma_toe_classic_MPa": s_toe / 1e6,
                    "is_reference": (base == cfg.reference_base
                                     and crest == cfg.reference_crest),
                }
                rows.append(row)

                mark = "  <- original" if row["is_reference"] else ""
                print(f"{base:5.0f}{crest:5.0f}{row['area_m2']:8.0f}"
                      f"{row['FS_fem']:7.2f}{row['x_resultant_m']:7.2f}"
                      f"{row['sigma1_far_MPa']:+10.3f}{row['sigma2_far_MPa']:+10.3f}"
                      f"{row['sigma_heel_classic_MPa']:+9.3f}"
                      f"{row['sigma_toe_classic_MPa']:+8.3f}{mark}")

    # Ancho minimo de base para los FS objetivo
    print("\nAncho de base minimo para alcanzar el FS objetivo (formula analitica):")
    minimum_rows = []
    for target in cfg.target_fs:
        for crest in cfg.crest_widths:
            base_min = minimum_base_for_fs(target, crest, cfg)
            area = 0.5 * (base_min + crest) * cfg.height
            minimum_rows.append({
                "FS_target": target, "crest_m": crest,
                "base_min_m": base_min, "area_m2": area
            })
            print(f"  FS = {target:.1f}, C = {crest:4.1f} m -> "
                  f"B_min = {base_min:6.2f} m, area = {area:6.1f} m2")

    _save_csv(rows, results_dir / "geometryStudy.csv")
    _save_csv(minimum_rows, results_dir / "minimumBase.csv")
    print(f"\nTablas guardadas en: {results_dir}")

    if cfg.plot:
        from source.studies.geometryPlots import plot_geometry_study
        plot_geometry_study(rows, minimum_rows, cfg, results_dir)
        print(f"Graficos guardados en: {results_dir}")

    return rows, minimum_rows


def _save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
