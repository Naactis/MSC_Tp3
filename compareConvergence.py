"""Comparacion de convergencia FEM para CST, LST, Q4 y Q8.

Ejecutar desde la raiz del proyecto:
    python compareConvergence.py

Lee los CSV generados previamente por runConvergence.py, incluida la energia. No ejecuta FEM,
no modifica las mallas y no necesita cambiar GravityDam.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CONVERGENCE_DIR = PROJECT_ROOT / "results" / "convergence"
COMPARISON_DIR = CONVERGENCE_DIR / "comparison"
ELEMENT_TYPES = ("CST", "LST", "Q4", "Q8")

# Puntos de evaluacion comunes. Ahora se verifican contra los CSV.
DISPLACEMENT_POINT = (3.0, 30.0)
STRESS_POINT = (2.0, 10.3)

REQUIRED_FIELDS = (
    "strain_energy_j",
    "delta_energy_percent",
    "energy_balance_percent",
    "ux_point_x_m",
    "ux_point_y_m",
    "stress_point_x_m",
    "stress_point_y_m",
    "element_type",
    "nx",
    "ny",
    "nodes",
    "elements",
    "dofs",
    "ux_crest_mm",
    "sigma_y_probe_mpa",
    "delta_ux_percent",
    "delta_sigma_y_percent",
)


def read_convergence_csv(element_type):
    """Lee y valida los datos de una familia de elementos."""
    csv_path = CONVERGENCE_DIR / element_type / f"convergence{element_type}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encontro {csv_path}. Ejecutar antes runConvergence.py."
        )

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Faltan columnas en {csv_path}: {sorted(missing)}. "
                "Volver a ejecutar runConvergence.py con meshConvergence.py actualizado "
                "para las cuatro formulaciones."
            )

        rows = []
        for row in reader:
            if row["element_type"].strip().upper() != element_type:
                raise ValueError(f"Tipo de elemento incorrecto en {csv_path}")
            converted = dict(row)
            for key in ("nx", "ny", "nodes", "elements", "dofs"):
                converted[key] = int(row[key])
            for key in ("ux_crest_mm", "sigma_y_probe_mpa", "strain_energy_j",
                        "energy_balance_percent", "ux_point_x_m", "ux_point_y_m",
                        "stress_point_x_m", "stress_point_y_m"):
                converted[key] = float(row[key])
            for key in ("delta_ux_percent", "delta_sigma_y_percent", "delta_energy_percent"):
                converted[key] = float(row[key]) if row[key].strip() else None
            rows.append(converted)

    if not rows:
        raise ValueError(f"El CSV esta vacio: {csv_path}")
    rows.sort(key=lambda r: r["dofs"])
    if any(b["dofs"] <= a["dofs"] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"Los GDL no son estrictamente crecientes para {element_type}")
    return rows


def plot_comparison(data, value_key, y_label, title, output_name):
    """Superpone las cuatro curvas frente a los GDL totales."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for element_type, rows in data.items():
        dofs = [row["dofs"] for row in rows]
        values = [row[value_key] for row in rows]
        ax.plot(dofs, values, marker="o", linewidth=1.8, markersize=5.5,
                label=element_type)

    # Escala logaritmica solo para distribuir mejor valores de 48 a 2050 GDL.
    ax.set_xscale("log")
    ax.set_xlabel("Grados de libertad totales (escala logaritmica)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Elemento")
    fig.tight_layout()

    output_path = COMPARISON_DIR / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_comparison_csv(data):
    """Une los cuatro CSV en una sola tabla para el informe."""
    output_path = COMPARISON_DIR / "convergenceComparison.csv"
    fields = [
        "element_type", "nx", "ny", "nodes", "elements", "dofs",
        "ux_crest_mm", "sigma_y_probe_mpa", "strain_energy_j",
        "delta_ux_percent", "delta_sigma_y_percent", "delta_energy_percent",
        "energy_balance_percent", "ux_point_x_m", "ux_point_y_m",
        "stress_point_x_m", "stress_point_y_m",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for element_type, rows in data.items():
            for row in rows:
                writer.writerow({key: "" if row[key] is None else row[key]
                                 for key in fields})
    return output_path


def main():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    data = {element_type: read_convergence_csv(element_type)
            for element_type in ELEMENT_TYPES}

    # Verifica que todos tengan las mismas divisiones nominales de malla.
    reference_levels = [(r["nx"], r["ny"]) for r in data[ELEMENT_TYPES[0]]]
    for element_type, rows in data.items():
        levels = [(r["nx"], r["ny"]) for r in rows]
        if levels != reference_levels:
            raise ValueError(
                f"Los niveles de {element_type} no coinciden con CST: {levels}"
            )

    # Los puntos fisicos deben ser exactamente iguales entre las familias.
    # Esto evita comparar tensiones correspondientes a posiciones distintas.
    for element_type, rows in data.items():
        for row in rows:
            ux_point = (row["ux_point_x_m"], row["ux_point_y_m"])
            stress_point = (row["stress_point_x_m"], row["stress_point_y_m"])
            if (any(abs(a - b) > 1e-9 for a, b in zip(ux_point, DISPLACEMENT_POINT))
                    or any(abs(a - b) > 1e-9 for a, b in zip(stress_point, STRESS_POINT))):
                raise ValueError(
                    f"Puntos de evaluacion distintos en CSV de {element_type}: "
                    f"ux={ux_point}, sigma={stress_point}. "
                    "Regenerar todos los CSV con la misma configuracion."
                )

    print("\nCOMPARACION FEM: CST, LST, Q4 Y Q8")
    print(f"Punto ux: {DISPLACEMENT_POINT} m")
    print(f"Punto sigma_y: {STRESS_POINT} m (verificado en los CSV)")
    print("\nUltimo refinamiento por tipo de elemento:")
    print(f"{'Tipo':<5} {'GDL':>6} {'ux [mm]':>13} {'sigma_y [MPa]':>17} "
          f"{'Energia [J]':>13} {'delta E':>10}")
    for element_type, rows in data.items():
        row = rows[-1]
        de = row["delta_energy_percent"]
        delta_e = "--" if de is None else f"{de:.3f}%"
        print(f"{element_type:<5} {row['dofs']:>6} {row['ux_crest_mm']:>13.6f} "
              f"{row['sigma_y_probe_mpa']:>17.6f} "
              f"{row['strain_energy_j']:>13.6f} {delta_e:>10}")

    displacement_plot = plot_comparison(
        data, "ux_crest_mm",
        f"Desplazamiento horizontal ux en {DISPLACEMENT_POINT} [mm]",
        "Convergencia del desplazamiento: CST, LST, Q4 y Q8",
        "comparisonDisplacement.png",
    )
    stress_plot = plot_comparison(
        data, "sigma_y_probe_mpa",
        f"Tension vertical sigma_y en {STRESS_POINT} [MPa]",
        "Convergencia de la tension: CST, LST, Q4 y Q8",
        "comparisonStress.png",
    )
    energy_plot = plot_comparison(
        data, "strain_energy_j", "Energia de deformacion [J]",
        "Convergencia energetica: CST, LST, Q4 y Q8",
        "comparisonEnergy.png",
    )
    csv_path = save_comparison_csv(data)

    print("\nArchivos guardados:")
    for path in (displacement_plot, stress_plot, energy_plot, csv_path):
        print(f"  {path}")
    print("\nNota: las variaciones entre mallas NO son errores exactos.")


if __name__ == "__main__":
    main()
