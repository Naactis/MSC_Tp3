"""
Graficos del estudio de geometria (PNG para el informe).

1. geometryFS.png:          FS al vuelco vs ancho de base, una curva por cresta.
2. geometryEfficiency.png:  FS vs area de hormigon, ensanchando la base o la cresta
                            a partir de la presa original.
3. geometryStresses.png:    compresion maxima y traccion lejos de las esquinas
                            vs ancho de base (dos paneles, un eje cada uno).

Paleta categorica validada (3 series). Cada serie lleva ademas su propio
marcador y una etiqueta directa, para no depender solo del color.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Colores por rol
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categoricos 1-3
MARKERS = ["o", "s", "^"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)
    ax.title.set_color(TEXT_PRIMARY)


def _new_figure(ncols=1, width=7.2):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.6), facecolor=SURFACE)
    axes = axes if ncols > 1 else [axes]
    for ax in axes:
        _style_axes(ax)
    return fig, axes


def _series_by_crest(rows, crest_widths):
    out = []
    for crest in crest_widths:
        sub = sorted((r for r in rows if r["crest_m"] == crest), key=lambda r: r["base_m"])
        if sub:
            out.append((crest, sub))
    return out


def _direct_label(ax, x, y, text):
    # Etiqueta al final de la curva, en tinta de texto (no en el color de la serie)
    ax.annotate(
        text, (x, y), xytext=(6, 0), textcoords="offset points",
        va="center", fontsize=9, color=TEXT_PRIMARY
    )


def plot_geometry_study(rows, minimum_rows, cfg, results_dir):

    reference = next((r for r in rows if r["is_reference"]), None)
    series = _series_by_crest(rows, cfg.crest_widths)

    # ---------------------------------------------------------
    # 1. FS vs ancho de base
    # ---------------------------------------------------------
    fig, (ax,) = _new_figure(width=7.6)

    for target in cfg.target_fs:
        ax.axhline(target, color=REFERENCE, linestyle="--", linewidth=1.2)
        ax.annotate(f"FS = {target:g}", (max(cfg.base_widths) + 2.4, target),
                    xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=9, color=TEXT_SECONDARY)

    for i, (crest, sub) in enumerate(series):
        b = [r["base_m"] for r in sub]
        fs = [r["FS_fem"] for r in sub]
        ax.plot(b, fs, color=SERIES[i], marker=MARKERS[i], markersize=6,
                linewidth=2, label=f"Cresta C = {crest:g} m")
        _direct_label(ax, b[-1], fs[-1], f"C = {crest:g} m")

    if reference:
        ax.plot(reference["base_m"], reference["FS_fem"], marker="o", markersize=13,
                markerfacecolor="none", markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.5)
        ax.annotate("Original", (reference["base_m"], reference["FS_fem"]),
                    xytext=(-10, 14), textcoords="offset points", ha="right",
                    fontsize=9, color=TEXT_PRIMARY)

    ax.set_xlabel("Ancho de base B [m]")
    ax.set_ylabel("Factor de seguridad al vuelco")
    ax.set_title("FS al vuelco segun el ancho de base")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper left")
    ax.set_xlim(right=max(cfg.base_widths) + 2.5)
    fig.tight_layout()
    fig.savefig(results_dir / "geometryFS.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------
    # 2. Eficiencia: FS vs area, ensanchando base o cresta
    # ---------------------------------------------------------
    # Solo tiene sentido si hay al menos dos puntos en alguna direccion
    if reference and (
        sum(r["crest_m"] == reference["crest_m"] and r["base_m"] >= reference["base_m"] for r in rows) >= 2
        or sum(r["base_m"] == reference["base_m"] and r["crest_m"] >= reference["crest_m"] for r in rows) >= 2
    ):
        widen_base = sorted(
            (r for r in rows if r["crest_m"] == reference["crest_m"]
             and r["base_m"] >= reference["base_m"]),
            key=lambda r: r["area_m2"])
        widen_crest = sorted(
            (r for r in rows if r["base_m"] == reference["base_m"]
             and r["crest_m"] >= reference["crest_m"]),
            key=lambda r: r["area_m2"])

        fig, (ax,) = _new_figure(width=7.6)
        for i, (label, sub) in enumerate([
            ("Ensanchando la base", widen_base),
            ("Ensanchando la cresta", widen_crest),
        ]):
            if len(sub) < 2:
                continue
            a = [r["area_m2"] for r in sub]
            fs = [r["FS_fem"] for r in sub]
            ax.plot(a, fs, color=SERIES[i], marker=MARKERS[i], markersize=6,
                    linewidth=2, label=label)
            _direct_label(ax, a[-1], fs[-1], label.split()[-1])

        ax.plot(reference["area_m2"], reference["FS_fem"], marker="o", markersize=13,
                markerfacecolor="none", markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.5)
        ax.annotate("Original", (reference["area_m2"], reference["FS_fem"]),
                    xytext=(0, -18), textcoords="offset points", ha="center",
                    fontsize=9, color=TEXT_PRIMARY)

        ax.set_xlabel("Area de hormigon por metro de presa [m$^2$]")
        ax.set_ylabel("Factor de seguridad al vuelco")
        ax.set_title("Aumento del FS por unidad de hormigon agregado")
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper left")
        ax.margins(x=0.12, y=0.12)
        fig.tight_layout()
        fig.savefig(results_dir / "geometryEfficiency.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)

    # ---------------------------------------------------------
    # 3. Tensiones vs ancho de base (dos paneles)
    # ---------------------------------------------------------
    fig, (ax_c, ax_t) = _new_figure(ncols=2, width=11.0)

    for i, (crest, sub) in enumerate(series):
        b = [r["base_m"] for r in sub]
        comp = [r["sigma2_far_MPa"] for r in sub]
        tens = [r["sigma1_far_MPa"] for r in sub]
        ax_c.plot(b, comp, color=SERIES[i], marker=MARKERS[i], markersize=6,
                  linewidth=2, label=f"C = {crest:g} m")
        ax_t.plot(b, tens, color=SERIES[i], marker=MARKERS[i], markersize=6,
                  linewidth=2, label=f"C = {crest:g} m")

    ax_t.axhline(0.0, color=REFERENCE, linewidth=1.0)

    ax_c.set_title(f"Compresion maxima ($\\sigma_2$) a mas de {cfg.corner_exclusion:g} m de las esquinas")
    ax_c.set_ylabel("Tension [MPa]")
    ax_t.set_title(f"Traccion maxima ($\\sigma_1$) a mas de {cfg.corner_exclusion:g} m de las esquinas")
    ax_t.set_ylabel("Tension [MPa]")
    for ax in (ax_c, ax_t):
        ax.set_xlabel("Ancho de base B [m]")
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(results_dir / "geometryStresses.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
