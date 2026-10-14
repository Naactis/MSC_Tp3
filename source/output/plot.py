import os
import matplotlib.pyplot as plt
import numpy as np


def plot_deformed_mesh(mesh, elements, u, scale_factor=200.0, title="Malla Deformada", save_filename=None):
    """
    Grafica la malla original y deformada. Si se especifica save_filename,
    guarda la imagen en la carpeta 'output/'.
    """
    plt.figure(figsize=(7, 7))
    
    for elem in elements:
        node_ids = elem.node_global_ids
        
        if len(node_ids) == 4:
            order = [0, 1, 2, 3, 0]
        elif len(node_ids) == 8:
            order = [0, 4, 1, 5, 2, 6, 3, 7, 0]
        else:
            order = list(range(len(node_ids))) + [0]
            
        x_orig, y_orig = [], []
        x_def, y_def = [], []
        
        for idx in order:
            nid = node_ids[idx]
            node = mesh.node(nid)
            x0, y0 = node.x, node.y
            ux = u[2 * nid]
            uy = u[2 * nid + 1]
            
            x_orig.append(x0)
            y_orig.append(y0)
            x_def.append(x0 + scale_factor * ux)
            y_def.append(y0 + scale_factor * uy)
            
        plt.plot(x_orig, y_orig, 'k--', alpha=0.4, linewidth=1.2, label='Original' if elem == elements[0] else "")
        plt.plot(x_def, y_def, 'b-o', linewidth=1.5, markersize=4, label='Deformada' if elem == elements[0] else "")

    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{title} (Escala: {scale_factor}x)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Guardar en la carpeta output si se solicita
    if save_filename:
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", save_filename)
        plt.savefig(filepath, dpi=300)
        print(f"Gráfico guardado exitosamente en: {filepath}")

    plt.show()