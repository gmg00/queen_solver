#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Importa le funzioni necessarie dal modulo functions
from functions.functions import (
    load_and_preprocess_image,
    detect_grid,
    estimate_grid_size,
    extract_cell_size,
    extract_colors,
    convert_color_matrix,
    solver_final
)

def save_solved_matrix(indexed_matrix, color_dict, output_path):
    """
    Versione modificata di plot_indexed_matrix per salvare l'immagine 
    invece di mostrarla a schermo.
    """
    grid_size = indexed_matrix.shape[0]
    color_matrix = indexed_matrix[:, :, 0]
    
    rgb_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    for row in range(grid_size):
        for col in range(grid_size):
            color_index = color_matrix[row, col]
            rgb_image[row, col] = color_dict.get(color_index, [255, 255, 255])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_image)
    
    for row in range(grid_size):
        for col in range(grid_size):
            value = indexed_matrix[row, col, 1]
            if value == 1:
                ax.text(col, row, '⚫', ha='center', va='center', fontsize=40, color='black')
            elif value == -1:
                ax.text(col, row, 'X', ha='center', va='center', fontsize=30, color='black')

    ax.set_xticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)
    
    # Rimuove i bordi bianchi inutili e salva
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def main():
    # Controllo degli argomenti
    if len(sys.argv) != 2:
        print("Uso: python3 solve_queens.py <path_immagine>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    if not os.path.isfile(image_path):
        print(f"Errore: Il file '{image_path}' non esiste.")
        sys.exit(1)
        
    print(f"Elaborazione di {image_path}...")
    
    # 1. Computer Vision
    try:
        image, thresh, gray = load_and_preprocess_image(image_path)
        x, y, w, h = detect_grid(thresh)
        grid_size = estimate_grid_size(gray, x, y, w, h)
        cell_w, cell_h = extract_cell_size(w, h, grid_size)
        color_matrix = extract_colors(image, x, y, w, h, cell_w, cell_h, grid_size)
    except Exception as e:
        print(f"Errore durante l'elaborazione dell'immagine: {e}")
        sys.exit(1)

    # 2. Conversione e setup matrice logica
    indexed_matrix, color_dict = convert_color_matrix(color_matrix)

    # 3. Risoluzione
    print("Ricerca della soluzione in corso...")
    result_matrix = solver_final(indexed_matrix, color_dict)
    
    if result_matrix is False:
        print("Errore: Impossibile trovare una soluzione per questo livello.")
        sys.exit(1)
        
    # 4. Salvataggio del risultato
    # Genera il nome del file di output (es. livello1.png -> livello1_solved.png)
    base_name, ext = os.path.splitext(image_path)
    output_path = f"{base_name}_solved{ext}"
    
    save_solved_matrix(result_matrix, color_dict, output_path)
    print(f"Completato! Immagine risolta salvata in: {output_path}")

if __name__ == "__main__":
    main()