#!/usr/bin/env python3

import json
import base64
from pathlib import Path

def extraer_imagenes_notebook(notebook_path, carpeta_destino, tipo_red):
    
    print(f"\nProcesando: {notebook_path}")
    print(f"Destino: {carpeta_destino}")
    
    # crear carpeta
    Path(carpeta_destino).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado: {notebook_path}")
        return 0
    except json.JSONDecodeError:
        print(f"Error: JSON inválido: {notebook_path}")
        return 0
    
    imagen_count = 0
    
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        if 'outputs' not in cell:
            continue
        
        for output_idx, output in enumerate(cell.get('outputs', [])):
            if 'data' not in output:
                continue
            
            # PNG en outputs
            if 'image/png' in output['data']:
                try:
                    img_data = output['data']['image/png']
                    img_bytes = base64.b64decode(img_data)
                    
                    celda_num = cell_idx + 1
                    filename = f"{tipo_red}_celda{celda_num:02d}_grafica{output_idx}.png"
                    filepath = Path(carpeta_destino) / filename
                    
                    with open(filepath, 'wb') as img_file:
                        img_file.write(img_bytes)
                    
                    imagen_count += 1
                    tamanio_kb = len(img_bytes) / 1024
                    print(f"  {filename} ({tamanio_kb:.1f} KB)")
                    
                except Exception as e:
                    print(f"  Error en celda {cell_idx}: {str(e)}")
    
    print(f"Guardadas {imagen_count} gráficas\n")
    return imagen_count


def main():
    print("\nExtrayendo gráficas...\n")
    
    notebooks = [
        ('classification_cifar10_mlp.ipynb', 'graficas_MLP', 'MLP'),
        ('classification_cifar10_cnn.ipynb', 'graficas_CNN', 'CNN'),
    ]
    
    total_imagenes = 0
    
    for notebook_file, carpeta_dest, tipo_red in notebooks:
        notebook_path = Path(notebook_file)
        
        if not notebook_path.exists():
            print(f"No encontrado: {notebook_file}")
            continue
        
        count = extraer_imagenes_notebook(str(notebook_path), carpeta_dest, tipo_red)
        total_imagenes += count
    
    print(f"\nTotal: {total_imagenes} gráficas extraídas")


if __name__ == '__main__':
    main()
