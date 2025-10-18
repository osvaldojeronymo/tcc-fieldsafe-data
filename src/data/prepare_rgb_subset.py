
#!/usr/bin/env python3
"""
Pipeline robusto para preparação do dataset FieldSAFE RGB.

Funcionalidades:
- Varre múltiplas sequências e encontra pares imagem ↔ rótulo por ID
- Redimensiona imagens para 640×640 (máscaras com INTER_NEAREST)
- Suporta máscaras semânticas (PNG) ou rótulos YOLO (TXT)
- Converte máscaras → YOLO opcionalmente
- Split 70/15/15 por sequência via GroupShuffleSplit (evita vazamento)
- Gera manifests e relatório de estatísticas

Saída:
- fieldsafe_rgb_ready/images/ e labels/ (padronizados)
- fieldsafe_rgb_ready/split/{train,val,test}/{images,labels}
- fieldsafe_rgb_ready/manifests/manifest_{all,train,val,test}.csv
- fieldsafe_rgb_ready/relatorio_summary.txt
"""

import os
import argparse
import shutil
import glob
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

def ensure_dir(p):
    """Cria diretório se não existir."""
    os.makedirs(p, exist_ok=True)

def extract_id_from_filename(filename):
    """Extrai ID temporal do nome do arquivo RGB."""
    # Formato: rgb_1477386583897396523.png
    match = re.search(r'rgb_(\d+)', filename)
    return match.group(1) if match else None

def find_image_label_pairs(rgb_root, label_root=None, label_type="none"):
    """
    Encontra pares imagem ↔ rótulo por ID em todas as sequências.
    
    Returns:
        dict: {sequence: [(rgb_path, label_path, image_id), ...]}
    """
    pairs_by_sequence = defaultdict(list)
    
    for seq_dir in glob.glob(os.path.join(rgb_root, "*/")):
        sequence_name = os.path.basename(seq_dir.rstrip('/'))
        images_dir = os.path.join(seq_dir, "images")
        
        if not os.path.exists(images_dir):
            continue
            
        for rgb_file in glob.glob(os.path.join(images_dir, "rgb_*.png")):
            image_id = extract_id_from_filename(os.path.basename(rgb_file))
            if not image_id:
                continue
                
            label_path = None
            if label_root and label_type != "none":
                if label_type == "semantic":
                    # Procura máscara semântica correspondente
                    label_pattern = os.path.join(label_root, sequence_name, f"mask_{image_id}.png")
                elif label_type == "yolo":
                    # Procura arquivo YOLO correspondente
                    label_pattern = os.path.join(label_root, sequence_name, f"yolo_{image_id}.txt")
                
                if os.path.exists(label_pattern):
                    label_path = label_pattern
            
            pairs_by_sequence[sequence_name].append((rgb_file, label_path, image_id))
    
    return pairs_by_sequence

def resize_image(image_path, size=(640, 640)):
    """Redimensiona imagem RGB."""
    img = cv2.imread(image_path)
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def resize_mask(mask_path, size=(640, 640)):
    """Redimensiona máscara semântica preservando rótulos."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

def mask_to_yolo_bbox(mask, class_mapping=None):
    """
    Converte máscara semântica para bounding boxes YOLO.
    
    Returns:
        list: [(class_id, x_center, y_center, width, height), ...]
    """
    bboxes = []
    unique_classes = np.unique(mask)
    
    for class_val in unique_classes:
        if class_val == 0:  # Skip background
            continue
            
        # Encontra contornos da classe
        class_mask = (mask == class_val).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calcula bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Normaliza para formato YOLO (0-1)
            img_h, img_w = mask.shape
            x_center = (x + w/2) / img_w
            y_center = (y + h/2) / img_h
            width = w / img_w
            height = h / img_h
            
            # Mapeia classe (default: class_val - 1)
            class_id = class_val - 1 if not class_mapping else class_mapping.get(class_val, class_val - 1)
            
            bboxes.append((class_id, x_center, y_center, width, height))
    
    return bboxes

def process_data_pairs(pairs_by_sequence, outdir, resize_size, label_type, gen_yolo_from_mask):
    """
    Processa todos os pares imagem-rótulo, aplicando redimensionamento e conversões.
    """
    ensure_dir(os.path.join(outdir, "images"))
    ensure_dir(os.path.join(outdir, "labels"))
    
    processed_pairs = defaultdict(list)
    stats = {"total_images": 0, "total_labels": 0, "sequences": Counter()}
    
    for sequence, pairs in pairs_by_sequence.items():
        print(f"Processando sequência: {sequence} ({len(pairs)} pares)")
        
        for rgb_path, label_path, image_id in tqdm(pairs, desc=f"  {sequence}"):
            # Processa imagem RGB
            img_resized = resize_image(rgb_path, resize_size)
            output_img_name = f"{sequence}_{image_id}.png"
            output_img_path = os.path.join(outdir, "images", output_img_name)
            cv2.imwrite(output_img_path, img_resized)
            
            output_label_path = None
            
            # Processa rótulo se disponível
            if label_path and label_type != "none":
                output_label_name = f"{sequence}_{image_id}.txt"
                output_label_path = os.path.join(outdir, "labels", output_label_name)
                
                if label_type == "semantic":
                    # Redimensiona máscara
                    mask_resized = resize_mask(label_path, resize_size)
                    
                    if gen_yolo_from_mask:
                        # Converte máscara → YOLO
                        yolo_bboxes = mask_to_yolo_bbox(mask_resized)
                        with open(output_label_path, 'w') as f:
                            for bbox in yolo_bboxes:
                                f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                    else:
                        # Salva máscara redimensionada
                        output_label_path = output_label_path.replace('.txt', '.png')
                        cv2.imwrite(output_label_path, mask_resized)
                
                elif label_type == "yolo":
                    # Copia arquivo YOLO
                    shutil.copy2(label_path, output_label_path)
            
            processed_pairs[sequence].append((output_img_path, output_label_path, image_id))
            stats["sequences"][sequence] += 1
            stats["total_images"] += 1
            if output_label_path:
                stats["total_labels"] += 1
    
    return processed_pairs, stats

def create_splits_by_sequence(processed_pairs, outdir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Cria splits 70/15/15 usando GroupShuffleSplit para evitar vazamento entre sequências.
    """
    # Prepara dados para split
    all_samples = []
    groups = []
    
    for sequence, pairs in processed_pairs.items():
        for img_path, label_path, image_id in pairs:
            all_samples.append((img_path, label_path, image_id, sequence))
            groups.append(sequence)
    
    # Split por sequência
    gss = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=random_state)
    train_idx, temp_idx = next(gss.split(all_samples, groups=groups))
    
    # Split val/test
    temp_samples = [all_samples[i] for i in temp_idx]
    temp_groups = [groups[i] for i in temp_idx]
    
    val_size = val_ratio / (val_ratio + test_ratio)
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=(1-val_size), random_state=random_state)
    val_idx, test_idx = next(gss_val_test.split(temp_samples, groups=temp_groups))
    
    # Organiza splits
    splits = {
        "train": [all_samples[i] for i in train_idx],
        "val": [temp_samples[i] for i in val_idx],
        "test": [temp_samples[i] for i in test_idx]
    }
    
    # Cria diretórios e copia arquivos
    for split_name, samples in splits.items():
        split_images_dir = os.path.join(outdir, "split", split_name, "images")
        split_labels_dir = os.path.join(outdir, "split", split_name, "labels")
        ensure_dir(split_images_dir)
        ensure_dir(split_labels_dir)
        
        for img_path, label_path, image_id, sequence in samples:
            # Copia imagem
            shutil.copy2(img_path, split_images_dir)
            
            # Copia rótulo se existir
            if label_path and os.path.exists(label_path):
                shutil.copy2(label_path, split_labels_dir)
    
    return splits

def generate_manifests(splits, outdir):
    """Gera arquivos CSV de manifesto para cada split."""
    ensure_dir(os.path.join(outdir, "manifests"))
    
    all_samples = []
    for split_name, samples in splits.items():
        manifest_path = os.path.join(outdir, "manifests", f"manifest_{split_name}.csv")
        
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label_path", "image_id", "sequence", "split"])
            
            for img_path, label_path, image_id, sequence in samples:
                writer.writerow([img_path, label_path or "", image_id, sequence, split_name])
                all_samples.append((img_path, label_path, image_id, sequence, split_name))
    
    # Manifest completo
    with open(os.path.join(outdir, "manifests", "manifest_all.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label_path", "image_id", "sequence", "split"])
        writer.writerows(all_samples)

def generate_summary_report(stats, splits, outdir):
    """Gera relatório de estatísticas."""
    report_path = os.path.join(outdir, "relatorio_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== RELATÓRIO DE PROCESSAMENTO - FIELDSAFE RGB ===\n\n")
        
        f.write(f"Total de imagens processadas: {stats['total_images']}\n")
        f.write(f"Total de rótulos processados: {stats['total_labels']}\n\n")
        
        f.write("=== DISTRIBUIÇÃO POR SEQUÊNCIA ===\n")
        for seq, count in stats['sequences'].items():
            f.write(f"{seq}: {count} amostras\n")
        f.write("\n")
        
        f.write("=== DISTRIBUIÇÃO DOS SPLITS ===\n")
        for split_name, samples in splits.items():
            seq_counts = Counter([s[3] for s in samples])
            f.write(f"{split_name.upper()}: {len(samples)} amostras\n")
            for seq, count in seq_counts.items():
                f.write(f"  {seq}: {count}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de preparação do dataset FieldSAFE RGB")
    
    parser.add_argument("--rgb_root", required=True, help="Diretório raiz das imagens RGB extraídas")
    parser.add_argument("--label_root", help="Diretório raiz dos rótulos (opcional)")
    parser.add_argument("--outdir", required=True, help="Diretório de saída")
    parser.add_argument("--resize", nargs=2, type=int, default=[640, 640], help="Dimensões para redimensionamento")
    parser.add_argument("--label_type", choices=["none", "semantic", "yolo"], default="none", 
                       help="Tipo de rótulo: none, semantic (PNG), yolo (TXT)")
    parser.add_argument("--gen_yolo_from_mask", action="store_true", 
                       help="Converte máscaras semânticas para formato YOLO")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proporção de treino")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proporção de validação")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proporção de teste")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    
    args = parser.parse_args()
    
    print("=== PIPELINE DE PREPARAÇÃO FIELDSAFE RGB ===")
    print(f"RGB Root: {args.rgb_root}")
    print(f"Label Root: {args.label_root}")
    print(f"Output: {args.outdir}")
    print(f"Resize: {args.resize[0]}x{args.resize[1]}")
    print(f"Label Type: {args.label_type}")
    print(f"Generate YOLO from Mask: {args.gen_yolo_from_mask}")
    print()
    
    # 1. Encontra pares imagem-rótulo
    print("1. Encontrando pares imagem ↔ rótulo...")
    pairs_by_sequence = find_image_label_pairs(args.rgb_root, args.label_root, args.label_type)
    
    total_pairs = sum(len(pairs) for pairs in pairs_by_sequence.values())
    print(f"   Encontrados {total_pairs} pares em {len(pairs_by_sequence)} sequências")
    
    # 2. Processa dados
    print("2. Processando e redimensionando dados...")
    processed_pairs, stats = process_data_pairs(
        pairs_by_sequence, args.outdir, tuple(args.resize), 
        args.label_type, args.gen_yolo_from_mask
    )
    
    # 3. Cria splits
    print("3. Criando splits por sequência (evitando vazamento)...")
    splits = create_splits_by_sequence(
        processed_pairs, args.outdir, 
        args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    # 4. Gera manifests
    print("4. Gerando manifests...")
    generate_manifests(splits, args.outdir)
    
    # 5. Gera relatório
    print("5. Gerando relatório de estatísticas...")
    generate_summary_report(stats, splits, args.outdir)
    
    print(f"\n✅ Pipeline concluído! Dados prontos em: {args.outdir}")
    print(f"📊 Consulte o relatório: {os.path.join(args.outdir, 'relatorio_summary.txt')}")

if __name__ == "__main__":
    main()
