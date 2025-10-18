#!/usr/bin/env python3
"""
Pipeline completo de pré-processamento para o FieldSAFE RGB.

Implementa as etapas descritas na metodologia:
1. Redimensionamento: 640×640 pixels
2. Normalização: RGB para [0,1]
3. Organização por sequência temporal
4. Divisão estratificada 70/15/15

Saída compatível com os modelos ensemble (YOLOv5, SegNet, Autoencoder).
"""

import os
import argparse
import shutil
import glob
import csv
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

class FieldSAFEPreprocessor:
    """
    Classe principal para pré-processamento do dataset FieldSAFE RGB.
    
    Implementa pipeline completo de preparação seguindo metodologia descrita:
    - Redimensionamento padronizado
    - Normalização RGB
    - Organização por sequência
    - Divisão estratificada
    """
    
    def __init__(self, 
                 rgb_root: str,
                 output_dir: str,
                 target_size: Tuple[int, int] = (640, 640),
                 normalize_range: Tuple[float, float] = (0.0, 1.0),
                 splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 random_seed: int = 42):
        
        self.rgb_root = Path(rgb_root)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.normalize_range = normalize_range
        self.splits = splits
        self.random_seed = random_seed
        
        # Validação
        if not self.rgb_root.exists():
            raise ValueError(f"Diretório RGB não encontrado: {rgb_root}")
        
        if sum(splits) != 1.0:
            raise ValueError(f"Splits devem somar 1.0, mas somam {sum(splits)}")
        
        # Estatísticas de processamento
        self.stats = {
            "sequences_found": 0,
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "splits": {"train": 0, "val": 0, "test": 0}
        }
        
        print(f"=== FieldSAFE Preprocessor ===")
        print(f"RGB Root: {self.rgb_root}")
        print(f"Output: {self.output_dir}")
        print(f"Target Size: {self.target_size}")
        print(f"Normalize Range: {self.normalize_range}")
        print(f"Splits: Train={splits[0]:.1%}, Val={splits[1]:.1%}, Test={splits[2]:.1%}")
        print(f"Random Seed: {self.random_seed}")
        
    def extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """
        Extrai timestamp do nome do arquivo RGB.
        
        Args:
            filename: Nome do arquivo (ex: rgb_1477386583897396523.png)
            
        Returns:
            Timestamp como string ou None se não encontrar
        """
        match = re.search(r'rgb_(\d+)', filename)
        return match.group(1) if match else None
    
    def discover_sequences(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Descobre e organiza sequências temporais no dataset.
        
        Returns:
            Dict: {sequence_name: [(image_path, timestamp), ...]}
        """
        print("1. Descobrindo sequências temporais...")
        
        sequences = defaultdict(list)
        
        # Procura por diretórios de sequência
        for seq_dir in self.rgb_root.glob("*/"):
            if not seq_dir.is_dir():
                continue
                
            sequence_name = seq_dir.name
            images_dir = seq_dir / "images"
            
            if not images_dir.exists():
                print(f"   Aviso: {sequence_name} não tem diretório 'images'")
                continue
            
            # Encontra imagens RGB na sequência
            rgb_files = list(images_dir.glob("rgb_*.png"))
            
            if not rgb_files:
                print(f"   Aviso: {sequence_name} não tem imagens RGB")
                continue
            
            # Extrai timestamps e organiza
            for rgb_file in rgb_files:
                timestamp = self.extract_timestamp_from_filename(rgb_file.name)
                if timestamp:
                    sequences[sequence_name].append((str(rgb_file), timestamp))
            
            # Ordena por timestamp
            sequences[sequence_name].sort(key=lambda x: x[1])
            
            print(f"   {sequence_name}: {len(sequences[sequence_name])} imagens")
        
        self.stats["sequences_found"] = len(sequences)
        self.stats["total_images"] = sum(len(imgs) for imgs in sequences.values())
        
        print(f"   Total: {len(sequences)} sequências, {self.stats['total_images']} imagens")
        
        return dict(sequences)
    
    def preprocess_image(self, 
                        image_path: str, 
                        save_normalized: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica pré-processamento completo à imagem.
        
        Etapas:
        1. Carregamento
        2. Redimensionamento para target_size
        3. Normalização para [0,1] (opcional)
        
        Args:
            image_path: Caminho da imagem
            save_normalized: Se deve retornar versão normalizada
            
        Returns:
            Tuple: (imagem_original_resized, imagem_normalizada)
        """
        # 1. Carrega imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar: {image_path}")
        
        # Converte BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Redimensionamento para 640×640
        image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 3. Normalização RGB para [0,1]
        if save_normalized:
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Ajusta para range específico se necessário
            if self.normalize_range != (0.0, 1.0):
                min_val, max_val = self.normalize_range
                image_normalized = image_normalized * (max_val - min_val) + min_val
        else:
            image_normalized = image_resized
        
        return image_resized, image_normalized
    
    def process_sequences(self, sequences: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Dict]]:
        """
        Processa todas as sequências aplicando pré-processamento.
        
        Args:
            sequences: Dicionário com sequências descobertas
            
        Returns:
            Dict com metadados das imagens processadas
        """
        print("2. Processando e redimensionando imagens...")
        
        # Cria diretórios de saída
        processed_dir = self.output_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = processed_dir / "images"
        normalized_dir = processed_dir / "normalized"
        images_dir.mkdir(exist_ok=True)
        normalized_dir.mkdir(exist_ok=True)
        
        processed_sequences = defaultdict(list)
        
        for sequence_name, image_list in sequences.items():
            print(f"   Processando {sequence_name}...")
            
            sequence_processed = []
            
            for image_path, timestamp in tqdm(image_list, desc=f"  {sequence_name}"):
                try:
                    # Pré-processamento
                    img_resized, img_normalized = self.preprocess_image(image_path, save_normalized=True)
                    
                    # Nomes de saída
                    output_name = f"{sequence_name}_{timestamp}.png"
                    
                    # Salva imagem redimensionada (uint8)
                    output_path = images_dir / output_name
                    cv2.imwrite(str(output_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                    
                    # Salva imagem normalizada (float32 como uint8 scaled)
                    normalized_path = normalized_dir / output_name
                    img_normalized_uint8 = (img_normalized * 255).astype(np.uint8)
                    cv2.imwrite(str(normalized_path), cv2.cvtColor(img_normalized_uint8, cv2.COLOR_RGB2BGR))
                    
                    # Metadados
                    metadata = {
                        "original_path": image_path,
                        "output_path": str(output_path),
                        "normalized_path": str(normalized_path),
                        "timestamp": timestamp,
                        "sequence": sequence_name,
                        "original_size": cv2.imread(image_path).shape[:2][::-1],  # (W, H)
                        "processed_size": self.target_size,
                        "normalization_range": self.normalize_range
                    }
                    
                    sequence_processed.append(metadata)
                    self.stats["processed_images"] += 1
                    
                except Exception as e:
                    print(f"     Erro ao processar {image_path}: {e}")
                    self.stats["failed_images"] += 1
                    continue
            
            processed_sequences[sequence_name] = sequence_processed
        
        print(f"   Processadas: {self.stats['processed_images']} imagens")
        print(f"   Falhas: {self.stats['failed_images']} imagens")
        
        return dict(processed_sequences)
    
    def create_stratified_splits(self, 
                               processed_sequences: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Cria divisão estratificada 70/15/15 baseada em sequências.
        
        Usa GroupShuffleSplit para garantir que imagens da mesma sequência
        não apareçam em splits diferentes, evitando vazamento temporal.
        
        Args:
            processed_sequences: Sequências processadas
            
        Returns:
            Dict: {split_name: [metadata_list]}
        """
        print("3. Criando divisão estratificada por sequência...")
        
        # Prepara dados para split
        all_samples = []
        groups = []
        
        for sequence_name, samples in processed_sequences.items():
            for sample in samples:
                all_samples.append(sample)
                groups.append(sequence_name)
        
        print(f"   Total de {len(all_samples)} amostras em {len(set(groups))} sequências")
        
        # Calcula tamanhos dos splits
        train_ratio, val_ratio, test_ratio = self.splits
        
        # Primeiro split: train vs (val + test)
        gss1 = GroupShuffleSplit(
            n_splits=1, 
            test_size=(val_ratio + test_ratio), 
            random_state=self.random_seed
        )
        
        train_idx, temp_idx = next(gss1.split(all_samples, groups=groups))
        
        # Segundo split: val vs test
        temp_samples = [all_samples[i] for i in temp_idx]
        temp_groups = [groups[i] for i in temp_idx]
        
        val_size_in_temp = val_ratio / (val_ratio + test_ratio)
        
        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=(1 - val_size_in_temp),
            random_state=self.random_seed
        )
        
        val_idx, test_idx = next(gss2.split(temp_samples, groups=temp_groups))
        
        # Organiza splits finais
        splits = {
            "train": [all_samples[i] for i in train_idx],
            "val": [temp_samples[i] for i in val_idx], 
            "test": [temp_samples[i] for i in test_idx]
        }
        
        # Estatísticas dos splits
        for split_name, samples in splits.items():
            self.stats["splits"][split_name] = len(samples)
            
            # Conta sequências por split
            seq_counts = Counter([s["sequence"] for s in samples])
            print(f"   {split_name.upper()}: {len(samples)} amostras de {len(seq_counts)} sequências")
            
            for seq, count in seq_counts.most_common():
                print(f"     {seq}: {count} amostras")
        
        return splits
    
    def save_splits_to_directories(self, splits: Dict[str, List[Dict]]):
        """
        Organiza splits em diretórios separados para fácil acesso.
        
        Estrutura:
        output_dir/
        ├── split/
        │   ├── train/images/
        │   ├── val/images/
        │   └── test/images/
        └── split_normalized/
            ├── train/images/
            ├── val/images/
            └── test/images/
        """
        print("4. Organizando splits em diretórios...")
        
        # Diretórios para imagens originais e normalizadas
        split_dir = self.output_dir / "split"
        split_normalized_dir = self.output_dir / "split_normalized"
        
        for split_name, samples in splits.items():
            # Diretórios do split
            split_images_dir = split_dir / split_name / "images"
            split_norm_images_dir = split_normalized_dir / split_name / "images"
            
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_norm_images_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"   Copiando {len(samples)} amostras para {split_name}...")
            
            for sample in tqdm(samples, desc=f"  {split_name}"):
                filename = Path(sample["output_path"]).name
                
                # Copia imagem original processada
                shutil.copy2(sample["output_path"], split_images_dir / filename)
                
                # Copia imagem normalizada
                shutil.copy2(sample["normalized_path"], split_norm_images_dir / filename)
    
    def generate_manifests_and_reports(self, 
                                     splits: Dict[str, List[Dict]], 
                                     processed_sequences: Dict[str, List[Dict]]):
        """
        Gera manifests CSV e relatórios de estatísticas.
        """
        print("5. Gerando manifests e relatórios...")
        
        manifests_dir = self.output_dir / "manifests"
        manifests_dir.mkdir(exist_ok=True)
        
        # Manifest por split
        for split_name, samples in splits.items():
            manifest_path = manifests_dir / f"manifest_{split_name}.csv"
            
            with open(manifest_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "filename", "sequence", "timestamp", "split", 
                    "original_path", "processed_path", "normalized_path",
                    "original_width", "original_height", "processed_width", "processed_height"
                ])
                
                for sample in samples:
                    filename = Path(sample["output_path"]).name
                    orig_w, orig_h = sample["original_size"]
                    proc_w, proc_h = sample["processed_size"]
                    
                    writer.writerow([
                        filename, sample["sequence"], sample["timestamp"], split_name,
                        sample["original_path"], sample["output_path"], sample["normalized_path"],
                        orig_w, orig_h, proc_w, proc_h
                    ])
        
        # Manifest completo
        all_samples = []
        for split_name, samples in splits.items():
            for sample in samples:
                sample_copy = sample.copy()
                sample_copy["split"] = split_name
                all_samples.append(sample_copy)
        
        manifest_all_path = manifests_dir / "manifest_all.csv"
        with open(manifest_all_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "sequence", "timestamp", "split",
                "original_path", "processed_path", "normalized_path",
                "original_width", "original_height", "processed_width", "processed_height"
            ])
            
            for sample in all_samples:
                filename = Path(sample["output_path"]).name
                orig_w, orig_h = sample["original_size"]
                proc_w, proc_h = sample["processed_size"]
                
                writer.writerow([
                    filename, sample["sequence"], sample["timestamp"], sample["split"],
                    sample["original_path"], sample["output_path"], sample["normalized_path"],
                    orig_w, orig_h, proc_w, proc_h
                ])
        
        # Relatório de estatísticas
        report_path = self.output_dir / "preprocessing_report.txt"
        with open(report_path, 'w') as f:
            f.write("=== RELATÓRIO DE PRÉ-PROCESSAMENTO - FIELDSAFE RGB ===\n\n")
            f.write(f"Data de processamento: {pd.Timestamp.now()}\n")
            f.write(f"Diretório fonte: {self.rgb_root}\n")
            f.write(f"Diretório saída: {self.output_dir}\n\n")
            
            f.write("=== CONFIGURAÇÃO ===\n")
            f.write(f"Tamanho alvo: {self.target_size}\n")
            f.write(f"Range normalização: {self.normalize_range}\n")
            f.write(f"Splits: {self.splits}\n")
            f.write(f"Seed aleatória: {self.random_seed}\n\n")
            
            f.write("=== ESTATÍSTICAS DE PROCESSAMENTO ===\n")
            f.write(f"Sequências encontradas: {self.stats['sequences_found']}\n")
            f.write(f"Total de imagens: {self.stats['total_images']}\n")
            f.write(f"Imagens processadas: {self.stats['processed_images']}\n")
            f.write(f"Falhas: {self.stats['failed_images']}\n")
            f.write(f"Taxa de sucesso: {self.stats['processed_images']/self.stats['total_images']*100:.1f}%\n\n")
            
            f.write("=== DISTRIBUIÇÃO DOS SPLITS ===\n")
            total_splits = sum(self.stats["splits"].values())
            for split_name, count in self.stats["splits"].items():
                percentage = count / total_splits * 100
                f.write(f"{split_name.upper()}: {count} amostras ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("=== DISTRIBUIÇÃO POR SEQUÊNCIA ===\n")
            for sequence_name, samples in processed_sequences.items():
                f.write(f"{sequence_name}: {len(samples)} amostras\n")
        
        # JSON com metadados completos
        metadata_path = self.output_dir / "preprocessing_metadata.json"
        metadata = {
            "configuration": {
                "target_size": self.target_size,
                "normalize_range": self.normalize_range,
                "splits": self.splits,
                "random_seed": self.random_seed
            },
            "statistics": self.stats,
            "sequences": {name: len(samples) for name, samples in processed_sequences.items()}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Manifests salvos em: {manifests_dir}")
        print(f"   Relatório salvo em: {report_path}")
        print(f"   Metadados salvos em: {metadata_path}")
    
    def run_preprocessing_pipeline(self) -> bool:
        """
        Executa pipeline completo de pré-processamento.
        
        Returns:
            bool: True se bem-sucedido
        """
        try:
            # 1. Descoberta de sequências
            sequences = self.discover_sequences()
            if not sequences:
                print("❌ Nenhuma sequência encontrada!")
                return False
            
            # 2. Processamento de imagens
            processed_sequences = self.process_sequences(sequences)
            if not any(processed_sequences.values()):
                print("❌ Nenhuma imagem foi processada com sucesso!")
                return False
            
            # 3. Divisão estratificada
            splits = self.create_stratified_splits(processed_sequences)
            
            # 4. Organização em diretórios
            self.save_splits_to_directories(splits)
            
            # 5. Manifests e relatórios
            self.generate_manifests_and_reports(splits, processed_sequences)
            
            print(f"\n✅ Pipeline concluído com sucesso!")
            print(f"📁 Dados preparados em: {self.output_dir}")
            print(f"📊 Splits: Train={self.stats['splits']['train']}, "
                  f"Val={self.stats['splits']['val']}, Test={self.stats['splits']['test']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro no pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Função principal com argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline de pré-processamento FieldSAFE RGB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Processamento básico com configurações padrão
  python prepare_fieldsafe_preprocessing.py \\
    --rgb_root datasets/extract_rgb \\
    --output_dir datasets/fieldsafe_rgb_ready

  # Processamento com configurações personalizadas
  python prepare_fieldsafe_preprocessing.py \\
    --rgb_root datasets/extract_rgb \\
    --output_dir datasets/fieldsafe_rgb_ready \\
    --target_size 512 512 \\
    --splits 0.8 0.1 0.1 \\
    --seed 123
        """
    )
    
    parser.add_argument("--rgb_root", required=True, 
                       help="Diretório raiz com sequências RGB extraídas")
    parser.add_argument("--output_dir", required=True,
                       help="Diretório de saída para dados processados")
    parser.add_argument("--target_size", nargs=2, type=int, default=[640, 640],
                       help="Tamanho alvo para redimensionamento (padrão: 640 640)")
    parser.add_argument("--normalize_range", nargs=2, type=float, default=[0.0, 1.0],
                       help="Range de normalização (padrão: 0.0 1.0)")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                       help="Proporções train/val/test (padrão: 0.7 0.15 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed para reprodutibilidade (padrão: 42)")
    
    args = parser.parse_args()
    
    # Valida argumentos
    if sum(args.splits) != 1.0:
        print(f"❌ Erro: splits devem somar 1.0, mas somam {sum(args.splits)}")
        return 1
    
    # Cria e executa preprocessor
    preprocessor = FieldSAFEPreprocessor(
        rgb_root=args.rgb_root,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        normalize_range=tuple(args.normalize_range),
        splits=tuple(args.splits),
        random_seed=args.seed
    )
    
    success = preprocessor.run_preprocessing_pipeline()
    return 0 if success else 1

if __name__ == "__main__":
    import pandas as pd  # Para timestamp no relatório
    exit(main())