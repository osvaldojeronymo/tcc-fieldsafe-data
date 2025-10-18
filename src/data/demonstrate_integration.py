#!/usr/bin/env python3
"""
Exemplo de carregamento e uso dos dados pré-processados FieldSAFE.

Demonstra como:
1. Carregar dados corretamente formatados
2. Integrar com os modelos ensemble
3. Aplicar transformações adequadas
4. Verificar compatibilidade
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional, Dict
import torchvision.transforms as transforms

class FieldSAFEDataset(Dataset):
    """
    Dataset customizado para dados FieldSAFE pré-processados.
    
    Compatível com os modelos ensemble implementados.
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 use_normalized: bool = True,
                 transform: Optional[transforms.Compose] = None):
        """
        Inicializa dataset FieldSAFE.
        
        Args:
            data_dir: Diretório dos dados pré-processados
            split: 'train', 'val' ou 'test'
            use_normalized: Se deve usar imagens já normalizadas
            transform: Transformações adicionais
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_normalized = use_normalized
        self.transform = transform
        
        # Diretório das imagens
        if use_normalized:
            self.images_dir = self.data_dir / "split_normalized" / split / "images"
        else:
            self.images_dir = self.data_dir / "split" / split / "images"
        
        # Carrega manifest
        manifest_path = self.data_dir / "manifests" / f"manifest_{split}.csv"
        self.manifest = pd.read_csv(manifest_path)
        
        # Lista de imagens
        self.image_files = list(self.images_dir.glob("*.png"))
        self.image_files.sort()
        
        print(f"FieldSAFEDataset - Split: {split}, Imagens: {len(self.image_files)}")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Carrega uma amostra do dataset.
        
        Returns:
            Dict com tensors formatados para os modelos
        """
        # Carrega imagem
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Converte para tensor
        if self.use_normalized:
            # Imagem já está normalizada [0-255] -> converte para [0-1]
            image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        else:
            # Imagem em [0-255] -> normaliza para [0-1]
            image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        
        # Reorganiza dimensões: [H, W, C] -> [C, H, W]
        image_tensor = image_tensor.permute(2, 0, 1)
        
        # Aplica transformações se especificadas
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Extrai metadados do nome do arquivo
        filename = image_path.name
        sequence, timestamp = filename.replace('.png', '').split('_', 1)
        
        return {
            "image": image_tensor,
            "filename": filename,
            "sequence": sequence,
            "timestamp": timestamp,
            "path": str(image_path)
        }

def create_model_transforms() -> Dict[str, transforms.Compose]:
    """
    Cria transformações específicas para cada modelo do ensemble.
    
    Returns:
        Dict com transformações para cada modelo
    """
    
    # Transformações para YOLOv5 (ImageNet normalization)
    yolo_transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Transformações para SegNet (mesma que YOLOv5)
    segnet_transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Transformações para Autoencoder (mantém [0,1])
    autoencoder_transform = transforms.Compose([
        # Sem normalização adicional - usa [0,1]
    ])
    
    return {
        "yolo": yolo_transform,
        "segnet": segnet_transform,
        "autoencoder": autoencoder_transform
    }

def create_dataloaders(data_dir: str, 
                      batch_size: int = 16, 
                      num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Cria DataLoaders para todos os splits.
    
    Args:
        data_dir: Diretório dos dados pré-processados
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregamento
        
    Returns:
        Dict com DataLoaders para train/val/test
    """
    
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        # Dataset sem transformações (aplicadas nos modelos)
        dataset = FieldSAFEDataset(
            data_dir=data_dir,
            split=split,
            use_normalized=True,
            transform=None
        )
        
        # DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train")
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders

def test_ensemble_integration(data_dir: str):
    """
    Testa integração com os modelos ensemble.
    
    Args:
        data_dir: Diretório dos dados pré-processados
    """
    print("=== TESTE DE INTEGRAÇÃO COM ENSEMBLE ===")
    
    # Cria dataset de teste
    test_dataset = FieldSAFEDataset(
        data_dir=data_dir,
        split="test",
        use_normalized=True
    )
    
    # Carrega uma amostra
    sample = test_dataset[0]
    print(f"✅ Amostra carregada: {sample['filename']}")
    print(f"   Formato: {sample['image'].shape}")
    print(f"   Range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"   Sequência: {sample['sequence']}")
    
    # Testa transformações para cada modelo
    transforms_dict = create_model_transforms()
    
    # Adiciona dimensão batch
    image_batch = sample['image'].unsqueeze(0)  # [1, 3, 640, 640]
    
    for model_name, transform in transforms_dict.items():
        if transform.transforms:  # Se tem transformações
            transformed = transform(image_batch)
        else:
            transformed = image_batch
            
        print(f"✅ {model_name.upper()}: {transformed.shape}, "
              f"range=[{transformed.min():.3f}, {transformed.max():.3f}]")
    
    # Testa DataLoader
    print("\n=== TESTE DE DATALOADER ===")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    batch = next(iter(test_loader))
    print(f"✅ Batch carregado: {batch['image'].shape}")
    print(f"   Arquivos: {batch['filename']}")
    print(f"   Sequências: {set(batch['sequence'])}")
    
    return True

def demonstrate_model_usage(data_dir: str):
    """
    Demonstra uso com modelo ensemble simulado.
    
    Args:
        data_dir: Diretório dos dados pré-processados
    """
    print("\n=== DEMONSTRAÇÃO DE USO COM MODELO ===")
    
    # Simula carregamento de modelos (substitua por imports reais)
    print("📦 Simulando carregamento dos modelos...")
    
    # Cria DataLoader
    dataloaders = create_dataloaders(data_dir, batch_size=8)
    
    # Testa com batch do conjunto de teste
    test_loader = dataloaders["test"]
    batch = next(iter(test_loader))
    
    images = batch["image"]  # [batch_size, 3, 640, 640]
    print(f"✅ Batch processado: {images.shape}")
    
    # Simula processamento pelos modelos individuais
    transforms_dict = create_model_transforms()
    
    # YOLOv5
    yolo_input = transforms_dict["yolo"](images)
    print(f"✅ YOLOv5 Input: {yolo_input.shape}, range=[{yolo_input.min():.3f}, {yolo_input.max():.3f}]")
    
    # SegNet
    segnet_input = transforms_dict["segnet"](images)
    print(f"✅ SegNet Input: {segnet_input.shape}, range=[{segnet_input.min():.3f}, {segnet_input.max():.3f}]")
    
    # Autoencoder
    autoencoder_input = images  # Usa [0,1] diretamente
    print(f"✅ Autoencoder Input: {autoencoder_input.shape}, range=[{autoencoder_input.min():.3f}, {autoencoder_input.max():.3f}]")
    
    # Simula outputs dos modelos (para demonstração)
    batch_size = images.size(0)
    
    # Simula saídas probabilísticas
    yolo_probs = torch.rand(batch_size, 5)  # 5 classes
    segnet_probs = torch.rand(batch_size, 5)  # 5 classes
    autoencoder_probs = torch.rand(batch_size, 2)  # normal/anomaly
    
    print(f"\n✅ Simulação de saídas:")
    print(f"   YOLOv5 probabilities: {yolo_probs.shape}")
    print(f"   SegNet probabilities: {segnet_probs.shape}")
    print(f"   Autoencoder probabilities: {autoencoder_probs.shape}")
    
    # Ensemble final (simulado)
    ensemble_prob = (yolo_probs.max(dim=1)[0] + 
                    segnet_probs.max(dim=1)[0] + 
                    autoencoder_probs[:, 1]) / 3.0
    
    print(f"   Ensemble obstacle probability: {ensemble_prob.shape}")
    print(f"   Exemplo: {ensemble_prob[:3].tolist()}")
    
    return True

def main():
    """Função principal de demonstração."""
    
    data_dir = "datasets/fieldsafe_rgb_preprocessed"
    
    if not Path(data_dir).exists():
        print(f"❌ Diretório não encontrado: {data_dir}")
        print("Execute primeiro o pré-processamento!")
        return 1
    
    print("=== DEMONSTRAÇÃO DE USO DOS DADOS PRÉ-PROCESSADOS ===")
    
    try:
        # Testa integração básica
        test_ensemble_integration(data_dir)
        
        # Demonstra uso com modelos
        demonstrate_model_usage(data_dir)
        
        print("\n✅ INTEGRAÇÃO TESTADA COM SUCESSO!")
        print("\n📋 PRÓXIMOS PASSOS:")
        print("   1. Treine os modelos individuais (YOLOv5, SegNet, Autoencoder)")
        print("   2. Colete saídas probabilísticas de cada modelo")
        print("   3. Treine a camada de fusão supervisionada")
        print("   4. Avalie o ensemble no conjunto de teste")
        
        return 0
        
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())