#!/usr/bin/env python3
"""
Implementação do SegNet para segmentação semântica no FieldSAFE.

Arquitetura encoder-decoder especializada em segmentar:
- Áreas cultivadas vs não cultivadas
- Contexto espacial do ambiente agrícola
- Regiões navegáveis vs obstáculos

Produz mapas de probabilidade semântica para integração no ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import torchvision.transforms as transforms
from .base_model import BaseVisionModel, ModelOutput, model_registry

class SegNetFieldSAFE(BaseVisionModel):
    """
    SegNet adaptado para segmentação semântica do FieldSAFE.
    
    Classes semânticas:
    0: background/sky
    1: area_cultivada (cultivated area)
    2: area_nao_cultivada (non-cultivated area)  
    3: obstaculo (obstacle)
    4: caminho_navegavel (navigable path)
    """
    
    def __init__(self, 
                 num_classes: int = 5,
                 input_size: Tuple[int, int] = (640, 640),
                 encoder_depth: int = 5):
        
        super().__init__("segnet_fieldsafe", input_size)
        
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        
        # Classes semânticas
        self.class_names = [
            "background", "area_cultivada", "area_nao_cultivada", 
            "obstaculo", "caminho_navegavel"
        ]
        
        # Encoder (VGG-like backbone)
        self.encoder = self._build_encoder()
        
        # Decoder (espelhado do encoder)
        self.decoder = self._build_decoder()
        
        # Classificador final
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Feature extractor para ensemble
        self.feature_dim = 512
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(8 * 8 * 512, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Transformações de entrada
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Armazena índices de pooling para unpooling
        self.pool_indices = {}
        
    def _build_encoder(self):
        """Constrói encoder baseado em VGG."""
        # VGG16-like encoder
        layers = []
        in_channels = 3
        
        # Configuração das camadas: [out_channels, num_blocks]
        encoder_config = [
            [64, 2],   # Block 1
            [128, 2],  # Block 2  
            [256, 3],  # Block 3
            [512, 3],  # Block 4
            [512, 3]   # Block 5
        ]
        
        for i, (out_channels, num_blocks) in enumerate(encoder_config):
            # Bloco de convoluções
            for j in range(num_blocks):
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            
            # MaxPooling com return_indices para SegNet
            layers.append(nn.MaxPool2d(2, stride=2, return_indices=True))
        
        return nn.ModuleList(layers)
    
    def _build_decoder(self):
        """Constrói decoder espelhado do encoder."""
        layers = []
        
        # Configuração das camadas (reversa do encoder)
        decoder_config = [
            [512, 3],  # Block 5
            [512, 3],  # Block 4
            [256, 3],  # Block 3
            [128, 2],  # Block 2
            [64, 2]    # Block 1
        ]
        
        in_channels = 512
        
        for i, (out_channels, num_blocks) in enumerate(decoder_config):
            # MaxUnpooling
            layers.append(nn.MaxUnpool2d(2, stride=2))
            
            # Bloco de convoluções
            for j in range(num_blocks):
                if j == num_blocks - 1 and i < len(decoder_config) - 1:
                    # Última conv do bloco reduz canais
                    next_channels = decoder_config[i + 1][0]
                    layers.append(nn.Conv2d(in_channels, next_channels, 3, padding=1))
                    layers.append(nn.BatchNorm2d(next_channels))
                    in_channels = next_channels
                else:
                    layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
                    layers.append(nn.BatchNorm2d(out_channels))
                
                layers.append(nn.ReLU(inplace=True))
        
        return nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass principal."""
        # Encoder
        pool_indices = []
        encoder_outputs = []
        
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                pool_indices.append(indices)
                encoder_outputs.append(x)
            else:
                x = layer(x)
        
        # Decoder
        pool_idx = len(pool_indices) - 1
        decoder_layer_idx = 0
        
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, pool_indices[pool_idx])
                pool_idx -= 1
            else:
                x = layer(x)
        
        # Classificação final
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai features intermediárias para o ensemble."""
        # Extrai features do meio do encoder
        features = None
        layer_count = 0
        
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                x, _ = layer(x)
                layer_count += 1
                # Usa features do 3º bloco (meio da rede)
                if layer_count == 3:
                    features = x
                    break
            else:
                x = layer(x)
        
        if features is not None:
            return self.feature_extractor(features)
        else:
            # Fallback
            return torch.randn(x.size(0), self.feature_dim)
    
    def get_probabilistic_output(self, x: torch.Tensor) -> ModelOutput:
        """Produz saída probabilística para ensemble."""
        # Segmentação completa
        segmentation = self.forward(x)
        
        # Aplica softmax para probabilidades
        probabilities = F.softmax(segmentation, dim=1)
        
        # Calcula estatísticas agregadas por classe
        batch_size = x.size(0)
        class_probs = torch.zeros(batch_size, self.num_classes)
        confidence_scores = torch.zeros(batch_size, 1)
        
        for i in range(batch_size):
            # Probabilidade média por classe
            for cls in range(self.num_classes):
                class_probs[i, cls] = probabilities[i, cls].mean()
            
            # Confiança baseada na entropia
            entropy = -torch.sum(probabilities[i] * torch.log(probabilities[i] + 1e-8), dim=0)
            confidence_scores[i] = 1.0 - entropy.mean() / np.log(self.num_classes)
        
        # Extrai features para ensemble
        features = self.extract_features(x)
        
        # Metadados específicos do SegNet
        metadata = {
            "model_type": "semantic_segmentation",
            "classes": self.class_names,
            "segmentation_map": probabilities,
            "input_size": self.input_size
        }
        
        return ModelOutput(
            probabilities=class_probs,
            confidence=confidence_scores,
            features=features,
            metadata=metadata
        )
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Pré-processamento para SegNet."""
        # Normaliza para [0, 1] se necessário
        if x.max() > 1.0:
            x = x / 255.0
        
        # Aplica transformações
        return self.transform(x)
    
    def postprocess(self, output: torch.Tensor) -> Dict:
        """Pós-processamento da segmentação."""
        # Aplica softmax e obtém classes preditas
        probabilities = F.softmax(output, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        # Calcula estatísticas
        stats = {}
        for cls_id, cls_name in enumerate(self.class_names):
            mask = (predicted_classes == cls_id)
            stats[cls_name] = {
                "pixel_count": mask.sum().item(),
                "percentage": (mask.sum().float() / mask.numel()).item() * 100
            }
        
        return {
            "segmentation_map": predicted_classes,
            "probabilities": probabilities,
            "class_statistics": stats
        }
    
    def segment_scene(self, x: torch.Tensor, return_stats: bool = True) -> Dict:
        """
        Segmentação completa da cena.
        
        Args:
            x: Imagem de entrada
            return_stats: Se deve calcular estatísticas por classe
            
        Returns:
            Dict com mapa de segmentação e estatísticas
        """
        model_output = self.get_probabilistic_output(x)
        segmentation_map = model_output.metadata["segmentation_map"]
        
        result = {
            "class_probabilities": model_output.probabilities,
            "confidence": model_output.confidence,
            "segmentation_map": segmentation_map,
        }
        
        if return_stats:
            # Calcula área por classe
            for i, class_name in enumerate(self.class_names):
                class_mask = torch.argmax(segmentation_map, dim=1) == i
                area_percentage = class_mask.float().mean() * 100
                result[f"{class_name}_coverage"] = area_percentage.item()
        
        return result
    
    def get_navigable_areas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Identifica áreas navegáveis na cena.
        
        Returns:
            Máscara binária com áreas navegáveis
        """
        segmentation = self.forward(x)
        probabilities = F.softmax(segmentation, dim=1)
        
        # Classes consideradas navegáveis
        navigable_classes = [1, 4]  # area_cultivada, caminho_navegavel
        navigable_mask = torch.zeros_like(probabilities[:, 0])
        
        for cls_id in navigable_classes:
            navigable_mask += probabilities[:, cls_id]
        
        return navigable_mask > 0.5

# Registra o modelo
def create_segnet_fieldsafe(**kwargs) -> SegNetFieldSAFE:
    """Factory function para criar SegNet FieldSAFE."""
    model = SegNetFieldSAFE(**kwargs)
    model_registry.register(model)
    return model
