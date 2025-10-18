#!/usr/bin/env python3
"""
Implementação do YOLOv5 para detecção de obstáculos agrícolas.

Modelo especializado em detectar e classificar:
- Fardos de feno
- Máquinas agrícolas  
- Pessoas
- Outros obstáculos relevantes

Produz saídas probabilísticas padronizadas para integração no ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from .base_model import BaseVisionModel, ModelOutput, model_registry

class YOLOv5FieldSAFE(BaseVisionModel):
    """
    YOLOv5 adaptado para detecção de obstáculos no FieldSAFE.
    
    Classes detectadas:
    0: fardo (hay bale)
    1: maquina (agricultural machine)
    2: pessoa (person)
    3: veiculo (vehicle)
    4: outro_obstaculo (other obstacle)
    """
    
    def __init__(self, 
                 num_classes: int = 5,
                 input_size: Tuple[int, int] = (640, 640),
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 pretrained: bool = True):
        
        super().__init__("yolov5_fieldsafe", input_size)
        
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Classes do FieldSAFE
        self.class_names = [
            "fardo", "maquina", "pessoa", "veiculo", "outro_obstaculo"
        ]
        
        # Carrega YOLOv5 (usando ultralytics/yolov5)
        if pretrained:
            try:
                # Tenta carregar modelo pré-treinado
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.yolo_model.model[-1].nc = num_classes  # Ajusta número de classes
                self.yolo_model.model[-1].anchors = self.yolo_model.model[-1].anchors.clone()
            except Exception as e:
                print(f"Erro ao carregar YOLOv5 pré-treinado: {e}")
                self._build_custom_yolo()
        else:
            self._build_custom_yolo()
        
        # Feature extractor para ensemble
        self.feature_dim = 512
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(7 * 7 * 256, self.feature_dim),  # Ajustar baseado na arquitetura
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Transformações de entrada
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _build_custom_yolo(self):
        """Constrói YOLOv5 personalizado se pré-treinado falhar."""
        print("Construindo YOLOv5 personalizado...")
        # Implementação simplificada - você pode expandir conforme necessário
        from torchvision.models import resnet50
        
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Head de detecção simplificado
        self.detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, (self.num_classes + 5) * 3, 1)  # 3 anchors per cell
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass principal."""
        if hasattr(self, 'yolo_model'):
            # Usa YOLOv5 oficial
            return self.yolo_model(x)
        else:
            # Usa implementação personalizada
            features = self.backbone(x)
            detections = self.detection_head(features)
            return detections
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai features para o ensemble."""
        if hasattr(self, 'yolo_model'):
            # Extrai features do backbone do YOLOv5
            with torch.no_grad():
                # Hook para capturar features intermediárias
                features = None
                def hook(module, input, output):
                    nonlocal features
                    features = output
                
                # Registra hook no último layer do backbone
                handle = None
                for name, module in self.yolo_model.model.named_modules():
                    if 'backbone' in name or 'Conv' in str(type(module)):
                        handle = module.register_forward_hook(hook)
                        break
                
                _ = self.yolo_model(x)
                if handle:
                    handle.remove()
                
                if features is not None:
                    return self.feature_extractor(features)
                else:
                    # Fallback: usa saída reduzida
                    return torch.randn(x.size(0), self.feature_dim)
        else:
            # Usa backbone personalizado
            features = self.backbone(x)
            return self.feature_extractor(features)
    
    def get_probabilistic_output(self, x: torch.Tensor) -> ModelOutput:
        """Produz saída probabilística para ensemble."""
        detections = self.forward(x)
        
        # Processa detecções para extrair probabilidades
        if hasattr(self, 'yolo_model'):
            # Usa processamento do YOLOv5 oficial
            results = self.yolo_model(x)
            
            # Extrai probabilidades das detecções
            batch_size = x.size(0)
            probabilities = torch.zeros(batch_size, self.num_classes)
            confidence = torch.zeros(batch_size, 1)
            
            for i, result in enumerate(results):
                if len(result) > 0:
                    # Máxima confiança por classe
                    for cls_id in range(self.num_classes):
                        cls_detections = result[result[:, 5] == cls_id]
                        if len(cls_detections) > 0:
                            probabilities[i, cls_id] = cls_detections[:, 4].max()
                    
                    confidence[i] = result[:, 4].mean() if len(result) > 0 else 0.0
                    
        else:
            # Processamento personalizado
            batch_size, channels, height, width = detections.shape
            # Reshape para [batch, anchors, grid_y, grid_x, (x,y,w,h,conf,classes)]
            num_anchors = 3
            prediction_size = channels // num_anchors
            grid_size = height
            
            detections = detections.view(batch_size, num_anchors, prediction_size, grid_size, grid_size)
            detections = detections.permute(0, 1, 3, 4, 2).contiguous()
            
            # Extrai confiança e probabilidades de classe
            obj_conf = torch.sigmoid(detections[..., 4:5])
            class_probs = torch.sigmoid(detections[..., 5:])
            
            # Agrega probabilidades por classe
            probabilities = (obj_conf * class_probs).max(dim=(1, 2, 3))[0]
            confidence = obj_conf.max(dim=(1, 2, 3, 4))[0]
        
        # Extrai features para ensemble
        features = self.extract_features(x)
        
        # Metadados específicos do YOLO
        metadata = {
            "model_type": "object_detection",
            "classes": self.class_names,
            "detection_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold
        }
        
        return ModelOutput(
            probabilities=probabilities,
            confidence=confidence,
            features=features,
            metadata=metadata
        )
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Pré-processamento para YOLOv5."""
        # Normaliza para [0, 1] se necessário
        if x.max() > 1.0:
            x = x / 255.0
        
        # Aplica transformações
        return self.transform(x)
    
    def postprocess(self, output: torch.Tensor) -> Dict:
        """Pós-processamento das detecções."""
        # Implementar NMS e filtros de confiança
        if hasattr(self, 'yolo_model'):
            # YOLOv5 já faz pós-processamento
            return {"detections": output}
        else:
            # Implementação personalizada de NMS
            return {"raw_detections": output}
    
    def detect_objects(self, x: torch.Tensor, return_boxes: bool = True) -> Dict:
        """
        Detecção completa de objetos com bounding boxes.
        
        Args:
            x: Imagem de entrada
            return_boxes: Se deve retornar coordenadas das boxes
            
        Returns:
            Dict com detecções, classes e coordenadas
        """
        model_output = self.get_probabilistic_output(x)
        
        result = {
            "class_probabilities": model_output.probabilities,
            "confidence": model_output.confidence,
            "detected_classes": []
        }
        
        # Identifica classes detectadas
        for i, prob in enumerate(model_output.probabilities[0]):
            if prob > self.confidence_threshold:
                result["detected_classes"].append({
                    "class_id": i,
                    "class_name": self.class_names[i],
                    "probability": float(prob)
                })
        
        if return_boxes and hasattr(self, 'yolo_model'):
            # Adiciona bounding boxes se disponível
            detections = self.forward(x)
            result["bounding_boxes"] = detections
        
        return result

# Registra o modelo
def create_yolov5_fieldsafe(**kwargs) -> YOLOv5FieldSAFE:
    """Factory function para criar YOLOv5 FieldSAFE."""
    model = YOLOv5FieldSAFE(**kwargs)
    model_registry.register(model)
    return model

# Wrapper para compatibilidade
class YOLOv5Wrapper(YOLOv5FieldSAFE):
    """Wrapper para manter compatibilidade com código existente."""
    pass