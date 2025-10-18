#!/usr/bin/env python3
"""
Interface base para modelos de visão computacional no ensemble FieldSAFE.

Define a estrutura padrão que todos os modelos (YOLOv5, SegNet, Autoencoder)
devem seguir para garantir interoperabilidade na camada de fusão.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Union, Optional
from dataclasses import dataclass

@dataclass
class ModelOutput:
    """Estrutura padronizada para saídas dos modelos."""
    probabilities: torch.Tensor  # Probabilidades principais [0-1]
    confidence: torch.Tensor     # Confiança do modelo [0-1]
    features: torch.Tensor       # Features extraídas para ensemble
    metadata: Dict               # Informações específicas do modelo

class BaseVisionModel(ABC, nn.Module):
    """
    Classe base abstrata para modelos de visão computacional.
    
    Todos os modelos devem implementar:
    - forward(): Inferência principal
    - extract_features(): Extração de features para ensemble
    - get_probabilistic_output(): Saída probabilística padronizada
    """
    
    def __init__(self, model_name: str, input_size: Tuple[int, int] = (640, 640)):
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.is_trained = False
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inferência principal do modelo.
        
        Args:
            x: Tensor de entrada [B, C, H, W]
            
        Returns:
            Saída bruta do modelo
        """
        pass
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai features intermediárias para a camada de ensemble.
        
        Args:
            x: Tensor de entrada [B, C, H, W]
            
        Returns:
            Features extraídas [B, feature_dim]
        """
        pass
    
    @abstractmethod
    def get_probabilistic_output(self, x: torch.Tensor) -> ModelOutput:
        """
        Produz saída probabilística padronizada para o ensemble.
        
        Args:
            x: Tensor de entrada [B, C, H, W]
            
        Returns:
            ModelOutput com probabilidades, confiança e features
        """
        pass
    
    @abstractmethod
    def preprocess(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Pré-processamento específico do modelo.
        
        Args:
            x: Imagem de entrada
            
        Returns:
            Tensor pré-processado
        """
        pass
    
    @abstractmethod
    def postprocess(self, output: torch.Tensor) -> Dict:
        """
        Pós-processamento específico do modelo.
        
        Args:
            output: Saída bruta do modelo
            
        Returns:
            Resultado interpretado
        """
        pass
    
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> ModelOutput:
        """
        Pipeline completo de predição.
        
        Args:
            x: Imagem de entrada
            
        Returns:
            Saída probabilística padronizada
        """
        # Pré-processamento
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if len(x.shape) == 3:  # [H, W, C] -> [1, C, H, W]
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif len(x.shape) == 4 and x.shape[1] != 3:  # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        
        x = self.preprocess(x)
        
        # Inferência
        self.eval()
        with torch.no_grad():
            output = self.get_probabilistic_output(x)
        
        return output
    
    def get_model_info(self) -> Dict:
        """Retorna informações sobre o modelo."""
        return {
            "name": self.model_name,
            "input_size": self.input_size,
            "is_trained": self.is_trained,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

class ModelRegistry:
    """
    Registro central dos modelos disponíveis no ensemble.
    """
    
    def __init__(self):
        self.models = {}
        
    def register(self, model: BaseVisionModel):
        """Registra um modelo no ensemble."""
        self.models[model.model_name] = model
        
    def get_model(self, name: str) -> BaseVisionModel:
        """Obtém modelo por nome."""
        if name not in self.models:
            raise ValueError(f"Modelo '{name}' não encontrado. Disponíveis: {list(self.models.keys())}")
        return self.models[name]
    
    def get_all_models(self) -> Dict[str, BaseVisionModel]:
        """Retorna todos os modelos registrados."""
        return self.models.copy()
    
    def list_models(self) -> list:
        """Lista nomes dos modelos registrados."""
        return list(self.models.keys())

# Instância global do registro
model_registry = ModelRegistry()