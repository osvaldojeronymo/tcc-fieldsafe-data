#!/usr/bin/env python3
"""
Implementação da Camada de Fusão Supervisionada para o ensemble FieldSAFE.

Combina as saídas probabilísticas dos três modelos (YOLOv5, SegNet, Autoencoder)
através de uma rede neural supervisionada que aprende a ponderar optimamente
as contribuições de cada modelo para produzir a decisão final sobre obstáculos.

Características:
- Fusão interpretável com pesos adaptativos
- Incorporação de confiança dos modelos individuais
- Saída probabilística final sobre presença de obstáculos
- Mecanismo de atenção para diferentes tipos de cena
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from .base_model import BaseVisionModel, ModelOutput, model_registry

@dataclass
class EnsembleInput:
    """Estrutura para entrada do ensemble."""
    yolo_output: ModelOutput
    segnet_output: ModelOutput  
    autoencoder_output: ModelOutput
    image: torch.Tensor  # Imagem original para contexto

@dataclass
class EnsembleOutput:
    """Estrutura para saída do ensemble."""
    obstacle_probability: torch.Tensor    # Probabilidade final de obstáculo
    confidence: torch.Tensor              # Confiança da decisão ensemble
    model_weights: Dict[str, torch.Tensor]  # Pesos aplicados a cada modelo
    individual_contributions: Dict[str, torch.Tensor]  # Contribuição de cada modelo
    attention_map: Optional[torch.Tensor] = None  # Mapa de atenção espacial
    metadata: Dict = None

class SupervisedFusionLayer(BaseVisionModel):
    """
    Camada de fusão supervisionada que combina outputs dos três modelos.
    
    Arquitetura:
    1. Feature Fusion: Combina features extraídas de cada modelo
    2. Attention Mechanism: Aprende importância relativa por região/tipo de cena
    3. Probabilistic Combination: Produz probabilidade final ponderada
    4. Confidence Estimation: Estima confiança da decisão ensemble
    """
    
    def __init__(self,
                 yolo_feature_dim: int = 512,
                 segnet_feature_dim: int = 512,
                 autoencoder_feature_dim: int = 512,
                 fusion_hidden_dim: int = 256,
                 num_attention_heads: int = 4,
                 dropout_rate: float = 0.2):
        
        super().__init__("supervised_fusion_ensemble", (640, 640))
        
        self.yolo_feature_dim = yolo_feature_dim
        self.segnet_feature_dim = segnet_feature_dim
        self.autoencoder_feature_dim = autoencoder_feature_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_attention_heads = num_attention_heads
        
        # Projeção das features individuais para espaço comum
        self.yolo_projection = nn.Linear(yolo_feature_dim, fusion_hidden_dim)
        self.segnet_projection = nn.Linear(segnet_feature_dim, fusion_hidden_dim)
        self.autoencoder_projection = nn.Linear(autoencoder_feature_dim, fusion_hidden_dim)
        
        # Projeção das probabilidades individuais
        self.yolo_prob_projection = nn.Linear(5, fusion_hidden_dim // 4)  # 5 classes YOLO
        self.segnet_prob_projection = nn.Linear(5, fusion_hidden_dim // 4)  # 5 classes SegNet
        self.autoencoder_prob_projection = nn.Linear(2, fusion_hidden_dim // 4)  # 2 classes Autoencoder
        
        # Mecanismo de atenção multi-head
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Rede de fusão principal
        total_feature_dim = fusion_hidden_dim * 3 + fusion_hidden_dim  # features + probs
        self.fusion_network = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_hidden_dim * 2),
            nn.BatchNorm1d(fusion_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2)
        )
        
        # Cabeças de saída
        self.obstacle_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probabilidade de obstáculo
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Confiança da decisão
        )
        
        # Rede para pesos adaptativos dos modelos
        self.weight_network = nn.Sequential(
            nn.Linear(fusion_hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)  # Pesos normalizados para os 3 modelos
        )
        
        # Contexto espacial (opcional)
        self.spatial_context = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 64, fusion_hidden_dim // 4)
        )
        
    def forward(self, ensemble_input: EnsembleInput) -> EnsembleOutput:
        """Forward pass do ensemble."""
        return self.fuse_models(ensemble_input)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai features contextuais da imagem."""
        return self.spatial_context(x)
    
    def get_probabilistic_output(self, x: torch.Tensor) -> ModelOutput:
        """Interface compatível - requer ensemble_input adequado."""
        raise NotImplementedError("Use fuse_models() com EnsembleInput apropriado")
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Pré-processamento básico."""
        if x.max() > 1.0:
            x = x / 255.0
        return x
    
    def postprocess(self, output: torch.Tensor) -> Dict:
        """Pós-processamento da saída ensemble."""
        return {"ensemble_output": output}
    
    def fuse_models(self, ensemble_input: EnsembleInput) -> EnsembleOutput:
        """
        Fusão principal dos modelos.
        
        Args:
            ensemble_input: Saídas dos três modelos + imagem original
            
        Returns:
            EnsembleOutput com decisão final e metadados
        """
        batch_size = ensemble_input.image.size(0)
        
        # 1. Projeta features individuais para espaço comum
        yolo_features = self.yolo_projection(ensemble_input.yolo_output.features)
        segnet_features = self.segnet_projection(ensemble_input.segnet_output.features)  
        autoencoder_features = self.autoencoder_projection(ensemble_input.autoencoder_output.features)
        
        # 2. Projeta probabilidades individuais
        yolo_probs = self.yolo_prob_projection(ensemble_input.yolo_output.probabilities)
        segnet_probs = self.segnet_prob_projection(ensemble_input.segnet_output.probabilities)
        autoencoder_probs = self.autoencoder_prob_projection(ensemble_input.autoencoder_output.probabilities)
        
        # 3. Extrai contexto espacial da imagem
        spatial_context = self.extract_features(ensemble_input.image)
        
        # 4. Mecanismo de atenção entre modelos
        model_features = torch.stack([yolo_features, segnet_features, autoencoder_features], dim=1)
        attended_features, attention_weights = self.attention(
            model_features, model_features, model_features
        )
        attended_features = attended_features.mean(dim=1)  # Agrega dimensão sequence
        
        # 5. Concatena todas as informações
        prob_features = torch.cat([yolo_probs, segnet_probs, autoencoder_probs], dim=1)
        all_features = torch.cat([
            yolo_features, segnet_features, autoencoder_features,
            prob_features
        ], dim=1)
        
        # 6. Rede de fusão principal
        fused_features = self.fusion_network(all_features)
        
        # 7. Saídas finais
        obstacle_probability = self.obstacle_head(fused_features)
        confidence = self.confidence_head(fused_features)
        model_weights = self.weight_network(fused_features)
        
        # 8. Calcula contribuições individuais ponderadas
        individual_contributions = {
            "yolo": model_weights[:, 0:1] * ensemble_input.yolo_output.probabilities.max(dim=1, keepdim=True)[0],
            "segnet": model_weights[:, 1:2] * ensemble_input.segnet_output.probabilities.max(dim=1, keepdim=True)[0],
            "autoencoder": model_weights[:, 2:3] * ensemble_input.autoencoder_output.probabilities[:, 1:2]  # Anomaly prob
        }
        
        # 9. Metadados do ensemble
        metadata = {
            "individual_confidences": {
                "yolo": ensemble_input.yolo_output.confidence,
                "segnet": ensemble_input.segnet_output.confidence,
                "autoencoder": ensemble_input.autoencoder_output.confidence
            },
            "attention_weights": attention_weights,
            "spatial_context": spatial_context
        }
        
        return EnsembleOutput(
            obstacle_probability=obstacle_probability,
            confidence=confidence,
            model_weights={
                "yolo": model_weights[:, 0],
                "segnet": model_weights[:, 1], 
                "autoencoder": model_weights[:, 2]
            },
            individual_contributions=individual_contributions,
            attention_map=attention_weights,
            metadata=metadata
        )
    
    def predict_obstacles(self, 
                         yolo_output: ModelOutput,
                         segnet_output: ModelOutput,
                         autoencoder_output: ModelOutput,
                         image: torch.Tensor,
                         threshold: float = 0.5) -> Dict:
        """
        Predição completa de obstáculos usando ensemble.
        
        Args:
            yolo_output: Saída do YOLOv5
            segnet_output: Saída do SegNet
            autoencoder_output: Saída do Autoencoder
            image: Imagem original
            threshold: Limiar para classificação binária
            
        Returns:
            Dict com decisão final e análise detalhada
        """
        ensemble_input = EnsembleInput(
            yolo_output=yolo_output,
            segnet_output=segnet_output,
            autoencoder_output=autoencoder_output,
            image=image
        )
        
        ensemble_output = self.fuse_models(ensemble_input)
        
        # Decisão binária
        has_obstacle = ensemble_output.obstacle_probability > threshold
        
        result = {
            "has_obstacle": has_obstacle.bool(),
            "obstacle_probability": ensemble_output.obstacle_probability.squeeze(),
            "confidence": ensemble_output.confidence.squeeze(),
            "model_weights": {k: v.squeeze() for k, v in ensemble_output.model_weights.items()},
            "individual_contributions": {k: v.squeeze() for k, v in ensemble_output.individual_contributions.items()},
            "decision_threshold": threshold,
            "model_agreement": self._calculate_model_agreement(ensemble_input),
            "uncertainty": self._calculate_uncertainty(ensemble_output)
        }
        
        return result
    
    def _calculate_model_agreement(self, ensemble_input: EnsembleInput) -> torch.Tensor:
        """Calcula grau de concordância entre modelos."""
        # Extrai decisões binárias de cada modelo
        yolo_decision = ensemble_input.yolo_output.probabilities.max(dim=1)[0] > 0.5
        segnet_decision = ensemble_input.segnet_output.probabilities.max(dim=1)[0] > 0.5
        autoencoder_decision = ensemble_input.autoencoder_output.probabilities[:, 1] > 0.5
        
        # Calcula concordância
        decisions = torch.stack([yolo_decision, segnet_decision, autoencoder_decision], dim=1)
        agreement = decisions.float().mean(dim=1)  # [0, 1] onde 1 = concordância total
        
        return agreement
    
    def _calculate_uncertainty(self, ensemble_output: EnsembleOutput) -> torch.Tensor:
        """Calcula incerteza da decisão ensemble."""
        # Incerteza baseada na entropia da probabilidade
        prob = ensemble_output.obstacle_probability.squeeze()
        uncertainty = -prob * torch.log(prob + 1e-8) - (1 - prob) * torch.log(1 - prob + 1e-8)
        
        return uncertainty
    
    def get_model_importance(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Analisa importância relativa de cada modelo no conjunto de dados.
        
        Args:
            dataloader: DataLoader com amostras para análise
            
        Returns:
            Dict com importância média de cada modelo
        """
        self.eval()
        model_weights_sum = {"yolo": 0.0, "segnet": 0.0, "autoencoder": 0.0}
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Assumindo que batch contém EnsembleInput
                ensemble_output = self.fuse_models(batch)
                
                for model_name in model_weights_sum.keys():
                    model_weights_sum[model_name] += ensemble_output.model_weights[model_name].sum().item()
                
                total_samples += batch.image.size(0)
        
        # Calcula importância média
        model_importance = {
            model: weight_sum / total_samples 
            for model, weight_sum in model_weights_sum.items()
        }
        
        return model_importance

# Factory functions
def create_supervised_fusion_ensemble(**kwargs) -> SupervisedFusionLayer:
    """Factory function para criar ensemble supervisionado."""
    model = SupervisedFusionLayer(**kwargs)
    model_registry.register(model)
    return model

class EnsembleTrainer:
    """
    Classe auxiliar para treinamento do ensemble supervisionado.
    """
    
    def __init__(self, 
                 ensemble_model: SupervisedFusionLayer,
                 yolo_model: BaseVisionModel,
                 segnet_model: BaseVisionModel,
                 autoencoder_model: BaseVisionModel,
                 device: torch.device = torch.device('cpu')):
        
        self.ensemble_model = ensemble_model
        self.yolo_model = yolo_model
        self.segnet_model = segnet_model
        self.autoencoder_model = autoencoder_model
        self.device = device
        
        # Move modelos para device
        self.ensemble_model.to(device)
        self.yolo_model.to(device)
        self.segnet_model.to(device)
        self.autoencoder_model.to(device)
    
    def prepare_ensemble_input(self, images: torch.Tensor) -> EnsembleInput:
        """Prepara entrada para o ensemble executando os três modelos."""
        self.yolo_model.eval()
        self.segnet_model.eval()
        self.autoencoder_model.eval()
        
        with torch.no_grad():
            yolo_output = self.yolo_model.get_probabilistic_output(images)
            segnet_output = self.segnet_model.get_probabilistic_output(images)
            autoencoder_output = self.autoencoder_model.get_probabilistic_output(images)
        
        return EnsembleInput(
            yolo_output=yolo_output,
            segnet_output=segnet_output,
            autoencoder_output=autoencoder_output,
            image=images
        )
    
    def train_step(self, 
                   images: torch.Tensor, 
                   labels: torch.Tensor, 
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module = None) -> Dict[str, float]:
        """
        Executa um passo de treinamento do ensemble.
        
        Args:
            images: Batch de imagens
            labels: Rótulos binários (0=sem obstáculo, 1=com obstáculo)
            optimizer: Otimizador
            criterion: Função de perda (padrão: BCELoss)
            
        Returns:
            Dict com métricas do passo
        """
        if criterion is None:
            criterion = nn.BCELoss()
        
        # Prepara entrada do ensemble
        ensemble_input = self.prepare_ensemble_input(images)
        
        # Forward pass
        self.ensemble_model.train()
        ensemble_output = self.ensemble_model.fuse_models(ensemble_input)
        
        # Calcula perda
        loss = criterion(ensemble_output.obstacle_probability.squeeze(), labels.float())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Métricas
        with torch.no_grad():
            predictions = (ensemble_output.obstacle_probability.squeeze() > 0.5).float()
            accuracy = (predictions == labels.float()).float().mean()
            confidence_mean = ensemble_output.confidence.mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "confidence": confidence_mean.item()
        }
