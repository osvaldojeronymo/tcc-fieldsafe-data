#!/usr/bin/env python3
"""
Implementação do Autoencoder Convolucional para detecção de anomalias no FieldSAFE.

Modelo não supervisionado destinado à detecção de padrões visuais divergentes 
daqueles observados durante o treinamento. Identifica:
- Objetos não categorizados
- Situações atípicas no ambiente
- Anomalias visuais que fogem do padrão

Produz scores de anomalia probabilísticos para integração no ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import torchvision.transforms as transforms
from .base_model import BaseVisionModel, ModelOutput, model_registry

class ConvolutionalAutoencoder(BaseVisionModel):
    """
    Autoencoder Convolucional para detecção de anomalias no FieldSAFE.
    
    Arquitetura simétrica encoder-decoder que aprende a reconstruir 
    imagens normais e detecta anomalias através do erro de reconstrução.
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (640, 640),
                 latent_dim: int = 512,
                 num_filters: Tuple[int, ...] = (32, 64, 128, 256),
                 anomaly_threshold: float = 0.1):
        
        super().__init__("autoencoder_fieldsafe", input_size)
        
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.anomaly_threshold = anomaly_threshold
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Bottleneck (representação latente)
        self.bottleneck_size = self._calculate_bottleneck_size()
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bottleneck_size, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.bottleneck_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = self._build_decoder()
        
        # Feature extractor para ensemble
        self.feature_dim = latent_dim
        
        # Transformações de entrada
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Buffer para estatísticas de treinamento
        self.register_buffer('training_mean_loss', torch.tensor(0.0))
        self.register_buffer('training_std_loss', torch.tensor(1.0))
        
    def _build_encoder(self):
        """Constrói encoder convolucional."""
        layers = []
        in_channels = 3
        
        for out_channels in self.num_filters:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2)
            ])
            in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """Constrói decoder convolucional (espelhado do encoder)."""
        layers = []
        
        # Reshape bottleneck de volta para feature maps
        bottleneck_h = bottleneck_w = self.input_size[0] // (2 ** len(self.num_filters))
        self.bottleneck_spatial_size = (self.num_filters[-1], bottleneck_h, bottleneck_w)
        
        # Decoder layers (ordem reversa)
        reversed_filters = list(reversed(self.num_filters))
        
        for i, in_channels in enumerate(reversed_filters):
            out_channels = reversed_filters[i + 1] if i < len(reversed_filters) - 1 else 3
            
            layers.extend([
                nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),  # Upsampling
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels) if out_channels != 3 else nn.Identity(),
                nn.ReLU(inplace=True) if out_channels != 3 else nn.Sigmoid()
            ])
            
        return nn.Sequential(*layers)
    
    def _calculate_bottleneck_size(self):
        """Calcula tamanho do bottleneck baseado na arquitetura."""
        # Simulação forward para determinar tamanho
        dummy_input = torch.randn(1, 3, *self.input_size)
        with torch.no_grad():
            x = self.encoder(dummy_input)
            return x.numel() // x.size(0)  # Total elements per sample
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Codifica entrada para representação latente."""
        features = self.encoder(x)
        latent = self.bottleneck(features)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decodifica representação latente para reconstrução."""
        # Reshape de volta para feature maps
        batch_size = latent.size(0)
        features = latent.view(batch_size, *self.bottleneck_spatial_size)
        reconstruction = self.decoder(features)
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass completo (encode + decode)."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai representação latente como features para ensemble."""
        return self.encode(x)
    
    def compute_reconstruction_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Calcula erro de reconstrução."""
        reconstruction = self.forward(x)
        
        # MSE por pixel
        mse_loss = F.mse_loss(reconstruction, x, reduction='none')
        
        # Agrega por imagem
        if reduction == 'none':
            return mse_loss.view(x.size(0), -1).mean(dim=1)
        elif reduction == 'mean':
            return mse_loss.mean()
        elif reduction == 'sum':
            return mse_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula score de anomalia normalizado.
        
        Returns:
            Tensor com scores [0, 1] onde 1 = alta anomalia
        """
        reconstruction_loss = self.compute_reconstruction_loss(x, reduction='none')
        
        # Normaliza usando estatísticas de treinamento
        normalized_loss = (reconstruction_loss - self.training_mean_loss) / (self.training_std_loss + 1e-8)
        
        # Converte para probabilidade usando sigmoid
        anomaly_scores = torch.sigmoid(normalized_loss)
        
        return anomaly_scores
    
    def get_probabilistic_output(self, x: torch.Tensor) -> ModelOutput:
        """Produz saída probabilística para ensemble."""
        # Score de anomalia
        anomaly_scores = self.compute_anomaly_score(x)
        
        # Converte para formato probabilístico
        # [normal_prob, anomaly_prob]
        batch_size = x.size(0)
        probabilities = torch.zeros(batch_size, 2)
        probabilities[:, 0] = 1.0 - anomaly_scores  # Probabilidade normal
        probabilities[:, 1] = anomaly_scores        # Probabilidade anomalia
        
        # Confiança baseada na magnitude do score
        confidence = torch.abs(anomaly_scores - 0.5) * 2.0  # [0, 1]
        confidence = confidence.unsqueeze(1)
        
        # Features latentes
        features = self.extract_features(x)
        
        # Metadados específicos do autoencoder
        metadata = {
            "model_type": "anomaly_detection",
            "anomaly_scores": anomaly_scores,
            "reconstruction_loss": self.compute_reconstruction_loss(x, reduction='none'),
            "threshold": self.anomaly_threshold
        }
        
        return ModelOutput(
            probabilities=probabilities,
            confidence=confidence,
            features=features,
            metadata=metadata
        )
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Pré-processamento para autoencoder."""
        # Normaliza para [0, 1] se necessário
        if x.max() > 1.0:
            x = x / 255.0
        
        # Aplica transformações
        return self.transform(x)
    
    def postprocess(self, output: torch.Tensor) -> Dict:
        """Pós-processamento da reconstrução."""
        # Desnormaliza saída
        output_denorm = output * 255.0
        output_denorm = torch.clamp(output_denorm, 0, 255)
        
        return {
            "reconstruction": output_denorm,
            "reconstruction_normalized": output
        }
    
    def detect_anomalies(self, x: torch.Tensor, return_reconstruction: bool = False) -> Dict:
        """
        Detecção completa de anomalias.
        
        Args:
            x: Imagem de entrada
            return_reconstruction: Se deve retornar imagem reconstruída
            
        Returns:
            Dict com scores, classificação e opcionalmente reconstrução
        """
        model_output = self.get_probabilistic_output(x)
        anomaly_scores = model_output.metadata["anomaly_scores"]
        
        # Classifica como normal/anômalo
        is_anomaly = anomaly_scores > self.anomaly_threshold
        
        result = {
            "anomaly_scores": anomaly_scores,
            "is_anomaly": is_anomaly,
            "confidence": model_output.confidence.squeeze(),
            "normal_probability": model_output.probabilities[:, 0],
            "anomaly_probability": model_output.probabilities[:, 1]
        }
        
        if return_reconstruction:
            reconstruction = self.forward(x)
            result["reconstruction"] = self.postprocess(reconstruction)["reconstruction_normalized"]
        
        return result
    
    def update_training_stats(self, training_losses: torch.Tensor):
        """
        Atualiza estatísticas de treinamento para normalização.
        
        Args:
            training_losses: Tensor com losses de reconstrução do conjunto de treino
        """
        self.training_mean_loss.data = training_losses.mean()
        self.training_std_loss.data = training_losses.std()
        
        print(f"Estatísticas atualizadas - Média: {self.training_mean_loss:.4f}, "
              f"Desvio: {self.training_std_loss:.4f}")

# Registra o modelo
def create_autoencoder_fieldsafe(**kwargs) -> ConvolutionalAutoencoder:
    """Factory function para criar Autoencoder FieldSAFE."""
    model = ConvolutionalAutoencoder(**kwargs)
    model_registry.register(model)
    return model

# Variante com arquitetura mais profunda
class DeepAutoencoder(ConvolutionalAutoencoder):
    """Versão mais profunda do autoencoder para datasets complexos."""
    
    def __init__(self, **kwargs):
        # Configuração com mais filtros
        kwargs.setdefault('num_filters', (64, 128, 256, 512, 1024))
        kwargs.setdefault('latent_dim', 1024)
        super().__init__(**kwargs)
        self.model_name = "deep_autoencoder_fieldsafe"
