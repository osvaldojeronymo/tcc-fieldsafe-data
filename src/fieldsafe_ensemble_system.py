#!/usr/bin/env python3
"""
Exemplo de uso do sistema ensemble FieldSAFE RGB.

Demonstra como integrar os três modelos (YOLOv5, SegNet, Autoencoder)
com a camada de fusão supervisionada para detecção de obstáculos agrícolas.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2

# Importa os modelos implementados
from src.models.base_model import model_registry
from src.models.yolo_v5 import create_yolov5_fieldsafe
from src.models.segnet import create_segnet_fieldsafe
from src.models.autoencoder import create_autoencoder_fieldsafe
from src.models.ensemble import create_supervised_fusion_ensemble, EnsembleTrainer, EnsembleInput

class FieldSAFEEnsembleSystem:
    """
    Sistema completo de ensemble para detecção de obstáculos no FieldSAFE.
    
    Integra os três modelos com a camada de fusão supervisionada,
    fornecendo interface unificada para treinamento e inferência.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Inicializa o sistema ensemble.
        
        Args:
            device: Device para computação (CPU/GPU)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicializa modelos individuais
        self.yolo_model = create_yolov5_fieldsafe(
            num_classes=5,
            confidence_threshold=0.5,
            pretrained=True
        )
        
        self.segnet_model = create_segnet_fieldsafe(
            num_classes=5,
            input_size=(640, 640)
        )
        
        self.autoencoder_model = create_autoencoder_fieldsafe(
            input_size=(640, 640),
            latent_dim=512,
            anomaly_threshold=0.1
        )
        
        # Inicializa ensemble
        self.ensemble_model = create_supervised_fusion_ensemble(
            yolo_feature_dim=512,
            segnet_feature_dim=512,
            autoencoder_feature_dim=512,
            fusion_hidden_dim=256
        )
        
        # Move modelos para device
        self._move_to_device()
        
        # Trainer para ensemble
        self.trainer = EnsembleTrainer(
            ensemble_model=self.ensemble_model,
            yolo_model=self.yolo_model,
            segnet_model=self.segnet_model,
            autoencoder_model=self.autoencoder_model,
            device=self.device
        )
        
        print(f"Sistema FieldSAFE Ensemble inicializado em {self.device}")
        print(f"Modelos registrados: {model_registry.list_models()}")
    
    def _move_to_device(self):
        """Move todos os modelos para o device especificado."""
        self.yolo_model.to(self.device)
        self.segnet_model.to(self.device)
        self.autoencoder_model.to(self.device)
        self.ensemble_model.to(self.device)
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Carrega e pré-processa imagem.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Tensor da imagem pré-processada
        """
        # Carrega imagem
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensiona para 640x640
        image = cv2.resize(image, (640, 640))
        
        # Converte para tensor
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 640, 640]
        
        return image_tensor.to(self.device)
    
    def predict_single_image(self, image_path: str, threshold: float = 0.5) -> dict:
        """
        Predição completa para uma única imagem.
        
        Args:
            image_path: Caminho para a imagem
            threshold: Limiar para classificação de obstáculo
            
        Returns:
            Dict com resultados detalhados do ensemble
        """
        # Carrega imagem
        image = self.load_image(image_path)
        
        # Executa modelos individuais
        self.yolo_model.eval()
        self.segnet_model.eval()
        self.autoencoder_model.eval()
        self.ensemble_model.eval()
        
        with torch.no_grad():
            # Saídas dos modelos individuais
            yolo_output = self.yolo_model.get_probabilistic_output(image)
            segnet_output = self.segnet_model.get_probabilistic_output(image)
            autoencoder_output = self.autoencoder_model.get_probabilistic_output(image)
            
            # Decisão do ensemble
            ensemble_result = self.ensemble_model.predict_obstacles(
                yolo_output=yolo_output,
                segnet_output=segnet_output,
                autoencoder_output=autoencoder_output,
                image=image,
                threshold=threshold
            )
        
        # Compila resultados detalhados
        result = {
            "image_path": image_path,
            "ensemble_decision": {
                "has_obstacle": bool(ensemble_result["has_obstacle"].item()),
                "probability": float(ensemble_result["obstacle_probability"].item()),
                "confidence": float(ensemble_result["confidence"].item()),
                "uncertainty": float(ensemble_result["uncertainty"].item())
            },
            "model_contributions": {
                "yolo": float(ensemble_result["individual_contributions"]["yolo"].item()),
                "segnet": float(ensemble_result["individual_contributions"]["segnet"].item()),
                "autoencoder": float(ensemble_result["individual_contributions"]["autoencoder"].item())
            },
            "model_weights": {
                "yolo": float(ensemble_result["model_weights"]["yolo"].item()),
                "segnet": float(ensemble_result["model_weights"]["segnet"].item()),
                "autoencoder": float(ensemble_result["model_weights"]["autoencoder"].item())
            },
            "individual_results": {
                "yolo": {
                    "max_class_prob": float(yolo_output.probabilities.max().item()),
                    "confidence": float(yolo_output.confidence.item()),
                    "detected_classes": self._extract_yolo_classes(yolo_output)
                },
                "segnet": {
                    "max_class_prob": float(segnet_output.probabilities.max().item()),
                    "confidence": float(segnet_output.confidence.item()),
                    "class_coverage": self._extract_segnet_coverage(segnet_output)
                },
                "autoencoder": {
                    "anomaly_score": float(autoencoder_output.probabilities[0, 1].item()),
                    "confidence": float(autoencoder_output.confidence.item()),
                    "is_anomaly": bool(autoencoder_output.probabilities[0, 1] > 0.5)
                }
            },
            "model_agreement": float(ensemble_result["model_agreement"].item())
        }
        
        return result
    
    def _extract_yolo_classes(self, yolo_output) -> list:
        """Extrai classes detectadas pelo YOLO."""
        class_names = ["fardo", "maquina", "pessoa", "veiculo", "outro_obstaculo"]
        detected = []
        
        for i, prob in enumerate(yolo_output.probabilities[0]):
            if prob > 0.3:  # Threshold para considerar detectado
                detected.append({
                    "class": class_names[i],
                    "probability": float(prob.item())
                })
        
        return detected
    
    def _extract_segnet_coverage(self, segnet_output) -> dict:
        """Extrai cobertura de classes do SegNet."""
        class_names = ["background", "area_cultivada", "area_nao_cultivada", "obstaculo", "caminho_navegavel"]
        coverage = {}
        
        for i, prob in enumerate(segnet_output.probabilities[0]):
            coverage[class_names[i]] = float(prob.item())
        
        return coverage
    
    def predict_batch(self, image_paths: list, threshold: float = 0.5) -> list:
        """
        Predição para lote de imagens.
        
        Args:
            image_paths: Lista de caminhos para imagens
            threshold: Limiar para classificação
            
        Returns:
            Lista com resultados para cada imagem
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path, threshold)
                results.append(result)
            except Exception as e:
                print(f"Erro ao processar {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        return results
    
    def analyze_dataset_predictions(self, dataset_dir: str, sample_size: int = 100) -> dict:
        """
        Analisa predições em amostra do dataset.
        
        Args:
            dataset_dir: Diretório com imagens
            sample_size: Número máximo de imagens para analisar
            
        Returns:
            Estatísticas agregadas das predições
        """
        from pathlib import Path
        import random
        
        # Encontra imagens
        dataset_path = Path(dataset_dir)
        image_files = list(dataset_path.glob("**/*.png")) + list(dataset_path.glob("**/*.jpg"))
        
        # Amostra aleatória
        if len(image_files) > sample_size:
            image_files = random.sample(image_files, sample_size)
        
        print(f"Analisando {len(image_files)} imagens...")
        
        # Processa imagens
        results = self.predict_batch([str(f) for f in image_files])
        
        # Calcula estatísticas
        obstacle_detections = [r for r in results if "ensemble_decision" in r and r["ensemble_decision"]["has_obstacle"]]
        total_valid = [r for r in results if "ensemble_decision" in r]
        
        if not total_valid:
            return {"error": "Nenhuma predição válida"}
        
        stats = {
            "total_images": len(image_files),
            "valid_predictions": len(total_valid),
            "obstacles_detected": len(obstacle_detections),
            "obstacle_rate": len(obstacle_detections) / len(total_valid),
            "average_confidence": np.mean([r["ensemble_decision"]["confidence"] for r in total_valid]),
            "average_uncertainty": np.mean([r["ensemble_decision"]["uncertainty"] for r in total_valid]),
            "model_weight_averages": {
                "yolo": np.mean([r["model_weights"]["yolo"] for r in total_valid]),
                "segnet": np.mean([r["model_weights"]["segnet"] for r in total_valid]),
                "autoencoder": np.mean([r["model_weights"]["autoencoder"] for r in total_valid])
            },
            "model_agreement_average": np.mean([r["model_agreement"] for r in total_valid])
        }
        
        return stats
    
    def save_models(self, save_dir: str):
        """Salva todos os modelos treinados."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        torch.save(self.yolo_model.state_dict(), save_path / "yolo_model.pth")
        torch.save(self.segnet_model.state_dict(), save_path / "segnet_model.pth") 
        torch.save(self.autoencoder_model.state_dict(), save_path / "autoencoder_model.pth")
        torch.save(self.ensemble_model.state_dict(), save_path / "ensemble_model.pth")
        
        print(f"Modelos salvos em {save_dir}")
    
    def load_models(self, load_dir: str):
        """Carrega modelos pré-treinados."""
        load_path = Path(load_dir)
        
        if (load_path / "yolo_model.pth").exists():
            self.yolo_model.load_state_dict(torch.load(load_path / "yolo_model.pth", map_location=self.device))
        
        if (load_path / "segnet_model.pth").exists():
            self.segnet_model.load_state_dict(torch.load(load_path / "segnet_model.pth", map_location=self.device))
        
        if (load_path / "autoencoder_model.pth").exists():
            self.autoencoder_model.load_state_dict(torch.load(load_path / "autoencoder_model.pth", map_location=self.device))
        
        if (load_path / "ensemble_model.pth").exists():
            self.ensemble_model.load_state_dict(torch.load(load_path / "ensemble_model.pth", map_location=self.device))
            
        print(f"Modelos carregados de {load_dir}")

def main():
    """Exemplo de uso do sistema."""
    
    # Inicializa sistema
    system = FieldSAFEEnsembleSystem()
    
    # Exemplo: Analisa algumas imagens do dataset
    if Path("datasets/fieldsafe_rgb_ready/split/test/images").exists():
        print("\n=== Analisando imagens de teste ===")
        
        # Predição de uma imagem
        test_images = list(Path("datasets/fieldsafe_rgb_ready/split/test/images").glob("*.png"))
        if test_images:
            result = system.predict_single_image(str(test_images[0]))
            print(f"\nResultado para {test_images[0].name}:")
            print(f"  Obstáculo detectado: {result['ensemble_decision']['has_obstacle']}")
            print(f"  Probabilidade: {result['ensemble_decision']['probability']:.3f}")
            print(f"  Confiança: {result['ensemble_decision']['confidence']:.3f}")
            print(f"  Pesos dos modelos: YOLO={result['model_weights']['yolo']:.3f}, "
                  f"SegNet={result['model_weights']['segnet']:.3f}, "
                  f"Autoencoder={result['model_weights']['autoencoder']:.3f}")
        
        # Análise estatística
        stats = system.analyze_dataset_predictions(
            "datasets/fieldsafe_rgb_ready/split/test/images", 
            sample_size=20
        )
        print(f"\n=== Estatísticas do Dataset (amostra) ===")
        print(f"Taxa de obstáculos: {stats['obstacle_rate']:.1%}")
        print(f"Confiança média: {stats['average_confidence']:.3f}")
        print(f"Concordância entre modelos: {stats['model_agreement_average']:.3f}")
        print(f"Pesos médios - YOLO: {stats['model_weight_averages']['yolo']:.3f}, "
              f"SegNet: {stats['model_weight_averages']['segnet']:.3f}, "
              f"Autoencoder: {stats['model_weight_averages']['autoencoder']:.3f}")
    
    else:
        print("Dataset de teste não encontrado. Execute primeiro o prepare_rgb_subset.py")

if __name__ == "__main__":
    main()