#!/usr/bin/env python3
"""
Script de validação do pré-processamento FieldSAFE.

Verifica compatibilidade dos dados processados com os modelos ensemble:
- Tamanhos das imagens
- Ranges de valores
- Formato de entrada dos modelos
- Carregamento de amostras
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
import argparse
from typing import Tuple, Dict

def validate_image_format(image_path: str, expected_size: Tuple[int, int] = (640, 640)) -> Dict:
    """
    Valida formato e propriedades de uma imagem processada.
    
    Args:
        image_path: Caminho para a imagem
        expected_size: Tamanho esperado (width, height)
        
    Returns:
        Dict com resultados da validação
    """
    try:
        # Carrega imagem
        image = cv2.imread(image_path)
        if image is None:
            return {"valid": False, "error": "Não foi possível carregar a imagem"}
        
        # Converte BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Verifica dimensões
        height, width, channels = image_rgb.shape
        size_valid = (width, height) == expected_size
        
        # Verifica range de valores
        min_val, max_val = image_rgb.min(), image_rgb.max()
        
        # Converte para tensor (como os modelos fariam)
        tensor = torch.from_numpy(image_rgb).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        return {
            "valid": True,
            "size": (width, height),
            "size_valid": size_valid,
            "channels": channels,
            "dtype": str(image_rgb.dtype),
            "value_range": (float(min_val), float(max_val)),
            "tensor_shape": tuple(tensor.shape),
            "memory_mb": tensor.element_size() * tensor.numel() / (1024**2)
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

def validate_normalized_format(image_path: str, expected_range: Tuple[float, float] = (0.0, 1.0)) -> Dict:
    """
    Valida formato da imagem normalizada.
    
    Args:
        image_path: Caminho para imagem normalizada
        expected_range: Range esperado dos valores
        
    Returns:
        Dict com resultados da validação
    """
    try:
        # Carrega imagem normalizada
        image = cv2.imread(image_path)
        if image is None:
            return {"valid": False, "error": "Não foi possível carregar a imagem normalizada"}
        
        # Converte para float e normaliza de volta para [0,1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        
        # Verifica range
        min_val, max_val = image_float.min(), image_float.max()
        range_valid = (min_val >= expected_range[0] - 0.01 and 
                      max_val <= expected_range[1] + 0.01)
        
        return {
            "valid": True,
            "value_range": (float(min_val), float(max_val)),
            "expected_range": expected_range,
            "range_valid": range_valid,
            "mean_values": [float(image_float[:,:,i].mean()) for i in range(3)],
            "std_values": [float(image_float[:,:,i].std()) for i in range(3)]
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

def test_model_compatibility(image_path: str) -> Dict:
    """
    Testa compatibilidade com os modelos ensemble.
    
    Args:
        image_path: Caminho para imagem processada
        
    Returns:
        Dict com resultados dos testes
    """
    try:
        # Simula carregamento como os modelos fariam
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normaliza para [0,1] (como os modelos fazem)
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        # Converte para tensor PyTorch
        tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        
        # Simula entrada nos modelos
        results = {}
        
        # Teste YOLOv5 (normalmente usa normalização ImageNet)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        yolo_input = (tensor - imagenet_mean) / imagenet_std
        
        results["yolo"] = {
            "input_shape": tuple(yolo_input.shape),
            "value_range": (float(yolo_input.min()), float(yolo_input.max())),
            "memory_mb": yolo_input.element_size() * yolo_input.numel() / (1024**2)
        }
        
        # Teste SegNet (geralmente usa mesma normalização)
        results["segnet"] = {
            "input_shape": tuple(yolo_input.shape),
            "value_range": (float(yolo_input.min()), float(yolo_input.max())),
            "memory_mb": yolo_input.element_size() * yolo_input.numel() / (1024**2)
        }
        
        # Teste Autoencoder (pode usar [0,1] ou [-1,1])
        autoencoder_input = tensor  # [0,1]
        results["autoencoder"] = {
            "input_shape": tuple(autoencoder_input.shape),
            "value_range": (float(autoencoder_input.min()), float(autoencoder_input.max())),
            "memory_mb": autoencoder_input.element_size() * autoencoder_input.numel() / (1024**2)
        }
        
        return {"valid": True, "models": results}
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

def validate_dataset_structure(data_dir: str) -> Dict:
    """
    Valida estrutura completa do dataset processado.
    
    Args:
        data_dir: Diretório dos dados processados
        
    Returns:
        Dict com validação da estrutura
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return {"valid": False, "error": f"Diretório não encontrado: {data_dir}"}
    
    results = {"valid": True, "structure": {}}
    
    # Verifica estrutura esperada
    expected_dirs = [
        "split/train/images",
        "split/val/images", 
        "split/test/images",
        "split_normalized/train/images",
        "split_normalized/val/images",
        "split_normalized/test/images",
        "manifests"
    ]
    
    for dir_path in expected_dirs:
        full_path = data_path / dir_path
        exists = full_path.exists()
        
        if exists:
            count = len(list(full_path.glob("*.png")))
            results["structure"][dir_path] = {"exists": True, "count": count}
        else:
            results["structure"][dir_path] = {"exists": False, "count": 0}
            results["valid"] = False
    
    # Verifica arquivos de relatório
    report_files = ["preprocessing_report.txt", "preprocessing_metadata.json"]
    for file_name in report_files:
        file_path = data_path / file_name
        results["structure"][file_name] = {"exists": file_path.exists()}
        
        if not file_path.exists():
            results["valid"] = False
    
    return results

def run_validation(data_dir: str, num_samples: int = 10) -> Dict:
    """
    Executa validação completa do dataset processado.
    
    Args:
        data_dir: Diretório dos dados processados
        num_samples: Número de amostras para testar
        
    Returns:
        Dict com resultados completos da validação
    """
    print("=== VALIDAÇÃO DO PRÉ-PROCESSAMENTO FIELDSAFE ===")
    
    results = {
        "dataset_valid": False,
        "structure_validation": {},
        "sample_validations": [],
        "compatibility_tests": [],
        "summary": {}
    }
    
    # 1. Valida estrutura do dataset
    print("1. Validando estrutura do dataset...")
    structure_result = validate_dataset_structure(data_dir)
    results["structure_validation"] = structure_result
    
    if not structure_result["valid"]:
        print("❌ Estrutura do dataset inválida!")
        return results
    
    print("✅ Estrutura do dataset válida")
    
    # 2. Testa amostras de imagens
    print(f"2. Testando {num_samples} amostras de imagens...")
    
    data_path = Path(data_dir)
    test_images = list((data_path / "split/test/images").glob("*.png"))
    test_normalized = list((data_path / "split_normalized/test/images").glob("*.png"))
    
    if len(test_images) == 0:
        print("❌ Nenhuma imagem de teste encontrada!")
        return results
    
    # Testa amostras
    sample_size = min(num_samples, len(test_images))
    valid_samples = 0
    
    for i in range(sample_size):
        img_path = test_images[i]
        norm_path = test_normalized[i] if i < len(test_normalized) else None
        
        # Valida imagem original processada
        img_result = validate_image_format(str(img_path))
        
        # Valida imagem normalizada
        norm_result = {}
        if norm_path:
            norm_result = validate_normalized_format(str(norm_path))
        
        # Testa compatibilidade com modelos
        compat_result = test_model_compatibility(str(img_path))
        
        sample_validation = {
            "image_path": str(img_path),
            "image_validation": img_result,
            "normalized_validation": norm_result,
            "model_compatibility": compat_result
        }
        
        results["sample_validations"].append(sample_validation)
        
        if (img_result.get("valid", False) and 
            norm_result.get("valid", True) and 
            compat_result.get("valid", False)):
            valid_samples += 1
    
    # 3. Carrega metadados para verificação
    metadata_path = data_path / "preprocessing_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        results["metadata"] = metadata
    
    # 4. Sumário final
    results["summary"] = {
        "structure_valid": structure_result["valid"],
        "samples_tested": sample_size,
        "samples_valid": valid_samples,
        "validation_rate": valid_samples / sample_size if sample_size > 0 else 0,
        "total_images": {
            "train": structure_result["structure"]["split/train/images"]["count"],
            "val": structure_result["structure"]["split/val/images"]["count"],
            "test": structure_result["structure"]["split/test/images"]["count"]
        }
    }
    
    results["dataset_valid"] = (
        structure_result["valid"] and 
        valid_samples == sample_size
    )
    
    # Relatório final
    if results["dataset_valid"]:
        print(f"✅ Dataset válido! {valid_samples}/{sample_size} amostras testadas com sucesso")
        print(f"📊 Total: {sum(results['summary']['total_images'].values())} imagens processadas")
    else:
        print(f"❌ Dataset inválido! {valid_samples}/{sample_size} amostras válidas")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Validação do pré-processamento FieldSAFE")
    parser.add_argument("--data_dir", required=True, 
                       help="Diretório com dados pré-processados")
    parser.add_argument("--samples", type=int, default=10,
                       help="Número de amostras para testar (padrão: 10)")
    parser.add_argument("--save_report", action="store_true",
                       help="Salva relatório de validação em JSON")
    
    args = parser.parse_args()
    
    # Executa validação
    results = run_validation(args.data_dir, args.samples)
    
    # Salva relatório se solicitado
    if args.save_report:
        report_path = Path(args.data_dir) / "validation_report.json"
        
        # Converte numpy/torch types para JSON-serializable
        def convert_to_json_serializable(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            else:
                return obj
        
        json_results = convert_to_json_serializable(results)
        
        with open(report_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"📄 Relatório salvo em: {report_path}")
    
    return 0 if results["dataset_valid"] else 1

if __name__ == "__main__":
    exit(main())