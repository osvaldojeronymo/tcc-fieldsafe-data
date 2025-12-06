# Checklist de Coerência TCC ↔ Repositórios FieldSAFE

Este checklist mapeia as seções do TCC aos componentes implementados nos repositórios `tcc-fieldsafe-data`, `tcc-fieldsafe-models` e `tcc-fieldsafe-ensemble`, e aponta lacunas pendentes.

## Mapeamento por Seção

- Introdução / Objetivos:
  - Dataset FieldSAFE e foco em RGB: `README.md` e `README_PREPARE.md`.
  - Métodos (SegNet, YOLO, Autoencoder): estruturas em `tcc-fieldsafe-models/src/models/`.

- Metodologia / Método Ensemble:
  - Base do ensemble: `tcc-fieldsafe-ensemble/src/fusion/engine.py`.
  - Configuração: `tcc-fieldsafe-ensemble/experiments/configs/ensemble_config.yml`.

- Materiais e Métodos / Dados de entrada:
  - Detecção de tópico: `tcc-fieldsafe-data/src/data/detect_webcam_topic.py`.
  - Extração de frames: `tcc-fieldsafe-data/src/data/extract_rosbag.py` com Docker `ros_noetic_extractor.Dockerfile`.
  - Manifests de extração: `datasets/extract_rgb/<seq>/frames_manifest.csv`.

- Pré-processamento e rastreabilidade (ETL):
  - Script: `tcc-fieldsafe-data/src/data/preprocess_with_manifests.py`.
  - Saídas: `datasets/preprocessed_yolo/<seq>/manifest.csv` e `datasets/preprocessed_segnet/<seq>/manifest.csv`.
  - Parâmetros rastreados: resize, ruído, compressão, caminhos de rótulos.

- Protocolo de degradação controlada:
  - Implementado via parâmetros de ruído (gaussiano/sal-pimenta) e compressão (JPEG/PNG) no `manifest.csv`.

- Reprodutibilidade Computacional:
  - Makefile e ambientes: `tcc-fieldsafe-data/Makefile`, `environment.yml`, `requirements.txt`.
  - Docker: `tcc-fieldsafe-data/docker/ros_noetic_extractor.Dockerfile`.
  - Guia de anotação: `tcc-fieldsafe-data/docs/guia_anotacao_webcam.md`.

## Lacunas Identificadas

- Anotação em escala:
  - Pendente exportar rótulos do Label Studio para `datasets/labels_yolo` e `datasets/labels_semantic`.

- Treinamento e Métricas:
  - Scripts/notebooks de treino para YOLO/SegNet com logs de métricas (mAP, mIoU) ainda não consolidados.
  - Registro em `tcc-fieldsafe-ensemble/docs/experiments_log.md` a ser ampliado com resultados.

- Avaliação do Ensemble:
  - Pipeline completo de avaliação, agregação de métricas e `show_final_summary.py` precisa ser conectado aos artefatos de predição reais.

- Figuras/Visualizações:
  - Garantir conjunto de imagens em `docs/imgs/` cobrindo exemplos de letterbox, máscaras, e casos de degradação controlada.

## Próximas Ações Recomendas

- Finalizar anotação e reprocessar:
  - Exportar rótulos; executar `preprocess_with_manifests.py` para preencher `label_path`; validar com `tools/validate_labels_against_manifests.py`.

- Treinar modelos e registrar métricas:
  - Adicionar scripts de treino; salvar métricas e configurações no `docs/experiments_log.md`.

- Integrar e avaliar ensemble:
  - Conectar predições dos três modelos ao `fusion/engine.py`; gerar sumários com `show_final_summary.py`.

- Atualizar documentação:
  - Seção de reprodutibilidade com checklist prático dos comandos executados (Makefile, Docker, preprocess, validação).
