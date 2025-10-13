
# TCC — Fusão Multimodel Supervisionada (RGB) no FieldSAFE

Repositório do TCC de Osvaldo Jeronymo (UFPR — Data Science & Big Data).  
Objetivo: adaptar o framework de Mujkic et al. (2023) para um cenário **unimodal (RGB)** do dataset **FieldSAFE**, 
implementando uma **ensemble layer supervisionada e interpretável** que combina YOLOv5, SegNet e Autoencoder.

> **Pergunta central:** Como melhorar a **precisão** e a **confiabilidade** das decisões quando se dispõe **apenas de uma câmera RGB**?

## Estrutura

```
.
├── README.md
├── LICENSE                # MIT (código)
├── LICENSE.docs           # CC BY 4.0 (textos/figuras)
├── requirements.txt
├── environment.yml
├── Makefile
├── CITATION.cff
├── pyproject.toml
├── .gitignore
├── .github/workflows/ci.yml
├── docs/
│   ├── metodologia.md
│   └── figuras/
├── notebooks/
│   ├── 01_exploracao_fieldsafe_rgb.ipynb
│   ├── 02_preprocessamento_rgb.ipynb
│   ├── 03_predicoes_modelos.ipynb
│   └── 04_fusao_ensemble.ipynb
├── src/
│   ├── data/
│   │   ├── extract_rosbag.py
│   │   └── prepare_rgb_subset.py
│   ├── models/
│   │   ├── segnet.py
│   │   ├── autoencoder.py
│   │   └── ensemble.py
│   └── utils/
│       ├── metrics.py
│       └── viz.py
├── docker/
│   └── ros_noetic_extractor.Dockerfile
└── experiments/
    ├── configs/
    │   ├── dataset.yml
    │   └── ensemble.yml
    └── runs/
```

## Quickstart

```bash
# 1) Clone
git clone https://github.com/<usuario>/tcc-fieldsafe-rgb-ensemble.git
cd tcc-fieldsafe-rgb-ensemble

# 2) Ambiente (conda recomendado)
conda env create -f environment.yml
conda activate tcc-fieldsafe

# 3) (Opcional) ROS via Docker para extração de RGB de .bag
docker build -t fieldsafe:ros -f docker/ros_noetic_extractor.Dockerfile .
# Exemplo (ajuste caminhos e tópico):
docker run --rm -it -v $PWD:/work -v /path/to/bags:/bags fieldsafe:ros   bash -lc "python3 /work/src/data/extract_rosbag.py --bag /bags/2016-10-25-11-41-21.bag --topic /camera/rgb/image_color --outdir /work/datasets/extract_114121"

# 4) Preparar subset RGB (normalização + split 70/15/15)
python -m src.data.prepare_rgb_subset --input datasets/extract_114121 --output datasets/fieldsafe_rgb_ready
```

## Dataset

- FieldSAFE (Kragh et al., 2017): https://github.com/mikkelkh/FieldSAFE  
- **Atenção:** Não versionar dados brutos no Git. Use **DVC** (opcional) ou mantenha os dados em Google Drive:
  - `/datasets/` (local) — sincronizado com Drive
  - Link privado do autor: (preencher)

## Citação
Ver `CITATION.cff`.

## Licenças
- Código: MIT
- Documentação (textos, figuras): CC BY 4.0
