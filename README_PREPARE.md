# Pipeline de Preparação — FieldSAFE RGB

Este pipeline prepara o subset RGB do FieldSAFE para os modelos YOLOv5, SegNet e Autoencoder.

## Requisitos

- Python 3.9+
- pacotes: `opencv-python`, `numpy`, `scikit-learn`

```bash
pip install opencv-python numpy scikit-learn
```

## Estrutura esperada (exemplo)

```
datasets/
  extract_rgb/
    seq01/images/...
    seq02/images/...
    ...
datasets/
  labels_semantic/           # opcional (máscaras PNG)
    seq01/labels/...
    seq02/labels/...
```

## Execução — Máscaras Semânticas

```bash
python3 prepare_rgb_subset.py \
  --rgb_root datasets/extract_rgb \
  --label_root datasets/labels_semantic \
  --outdir fieldsafe_rgb_ready \
  --resize 640 640 \
  --label_type semantic \
  --gen_yolo_from_mask
```

## Execução — Rótulos YOLO já prontos

```bash
python3 prepare_rgb_subset.py \
  --rgb_root datasets/extract_rgb \
  --label_root datasets/labels_yolo \
  --outdir fieldsafe_rgb_ready \
  --resize 640 640 \
  --label_type yolo
```

## Saídas

- `fieldsafe_rgb_ready/images/` — imagens padronizadas (640x640)
- `fieldsafe_rgb_ready/labels/` — labels compatíveis (máscara PNG ou .txt YOLO)
- `fieldsafe_rgb_ready/split/{train,val,test}/{images,labels}` — conjuntos finais
- `fieldsafe_rgb_ready/manifests/*.csv` — manifestos dos conjuntos
- `fieldsafe_rgb_ready/relatorio_summary.txt` — estatísticas e observações

## Notas

- A divisão 70/15/15 usa GroupShuffleSplit por sequência (evita vazamento).
- Para máscaras, a interpolação é `NEAREST` para não “borrar” rótulos.
- A opção `--gen_yolo_from_mask` cria .txt YOLO por componentes conectados.
