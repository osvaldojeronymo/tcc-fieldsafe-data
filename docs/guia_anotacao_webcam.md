# Guia de Anotação — Webcam Logitech C920 (FieldSAFE)

Este guia padroniza a rotulagem das imagens RGB extraídas da webcam Logitech C920 para os dois tipos de tarefas do sistema híbrido: Detecção (YOLO) e Segmentação Semântica (SegNet).

## Objetivos

- Garantir consistência e reprodutibilidade das anotações.
- Cobrir variabilidade de cenário (iluminação, distância, oclusões, ângulos).
- Minimizar redundância temporal (usar subconjunto extraído com `EVERY_N=10`).

## Estrutura de Dados

- Imagens por sequência: `~/Área de Trabalho/TCC_FieldSAFE/datasets/extract_rgb/<SEQ>/images/*.png`
- Manifesto: `~/Área de Trabalho/TCC_FieldSAFE/datasets/extract_rgb/<SEQ>/frames_manifest.csv`
- Projeto de anotação sugerido (Label Studio): `~/Área de Trabalho/TCC_FieldSAFE/annotation_projects/fieldsafe_webcam/`

## Classes (Detecção YOLO)

1. `tractor`
2. `combine`
3. `trailer`
4. `combine header`
5. `baler`
6. `square bale`
7. `round bale`
8. `human`

Observações:
- Anotar todos os objetos visíveis pertencentes às classes acima.
- Em casos de oclusão parcial, anotar o que é visível; não extrapolar além do contorno observável.
- Se múltiplos objetos da mesma classe aparecerem, cada um recebe sua própria caixa.

## Critérios de Detecção (YOLO)

- Caixa “justa”: os limites devem tangenciar o objeto, evitando excesso de fundo.
- Orientação: boxes sempre eixos alinhados à imagem (YOLO padrão, sem rotação).
- Oclusões: anotar a parte visível; se a oclusão impede identificação de classe, não anotar.
- Objetos parciais nas bordas: anotar se >30% do objeto está dentro do frame.
- Ambiguidade: se não for possível distinguir entre `square bale` e `round bale`, anotar como a classe mais provável; registrar casos ambíguos para revisão.

## Critérios de Segmentação (SegNet)

- Foco: áreas cultiváveis e elementos agrícolas relevantes.
- Máscara por pixel (PNG indexada): cada pixel recebe o índice da classe; para áreas não rotuladas usar `background` (se adotado no projeto) ou ausência de máscara.
- Bordas: usar ferramental de polígono/livre com atenção às transições; evitar serrilhado excessivo.
- Oclusões: rotular o que é visível, sem inferir forma oculta.

## Exemplos Visuais

Use imagens de referência das pastas extraídas para montar exemplos no Label Studio:
- Exemplo 1 (Detecção): ![YOLO exemplo 1](imgs/exemplo_yolo_1.png) — anotar `tractor`, `trailer` se presentes, boxes justos.
- Exemplo 2 (Detecção): ![YOLO exemplo 2](imgs/exemplo_yolo_2.png) — anotar `human` se visível; cuidado com escala pequena.
- Exemplo 3 (Segmentação): ![SegNet exemplo 1](imgs/exemplo_segnet_1.png) — rotular região cultivável e separar objetos agrícolas por classe.

Sugestão: cole capturas de tela do Label Studio demonstrando boas e más anotações (salvar em `docs/imgs/` e referenciar aqui) — opcional.

## Fluxo de Trabalho (Label Studio)

1. Inicializar com Docker:
   ```bash
   docker run -d --name labelstudio -p 8080:8080 \
     -v "$HOME/Área de Trabalho/TCC_FieldSAFE/annotation_projects:/labelstudio/data" \
     heartexlabs/label-studio:latest
   ```
2. Criar projeto “FieldSAFE Webcam — YOLO & SegNet”.
3. Importar imagens: selecione `datasets/extract_rgb/<SEQ>/images` (sequências com `EVERY_N=10`).
4. Templates: “Image Bounding Boxes” (YOLO) e “Image Segmentation” (SegNet).
5. Definir classes (cores distintas por classe).
6. Anotar seguindo os critérios deste guia.
7. Revisar 10–15% das anotações por segunda pessoa (ou segunda passagem) para consistência.

## Controle de Qualidade (QC)

- YOLO:
  - Validar que todas as caixas estão dentro dos limites da imagem.
  - Remover boxes com largura/altura < 2 px (ruído).
  - Conferir classes válidas (sem nomes fora da lista).
- SegNet:
  - Verificar paleta/índices corretos por classe.
  - Ausência de buracos impossíveis na máscara (verificar conectividade quando adequado).
  - Tamanho mínimo de regiões para evitar rótulos espúrios.

## Exportação

- YOLO: TXT por imagem (`class x_center y_center width height`, normalizados em [0,1]).
- SegNet: PNG indexada por pixel.
- Armazenar em estrutura: `datasets/labels_yolo/<SEQ>/` e `datasets/labels_semantic/<SEQ>/`.

## Integração com o Pipeline

Usar `prepare_rgb_subset.py` para padronizar e gerar splits:
```bash
python3 src/data/prepare_rgb_subset.py \
  --rgb_root datasets/extract_rgb \
  --label_root datasets/labels_yolo \
  --outdir fieldsafe_rgb_ready \
  --resize 640 640 \
  --label_type yolo

python3 src/data/prepare_rgb_subset.py \
  --rgb_root datasets/extract_rgb \
  --label_root datasets/labels_semantic \
  --outdir fieldsafe_rgb_ready \
  --resize 640 640 \
  --label_type semantic
```

## Boas Práticas

- Documentar casos ambíguos e decisões em um `docs/experiences_log.md` (ou no projeto LS).
- Garantir distribuição por sequência e classe para evitar viés.
- Atualizar este guia conforme feedback dos anotadores e resultados dos modelos.
