
.PHONY: help setup lint test docs

help:
	@echo "make setup       # install deps (pip)"
	@echo "make detect-webcam-topic BAG=<path>   # detect likely webcam image topic from rosbag"
	@echo "make docker-build                      # build ROS Noetic extractor image"
	@echo "make extract-webcam BAG=<path> OUTDIR=<dir> [EVERY_N=1]  # auto-detect topic and extract via Docker"
	@echo "make trace-demo                         # run single-image ETL+degradation demo with side-by-side viz"
	@echo "make lint        # run basic lint (flake8 if present)"
	@echo "make test        # run tests (if any)"
	@echo "make docs        # build docs (if any)"

setup:
	/home/osvaldo/.venvs/fieldsafe/bin/python -m pip install -r requirements.txt || true

lint:
	@echo "No linter configured yet."

test:
	@echo "No tests yet."

docs:
	@echo "No docs pipeline yet."

detect-webcam-topic:
	/home/osvaldo/.venvs/fieldsafe/bin/python src/data/detect_webcam_topic.py "$(BAG)"

docker-build:
	docker build -t fieldsafe-noetic -f docker/ros_noetic_extractor.Dockerfile .

extract-webcam: docker-build
	$(eval TOPIC := $(shell /home/osvaldo/.venvs/fieldsafe/bin/python src/data/detect_webcam_topic.py "$(BAG)" --quiet))
	@if [ -z "$(TOPIC)" ]; then echo "Falha ao detectar tópico de webcam."; exit 2; fi
	@echo "Usando tópico: $(TOPIC)"
	docker run --rm \
	  -v "$(HOME)/Área de Trabalho:/data" \
	  -v "$(PWD)/src/data:/app" \
	  fieldsafe-noetic \
	  bash -lc 'python3 /app/extract_rosbag.py --bag "$(BAG)" --topic "$(TOPIC)" --outdir "$(OUTDIR)" --every_n "$(EVERY_N)"'

# Demo didático: ETL + Degradação + Visualização por imagem
trace-demo:
	@IMG?="/home/osvaldo/Área de Trabalho/TCC_FieldSAFE/datasets/extract_rgb/2016-10-25-11-09-42/images/rgb_1477386583133442258.png"; \
	OUTDIR?="/home/osvaldo/Área de Trabalho/TCC_FieldSAFE/datasets/trace_demo"; \
	SEQ?="2016-10-25-11-09-42"; \
	IMAGE_ID?="rgb_1477386583133442258"; \
	NOISE?=25.0; \
	JPEG?=40; \
	LABEL?=""; \
	/home/osvaldo/.venvs/fieldsafe/bin/python src/data/trace_etl_single.py $$IMG $$OUTDIR $$SEQ $$IMAGE_ID $$LABEL --size 640x640 --noise $$NOISE --jpeg $$JPEG --annotate; \
	echo "Saídas em: $$OUTDIR"
