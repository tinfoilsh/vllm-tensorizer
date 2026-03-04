FROM vllm/vllm-openai:v0.16.0-cu130@sha256:9e6e8a306b46b249d637d6da056753fd258f89ad84bf0e12afc579124571288e

RUN pip install --no-cache-dir tensorizer==2.11.0

COPY patch_tensorizer_tp.py /opt/patch_tensorizer_tp.py
RUN python3 /opt/patch_tensorizer_tp.py
