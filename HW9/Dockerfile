# from the provided image that has aws lambda and onnx in ./
FROM agrigorev/model-2025-hairstyle:v1



# use uv instead of pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml uv.lock ./
RUN uv export --format requirements-txt > requirements.txt
RUN uv pip install --system -r requirements.txt



# script.py contains the codes worked out in notebook
COPY script.py .

