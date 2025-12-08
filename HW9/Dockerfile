# from the provided image that has aws lambda and onnx in ./
FROM agrigorev/model-2024-hairstyle:v3

# use uv instead of pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r <(uv export --format requirements-txt)



# install the necessary packages
# onnx for onnx, pillow for image loading, numpy is included in onnx
RUN pip install onnx pillow

# script.py contains the codes worked out in notebook
COPY script.py .

