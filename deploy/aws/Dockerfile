FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    REFRAME_LAB_MODEL_DIR=/opt/reframe-models \
    REFRAME_OPENCV_DISTRIBUTION=standard

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY dist/reframe_lab-0.14.0-py3-none-any.whl /tmp/wheels/
COPY requirements/runtime.lock /tmp/constraints.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -c /tmp/constraints.txt \
        "opencv-contrib-python-headless==5.0.0.93" \
        boto3 \
        /tmp/wheels/reframe_lab-0.14.0-py3-none-any.whl \
    && python -m pip check \
    && reframe-lab doctor --strict-opencv-5 \
    && rm -rf /tmp/wheels

ARG DOWNLOAD_CORE_MODELS=1
RUN mkdir -p "$REFRAME_LAB_MODEL_DIR" \
    && if [ "$DOWNLOAD_CORE_MODELS" = "1" ]; then reframe-lab models --download core; fi

RUN useradd --create-home --uid 1000 reframe \
    && mkdir -p /work /tmp \
    && chown -R 1000:1000 /work /tmp "$REFRAME_LAB_MODEL_DIR"
USER 1000
WORKDIR /work
ENTRYPOINT ["reframe-lab"]
CMD ["doctor", "--strict-opencv-5"]
