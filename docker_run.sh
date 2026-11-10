#!/bin/bash
# rendering programowy (mesa), więc nie trzeba GPU ani /dev/dri
# buduje obraz i wchodzi do kontenera z repo w /workspace i oknami na ekranie hosta
set -e
cd "$(dirname "$0")"
docker build -t znimpra-mppi .
xhost +local: >/dev/null 2>&1 || true
docker run --rm -it --net=host \
    -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$PWD":/workspace znimpra-mppi
