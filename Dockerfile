# Prosty obraz do laboratorium MPPI: Python 3.12 + gymnasium/mujoco, okna przez X11 hosta
FROM python:3.12-slim

# biblioteki do okien: OpenGL (MuJoCo/GLFW), X11 (pygame w Pendulum), mesa jako rendering programowy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglx-mesa0 libgl1-mesa-dri \
        libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon-x11-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

# pakiety globalnie, żeby wystarczało zwykłe python3 (bez venv)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# użytkownik z UID hosta, żeby pliki w results/ nie należały do roota
ARG USER_UID=1000
RUN useradd -m -u ${USER_UID} -s /bin/bash student
USER student

# bez tego mesa próbuje sterownika GPU (iris) przez X i okno się zawiesza
ENV LIBGL_ALWAYS_SOFTWARE=1 \
    MUJOCO_GL=glfw \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
CMD ["bash"]
