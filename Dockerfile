# PyTorch w/ CUDA 12.1 runtime; works on HF GPU runners
# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_GENERATOR=Ninja
ENV CMAKE_MAKE_PROGRAM=Ninja

# System deps for C++ build
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++-12 gcc-12 ninja-build cmake wget unzip patchelf pkg-config ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

# (Optional) OpenCV dev headers if your C++ target needs them
# RUN apt-get update && apt-get install -y libopencv-dev && rm -rf /var/lib/apt/lists/*

# Conan v2 (if histmapseg or its deps use it)
RUN pip install --no-cache-dir conan==2.* 

# App workspace
WORKDIR /app

# Clone your repo (or COPY . if you push code into the Space repo)
# If you already vendored the repo contents into the Space, replace with: COPY . /app
# RUN git clone --depth=1 https://github.com/soduco/Benchmark_historical_map_vectorization.git src
RUN wget https://github.com/eunoia/Benchmark_historical_map_vectorization/archive/refs/heads/main.zip
RUN unzip main.zip
RUN mv Benchmark_historical_map_vectorization-main src
WORKDIR /app/src/watershed/histmapseg

# ---- Build the C++ watershed/histmapseg ----
# If this subproject uses Conan + CMake (typical):
# Generate a local profile (avoids interactive prompts)
RUN conan profile detect || true

RUN mkdir dockerbuild
WORKDIR dockerbuild
# Install deps and generate CMake toolchain files
RUN conan remote add lrde-public https://artifactory.lre.epita.fr/artifactory/api/conan/lrde-public --force
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

RUN conan install .. -of build -b missing \
  -g CMakeDeps -g CMakeToolchain -g VirtualRunEnv \
  -s compiler=gcc -s compiler.version=12 -s compiler.libcxx=libstdc++11 -s compiler.cppstd=20 \
  -c tools.cmake.cmaketoolchain:generator=Ninja \
  -c tools.build:compiler_executables='{"c":"/usr/bin/gcc","cpp":"/usr/bin/g++"}'

RUN cmake -S .. -B build -G Ninja   -DCMAKE_TOOLCHAIN_FILE="$PWD/build/conan_toolchain.cmake"   -DCMAKE_PREFIX_PATH="$PWD/build"   -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=   -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20
RUN cmake --build build -j"$(nproc)"

RUN cp build/histmapseg /usr/local/bin/histmapseg

# # ---- Python deps for the UNet inference wrapper ----
WORKDIR /app/src
# COPY requirements.txt /app/req.txt
RUN wget https://github.com/soduco/Benchmark_historical_map_vectorization/releases/download/pretrain/unet_best_weight.pth
RUN mv unet_best_weight.pth models
RUN pip install --no-cache-dir -r requirements.txt

# App entry (Gradio/CLI). If you use Gradio UI:
# COPY app.py /app/app.py

# Expose Gradio default port for Spaces
ENV PORT=7860
CMD ["python","app.py"]