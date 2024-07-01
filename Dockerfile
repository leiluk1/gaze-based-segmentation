FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    git \
    python3.9 python3.9-dev python3.9-venv python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN apt-get install wget -y
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.0-545.23.06-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.0-545.23.06-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2004-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-3

COPY requirements.txt /repo/requirements.txt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
bash ~/miniconda.sh -b -p $HOME/miniconda && eval "$(/root/miniconda/bin/conda shell.bash hook)" && \
conda init && conda config --set auto_activate_base true && \
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
git clone https://github.com/bowang-lab/MedSAM && pip install -e MedSAM/

WORKDIR /repo
RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && \
pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]