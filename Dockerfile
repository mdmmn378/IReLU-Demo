FROM python:3.9.5-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip &&\
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu &&\
    pip install -r requirements.txt 
COPY . .
RUN pip install . 

ARG USERNAME=TEST
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN sudo apt update && sudo apt install dumb-init
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD jupyter lab --port 8888 --ip 0.0.0.0

