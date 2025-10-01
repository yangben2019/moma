FROM ubuntu:22.04

ARG USER=robot
ARG LOCALE=en_US

RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
        build-essential \
        git \
        libgles2-mesa-dev \
        libglfw3-dev \
        libosmesa6 \
        libx11-dev \
        libxrender1 \
        locales \
        pipx \
        python3 \
        python3-venv && \
    echo "LC_ALL=${LOCALE}.UTF-8" >> /etc/environment && \
    echo "${LOCALE}.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "LANG=${LOCALE}.UTF-8" > /etc/locale.conf && \
    locale-gen && \
    apt-get -qq clean && rm -rf /var/lib/apt/lists/*

ENV LANG=${LOCALE}.UTF-8
ENV LANGUAGE=${LOCALE}:en
ENV LC_ALL=${LOCALE}.UTF-8

RUN useradd --create-home --shell /bin/bash ${USER} && \
    echo "export LANG=${LOCALE}.UTF-8" >> /home/${USER}/.profile && \
    echo "export PATH='/home/${USER}/.local/bin:${PATH}'" >> /home/${USER}/.profile

# Set the default container user
USER ${USER}

# Set the default shell to a Bash login shell
SHELL ["/bin/bash", "-l", "-c" ]

RUN pipx install poetry

# Set the working directory to the home folder of the container user
WORKDIR /home/${USER}

