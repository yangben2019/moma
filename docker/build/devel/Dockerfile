FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LOCALE=en_US
ARG USER=robot

RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
        build-essential \
        cm-super \
        dvipng \
        git \
        libgles2-mesa-dev \
        libglfw3-dev \
        libosmesa6 \
        libx11-dev \
        libxrender1 \
        locales \
        pipx \
        python3 \
        python3-tk \
        python3-venv \
        qtwayland5 \
        texlive \
        texlive-fonts-recommended \
        texlive-latex-extra && \
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
    echo "export PATH='/home/${USER}/.local/bin:${PATH}'" >> /home/${USER}/.bashrc

# Set the default container user
USER ${USER}

# Set the default shell to a Bash login shell
SHELL ["/bin/bash", "-l", "-c" ]

RUN pipx install poetry

# Set the working directory to the home folder of the container user
WORKDIR /home/${USER}

###############################################################################
# Add container labels                                                        #
###############################################################################

LABEL org.opencontainers.image.authors="Dario Sirangelo (dev@dariosirangelo.me)"
LABEL org.opencontainers.image.url=https://github.com/au-agv/moma
LABEL org.opencontainers.image.documentation=https://au-agv.github.io/moma
LABEL org.opencontainers.image.source=https://github.com/au-agv/moma/tree/main/docker/build/devel
LABEL org.opencontainers.image.version=main
LABEL org.opencontainers.image.licenses=MIT
LABEL org.opencontainers.image.title="MOMA (devel)"
LABEL org.opencontainers.image.description="Mobility maps and optimal routes generator for ground vehicles "
