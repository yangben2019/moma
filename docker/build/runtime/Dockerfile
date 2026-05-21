FROM ghcr.io/au-agv/moma:devel

ARG USER=robot
ARG MOMA_PATH="/home/robot/workspace"
ARG VENV_PATH=/home/${USER}/.venv 

RUN mkdir -p ${MOMA_PATH}/{data,src}

COPY poetry.lock pyproject.toml README.md ${MOMA_PATH}
COPY data ${MOMA_PATH}/data
COPY src ${MOMA_PATH}/src

ENV VIRTUAL_ENV=${MOMA_PATH}/.venv
ENV PATH="${MOMA_PATH}/.venv/bin:$PATH"

RUN cd ${MOMA_PATH} && \
    python3 -m venv .venv && \
    poetry install

# Set the working directory to the home folder of the container user
WORKDIR ${MOMA_PATH}

###############################################################################
# Add container labels                                                        #
###############################################################################

LABEL org.opencontainers.image.authors="Dario Sirangelo (dev@dariosirangelo.me)"
LABEL org.opencontainers.image.url=https://github.com/au-agv/moma
LABEL org.opencontainers.image.documentation=https://au-agv.github.io/moma
LABEL org.opencontainers.image.source=https://github.com/au-agv/moma/tree/main/docker/build/runtime
LABEL org.opencontainers.image.version=main
LABEL org.opencontainers.image.licenses=MIT
LABEL org.opencontainers.image.title="MOMA (runtime)"
LABEL org.opencontainers.image.description="Mobility maps and optimal routes generator for ground vehicles "
