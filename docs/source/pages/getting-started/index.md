# Getting started

```{toctree}
:hidden:
using-docker/index.md
using-venv/index.md
```

## Overview

This section provides information on how to install DYNO and integrate it in
your own application.

The recommended installation method is [using the Docker
image](./using-docker/index.md#using-docker). There are Docker image suitable
for developers or end users. If you are unable to use the Docker image, you can
[run MOMA using a virtual
environment](./building-from-source/index.md#using-venv).

## Quick start

If you are unsure about your specific application and just want to try and
quicky run MOMA on your machine, you can run the following Docker command to get
the latest runtime:

```shell
docker run -it ghcr.io/aarhus-robotics/moma:runtime
```

Once inside the container, generate a simple map using default options with the
following command:

```shell
moma --clear mobility --show_plots --vehicle mungo --site Vejlby --n_samples 10 --sample_reduction quantile 0.9 --path_coordinate_system utm --path_start 573500 6230500 --path_end 575900 6230250
```
