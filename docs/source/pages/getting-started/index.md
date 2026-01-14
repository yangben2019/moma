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
quickly run MOMA on your machine, you can run the following Docker command to get
the latest runtime:

```shell
docker run -it ghcr.io/aarhus-robotics/moma:runtime
```
<span style="color:red">Activate environment?</span> 


Once inside the container, generate a simple map using default options with the
following command:

```shell
moma --clear mobility --vehicle mock --site testmap --n_samples 50 --sample_reduction mean --directions N NE E SE S SW W NW omni --path_coordinate_system utm --path_start 10 95 --path_end 255 95
```

## Step guide

For a better understanding in the functions of MOMA follow this guide:

1. Test MOMA installation

   test the MOMA installation by navigating to the main moma directory and run the pytest
```shell
pytest
```
  The test should create a tes directory called `test/output` where a run of the moma map generation is done to test that all the functions runs correct. Output maps are placed in `test/output/sites/export`. These does not contain valid data due to the limited test map site used to run the test quickly.

1. Generate the test map
   <ol type="a">
     <li>something with the parameters</li>
   </ol>

1. Run moma

   something + `--clear` + `\\-\\-show_plots` if not in docker

1. Used more parameters

   The output generated with the following parameters can be seen
   <span style="color:red">Where?</span>

   Refer example calls  
   Refer full parameter

   <ol type="a">
     <li>parset 1</li>
     <li>parset 2</li>
     <li>parset 3</li>
     <li>parset other?</li>
     <li>fill mobility for paths</li>
   </ol>