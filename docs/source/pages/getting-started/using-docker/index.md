# Using Docker

## Pulling the image

The prebuilt image is available on the GitHub container registry. You can pull
the image with a tag matching your desired application, or you can build the
Docker image locally.

There are two different Docker image variants:

* `devel`, suitable for development, The MOMA environment must be activated
  manually inside the container.
* `runtime`, suitable for deployment, The MOMA environment is activated and its
  executable is available in PATH.

Moreover, all images are available in their default and CUDA (labelled with the
trailing `-cuda`) variants. The latter are much larger, but add support for
GPU-accelerated map generation.

* The latest built version is labelled as `-stable`.
* Specific versions are labelled based on to the commits they are build against
  (`<hash>`).

To pull a specific Docker image, use the following command:

```shell
docker pull gchr.io/aarhus-robotics/moma:{tag}}
```

Below is a list of the available tag and a brief description of each.

:::{note}
If you are an end-user looking to run MOMA, the `runtime` image is suitable for
your application.
:::


| Tag | Description |
| --- | --- |
| `devel`, `devel-<hash>`, `devel-stable` | Development image |
| `devel-cuda`, `devel-<hash>-cuda`, `devel-stable-cuda` | Development image, with CUDA support |
| `runtime`, `runtime-<hash>`, `runtime-stable` | Runtime image with latest stable version |
| `runtime-cuda`, `runtime-cuda`, `runtime-<hash>-cuda`, `runtime-stable-cuda` | Runtime image with latest stable version, with CUDA support |

## Running the image

You can spawn a new interactive shell with `moma` in a container by running:

```shell
docker run -it ghcr.io/aarhus-robotics/moma:runtime
```

You may then use the command `moma` to start generating your mobility maps. For
example, to generate a testmap:

```shell
moma generate_test_map --create_rasters
```

Refer to the [user guide](../../user-guide/index.md#user-guide) for more
details on the command line interface options and arguments.