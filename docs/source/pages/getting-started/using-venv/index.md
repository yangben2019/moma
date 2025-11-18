# Using virtual environment

MOMA can be installed using [Poetry](https://python-poetry.org/), a popular
Python dependency management and packaging tool. With Poetry installed on your
system, run the following command from the root of the MOMA repository:

```shell
poetry lock && poetry install && $(poetry env activate)
```