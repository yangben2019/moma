# Examples

## Testmap

### Generating the map rasters

### Generating mobility maps

Basic call where the console is cleared before run

```shell
moma  --clear mobility
```

Vehicle and site specified and plot show as popups

```shell
moma  mobility --show_plots --vehicle mock --site testmap
```

Destitution used for sampling values and 90% quantile is used for reducing tio single mobility value

```shell
moma  mobility --n_samples 50 --sample_reduction quantile 0.9
```

Specifying directions

```shell
moma  mobility --directions N omni SE
```

```shell
moma  mobility --n_directions 12
```

Finding path between points

```shell
moma mobility --path_coordinate_system utm --path_start 10 95 --path_end 255 95
```

Adding mobility calculations for paths in specified folder

```shell
moma  mobility --path_folder src/moma/data/examples/sites/testmap/paths
```


Combined call with use of the different arguments

```shell
moma  --clear mobility --vehicle mock --site testmap --n_samples 50 --sample_reduction mean --directions N NE E SE S SW W NW omni --path_coordinate_system utm --path_start 10 95 --path_end 255 95
```

### Generating straight paths

### Generating optimal paths
