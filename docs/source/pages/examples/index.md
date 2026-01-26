# Examples

## Testmap generation

Creating testmap with default settings

```shell
moma generate_test_map
```

To create rasters that can be used by moma the `--create_rasters` argument is used

```shell
moma generate_test_map --create_rasters
```

---

The different options for testmap generation can be mixed and matched to create maps with different features

```shell
moma generate_test_map --create_rasters --seed 9 --max_slope_percent 10 --map_dim 80 100 --plateaus 2 3 --slope_length 40 20
```



## Mobility map generation

Basic call where the console is cleared before run

```shell
moma --clear mobility
```

---

Vehicle and site specified and plot show as popups

```shell
moma mobility --show_plots --vehicle mock --site testmap
```

---

Destitution used for sampling values and 90% quantile is used for reducing tio single mobility value

```shell
moma mobility --n_samples 50 --sample_reduction quantile 0.9
```

---

Specifying target and minimum speed

```shell
moma mobility --directions --target_speed 15 --minimum_speed 1
```

---

Specifying directions

```shell
moma mobility --directions N omni SE
```

```shell
moma mobility --n_directions 12
```

---

Finding optimal path between points

```shell
moma mobility --path_coordinate_system utm --path_start 10 210 --path_end 140 40
```

---

Adding mobility calculations for paths in specified folder

```shell
moma mobility --path_folder src/moma/data/examples/sites/testmap/paths
```

---


Combined call with use of the different arguments

```shell
moma mobility --vehicle mock --site testmap --n_samples 50 --sample_reduction mean --directions N NE E SE S SW W NW omni --path_coordinate_system utm --path_start 10 200 
   --path_end 240 50 --target_speed 20
```


# Mobility file generation

```shell
moma generate_vehicle_mobility_file --vehicle Olav --minimum_speed 0.1 --plot
```

