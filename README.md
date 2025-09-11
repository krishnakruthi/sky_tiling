# Ranked-Tiling & Scheduling for GW Follow-Up

## Introduction

Follow-up of gravitational wave (GW) sky-localization regions using large FOV telescopes requires covering the localization interval with minimal pointings. Each pointing leaves a **tile** (footprint) on the sky.

The **ranked-tiling strategy** (arXiv:1511.02673) works as follows:

1. Compute probability contained in each tile of a predefined sky-grid.  
2. Rank tiles by probability.  
3. Select top `N` tiles covering 90% of the GW localization probability → these are the **ranked-tiles**.  

For telescopes with predetermined grids, the tile centers are fixed. For a given sky-map resolution, the pixels inside each tile are known in advance, allowing rapid computation via a **tile-pixel map**.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/krishnakruthi/sky_tiling.git
````

2. Install as a module:

```bash
python setup.py install
# or
pip install -e .
```

Now `sky_tiling` can be imported as a Python module.

---

## Setting Up a New Telescope

Create a `config.ini` file using `write_config_file.py` for the telescope you want to use. This will also generate tile-pixel maps that enable fast ranked-tile generation if not already available when you download this repo.

### Arguments

* `--work` : Working directory where the config file will be created.
* `--telescope` : Telescope name. Standard telescopes: **ATLAS, BlackGEM, PS1, ZTF**. Using these with `--nside=256` uses predefined tile-center files.
* `--nside` : Sky-map resolution (powers of 2, 64–2048). Default: 256. Non-standard telescopes or other nsides require generating a tile-pixel map.
* `--fov` : Field-of-view (required for non-standard telescopes without a tile-center file; only square FOVs supported).
* `--tilefile` : Full path to tile center file (required in all cases).
* `--site` : Observatory site. Required by the scheduler, optional for ranked-tiles.
* `--timemag` : Telescope integration time vs limiting magnitude file. Can be `None` if only generating ranked-tiles.
* `--extension` : Plot file extension (png, pdf). Required for plotting; can be a dummy value if not plotting.

### Example Commands

```bash
python write_config_file.py --work ~/RunDir/sky_tile_work --telescope BlackGEM --site None --timemag None --extension png
python write_config_file.py --work ~/RunDir/sky_tile_work --telescope BlackGEM --site lasilla --timemag None --extension pdf
```

> **Note:** Currently, the script must be run from the repository itself. It generates the config file and outputs environment variables to export.

---

## Tile-Pixel Maps

Tile-pixel maps enable fast ranked-tile generation. Precomputed maps for **ATLAS, BlackGEM, PS1, ZTF** at `nside=256` are included. For non-standard telescopes or higher resolutions, generate your own.

### Options for Non-Standard Telescopes

**Option 1: Provide a tile center file**

```
ID        ra_center   dec_center
0         25.00       -89.9
1         51.43       -88.36
2         102.86      -88.36
3         154.29      -88.36
4         205.71      -88.36
```

Command:

```bash
python write_config_file.py --work ~/RunDir/sky_tile_work --telescope <telescope> --site <site or None> --timemag <file or None> --tilefile <full path> --extension png
```

**Option 2: Generate a tile-center file using FOV**

```bash
python write_config_file.py --work ~/RunDir/sky_tile_work --telescope <telescope> --site <site or None> --timemag <file or None> --fov <FOV> --extension png
```

> Both options create the tile-pixel map and save paths in the config file. Construction time depends on `--nside` and FOV. Default nside is 256.

---

## Running Ranked-Tiling Codes

Create a ranked-tile object:

```python
from rankedTilesGenerator import RankedTileGenerator

tileObj = RankedTileGenerator('bayestar.fits.gz', 'config.ini')
[ranked_tile_indices, ranked_tile_probs] = tileObj.getRankedTiles(resolution=256)
```

> Higher-resolution ranked tiles require corresponding tile-pixel maps.

### Plotting Ranked Tiles

```python
rankedTileData = tileObj.plotTiles(
    ranked_tile_indices, ranked_tile_probs,
    CI=0.90, FOV=47.3, tileEdges=True,
    save=True
)
```

* `tileEdges=False` plots only tile centers
* `FOV` is optional for boundaries
* `save` controls saving the plot
* `CI` specifies the confidence interval

Example output:

| Rank | index | RA        | Dec       | Probability |
| ---- | ----- | --------- | --------- | ----------- |
| 1    | 691   | 24.71429  | -85.93846 | 0.1228203   |
| 2    | 733   | 76.14286  | -85.93846 | 0.1129290   |
| 3    | 644   | 127.57143 | -85.93846 | 0.1043966   |

---

## Other Methods

1. **searchedArea**: Computes total searched area and probability for a given injection position
2. **sourceTile**: Finds the tile containing the source
3. **detectability**: Computes detectability based on source tile rank, integration time, total observation time, distance, and telescope limiting magnitude vs time

