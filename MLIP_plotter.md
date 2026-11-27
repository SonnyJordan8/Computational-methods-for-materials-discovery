# MLIP_plotter.py

**Role in the workflow:** draws one comparison figure of phonon band structures from runs that have already been calculated, with or without a DFT reference. It reruns no calculations: each run is rebuilt from the `phonopy_disp.yaml` and `FORCE_SETS` saved in its folder.

It is a script, run on its own after `MLIP_runner.py`. It imports from `MLIP_config.py`, `MLIP_parser.py` and `MLIP_processor.py`. Nothing imports it.

```
mlip_plotter_input.txt ──► read_input_plot_file ──► MATERIAL, TITLE, FIGURE_NAME, REFERENCE, RUNS
                                                              │
materials_cif/<MATERIAL>.cif ──► cif_parser ──► band path     │
results/<MATERIAL>/<model>/<run>/ ──► load_run_phonon ──► one Phonopy object per RUN
REFERENCE ──► requested_reference ──► reference_data_parser ──► reference Phonopy object
                                                              │
                              run_styles ──► plot_bands ──► results/<MATERIAL>/comparison_<FIGURE_NAME>.png
```

---

## 1. How to use it

1. Run the calculations first with `MLIP_runner.py`, so the run folders exist.
2. Write a plot file (see section 2), by default `mlip_plotter_input.txt` in the code folder.
3. Run:
   ```
   python MLIP_plotter.py
   python MLIP_plotter.py --config my_settings.txt     # if the folders are set in another settings file
   ```
4. The figure is saved to `<RESULTS_FOLDER>/<MATERIAL>/comparison_<FIGURE_NAME>.png`, and its path is printed.

To make a different figure, edit the plot file, or change `INPUT_PLOT_FILE` at the top of the script to point to another plot file.

---

## 2. The plot file

Same layout as the settings file: one `KEY = value` per line, `#` starts a comment, keys are not case-sensitive.

| Key | Required | Repeatable | Meaning | If left out |
|---|---|---|---|---|
| `MATERIAL` | yes | no | Material name, the same as its CIF file name without `.cif` | error |
| `RUN` | yes, at least one | **yes** | One run to plot, as `<model>/<run folder>` inside `results/<MATERIAL>/` | error |
| `TITLE` | no | no | Figure title | the material name |
| `FIGURE_NAME` | no | no | Name of the saved figure (a trailing `.png` is removed) | the plot file's name without its extension |
| `REFERENCE` | no | no | Which reference to draw (see below) | `AUTO` |

If `TITLE`, `FIGURE_NAME` or `REFERENCE` is written more than once, the last one is used (unlike the settings file, this is not an error).

### `REFERENCE` values

| Value | Meaning |
|---|---|
| `AUTO` | Match the reference automatically from the CIF's formula and space group (same as the runner does with no entry in `REFERENCES`) |
| `NONE` | Draw no reference |
| `<mp-id>` | Use this Materials Project ID, e.g. `mp-149` |
| `<source> <mp-id>` | Use this ID from a named source, e.g. `MP mp-19907` |

The last two use the same format as one entry of `REFERENCES` in the settings file, without the material name.

### Example

```
# Si: MACE vs SevenNet, two supercells for MACE
MATERIAL    = Si
TITLE       = Si phonons: MACE vs SevenNet-Omni
FIGURE_NAME = Si_mace_vs_sevennet
REFERENCE   = AUTO
RUN = MACE-MP-0-medium/<run folder A>
RUN = MACE-MP-0-medium/<run folder B>
RUN = SevenNet-Omni-i12/<run folder C>
```

Use the run folder names exactly as they appear in `results/Si/<model>/`. Backslashes are turned into `/` and leading or trailing slashes are removed, so a path copied from Windows Explorer also works.

---

## 3. What the file needs

### From the other workflow scripts

| Name | From | Used for |
|---|---|---|
| `CODE_FOLDER` | `MLIP_config` | Finding the plot file |
| `MATERIALS_CIF_FOLDER`, `PHONON_DB_FOLDER`, `MP_REFERENCE_FOLDER` | `MLIP_parser` | Where the CIFs and reference data are |
| `MODELS` | `MLIP_parser` | Each model's line colour (`MODELS[model]["colour"]`) |
| `cif_parser`, `reference_data_parser` | `MLIP_parser` | Reading the structure and the reference |
| `RESULTS_FOLDER` | `MLIP_processor` | Where the runs are and where the figure goes |
| `load_run_phonon` | `MLIP_processor` | Rebuilding each run, with the settings it was calculated with |
| `get_primitive_cell`, `get_band_path`, `plot_bands` | `MLIP_processor` | The band path and the figure |

Because it imports `MLIP_config`, the plotter needs a **complete, valid settings file**, even though it only uses the folder settings.

### On disk

- `materials_cif/<MATERIAL>.cif`, for the band path.
- For each `RUN`: `results/<MATERIAL>/<model>/<run folder>/` containing `phonopy_disp.yaml` and `FORCE_SETS` (and the run's `run_config.txt`).
- The reference data, unless `REFERENCE = NONE`.

### Settings at the top of the script

| Name | Meaning |
|---|---|
| `INPUT_PLOT_FILE` | The plot file, relative to the code folder (default `mlip_plotter_input.txt`) |
| `LINE_STYLES` | Line styles given in turn to the runs of one model: `-`, `-.`, `--`, `:` |

---

## 4. Functions

### `read_input_plot_file(path)`

**What it does:** reads a plot file and returns a dictionary with `MATERIAL`, `TITLE`, `FIGURE_NAME`, `REFERENCE` and `RUNS` (a list, in the order written).

**Raises:** `FileNotFoundError` if the file doesn't exist; `ValueError` (with line number) for a line without `=` or an unknown key, and if `MATERIAL` or every `RUN` is missing.

### `requested_reference(material_name, reference)`

**What it does:** turns the plot file's `REFERENCE` into what `reference_data_parser` takes as `requested`:

| `REFERENCE` | Returns |
|---|---|
| `AUTO` | `{}` (match automatically) |
| `NONE` | `None` (no reference) |
| `mp-149` | `{"Si": "mp-149"}` |
| `MP mp-149` | `{"Si": ("MP", "mp-149")}` |

**Raises:** `ValueError` for three or more words.

### `run_styles(runs)`

**What it does:** gives each run a colour and line style.
- **Colour:** the model's colour from `MODELS` in `MLIP_parser.py`. A model folder that isn't in `MODELS` is drawn in grey, with a printed note.
- **Line style:** the runs of the same model get `-`, `-.`, `--`, `:` in the order they are listed in the plot file. A fifth run of the same model starts again at `-`.

**Returns:** `{run: {"color": ..., "linestyle": ...}}`.

### `make_comparison_plot(plot_file)`

**What it does:** makes the figure described by one plot file.

**Steps:**
1. Reads the plot file.
2. Reads **every** CIF in the CIF folder and picks out `MATERIAL`; builds the band path from its primitive cell.
3. Rebuilds every `RUN` with `load_run_phonon`. If any can't be found, it stops and lists all the missing ones.
4. Unless `REFERENCE = NONE`, loads the reference with `reference_data_parser`, and its label (where it came from) for the legend.
5. Calls `plot_bands` and returns the path of the saved figure.

**Raises:** `ValueError` if there is no CIF for the material, or if any run folder is missing its saved files.

---

## 5. Good to know

- **Bug: `REFERENCE = NONE` currently crashes.** In `make_comparison_plot`, the line that sets `figure_file` is indented inside `if requested is not None:`, so with no reference it is never set and the `return plot_bands(...)` line fails with `UnboundLocalError`. Moving that line out of the `if` block (to the same indentation as `return`) fixes it.
- **The runs are rebuilt with their own saved settings.** `load_run_phonon` reads each run's `run_config.txt`, so runs with different supercells or symmetry settings can be plotted together correctly. Settings missing from an old run's copy take the current config's value, with a printed note.
- **The band path comes from the CIF, not from the runs.** All runs and the reference are drawn along the same path, built from the unrelaxed CIF's primitive cell.
- **Every CIF in the folder is read**, not just `MATERIAL`, so a broken CIF for another material still stops the plot.
- **The reference is PBEsol if it comes from phononDB.** The legend label (from `sources`) says where it came from.
- **A figure with the same `FIGURE_NAME` is overwritten** without warning.
- **Line 9** has a leftover comment (`# the figure title`) after `INPUT_PLOT_FILE`. It is harmless.

---

## 6. Summary table

| Function | Called by | Relies on | Returns |
|---|---|---|---|
| `read_input_plot_file` | `make_comparison_plot` | the plot file | plot dictionary |
| `requested_reference` | `make_comparison_plot` | `REFERENCE` from the plot file | `{}`, `None` or `{material: ...}` |
| `run_styles` | `make_comparison_plot` | `MODELS` colours, `LINE_STYLES` | colour and line style per run |
| `make_comparison_plot` | the script, at the bottom | everything above, plus `cif_parser`, `load_run_phonon`, `reference_data_parser`, `plot_bands` | path of the saved figure |
