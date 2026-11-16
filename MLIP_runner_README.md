# MLIP_runner.py

**Role in the workflow:** the script you actually run. It holds the settings, calls the parser to load everything, then calls the processor functions in order for each material.

```
python MLIP_runner.py
```

```
MLIP_runner.py
   ├── imports from MLIP_parser.py     (folders, MODELS, cif_parser, reference_data_parser)
   └── imports from MLIP_processor.py  (relaxation, force constants, bands, CSV, plots)
```
## 1. What the file needs

- `MLIP_parser.py` and `MLIP_processor.py` in the same folder.
- `materials_cif/` with a CIF for every name in `MATERIALS`.
- Optional: `phononDB/` and/or `reference_data/` for the reference comparison.
- The conda environment for the chosen `MODEL` active (`mace-torch` for MACE, `prophet-mlip` for Prophet). The two models need separate environments; the install commands are in the top-level `README.md`.
- Standard library `time` and `datetime`, plus `numpy`.

## 2. Settings

Change these at the top of the file before each run.

| Setting | Current value | Meaning | Passed to |
|---|---|---|---|
| `MODEL` | `"Prophet-OAME-MBD"` | Which MLIP to use. Must be a key of `MODELS` in the parser file, and must match the active conda environment | `MODELS[MODEL](DEVICE)`, `get_run_folder`, CSV |
| `DEVICE` | `"cpu"` | `"cpu"` or `"cuda"` | the `make_...` function |
| `SUPERCELL` | `2` | Supercell size N for an N×N×N supercell | `force_constant_calculator`, folder name, plot |
| `DISPLACEMENT` | `0.03` | Finite displacement in Å | `force_constant_calculator`, folder name, plot |
| `RUN_CALCULATIONS` | `True` | Run relaxation + phonons with `MODEL` | switches section 4 on/off |
| `RUN_PLOTS` | `True` | Make the comparison figures | switches section 5 on/off |
| `IMAGINARY_LIMIT` | `-0.05` THz | A lowest frequency below this counts as an imaginary mode (Batatia et al. 2025) | compared with `lowest_band_frequency` |
| `MATERIALS` | `["Si", "GaAs", "MgO"]` | Which CIFs to run. `None` runs every CIF in the folder | filters the `cif_parser` result |
| `REFERENCE_MP_IDS` | `{}` | Manual phononDB ids, e.g. `{"Si": "mp-149"}`, for materials that cannot be matched automatically | `reference_data_parser` |

**Useful combinations:**
- `RUN_CALCULATIONS = True`, `RUN_PLOTS = True`: normal run.
- `RUN_CALCULATIONS = False`, `RUN_PLOTS = True`: redraw the figures from saved results without loading a model. Fast, and does not need the MLIP installed.
- Run once per `MODEL`. Each run adds that model to the results, and the plots pick up every model that has been run so far.

## 3. Parse section (always runs)

### Step 1: load the structures

```python
materials = cif_parser(MATERIALS_CIF_FOLDER)
```

- Reads every CIF in `materials_cif/`.
- If `MATERIALS` is not `None`: checks every name has a CIF (otherwise `ValueError` listing what is available), then keeps only those materials, in the order of the list.
- Prints the material names.

**Relies on:** `cif_parser` (parser file), CIF files.

### Step 2: load the reference data

```python
references, reference_sources = reference_data_parser(
    materials, PHONON_DB_FOLDER, REFERENCE_PHONOPY_PARAMS_FOLDER, mp_ids=REFERENCE_MP_IDS
)
```

- For each material: phononDB first, then `reference_data/<material>_phonopy_params.yaml`, then nothing.
- `references` only contains materials that have reference data.
- `reference_sources` says where each one came from. It is printed by the parser but not used again in the runner.

**Relies on:** `reference_data_parser` (parser file), the Kyoto phononDB folder and lookup table, the `reference_data/` folder.

**For the current materials:** Si and MgO are in phononDB (mp-149 and mp-1265 by automatic matching, if your CIFs are the diamond and rock-salt structures). **GaAs is not in phononDB**, so it needs `reference_data/GaAs_phonopy_params.yaml` or it will run with no reference.

Both steps run even when `RUN_CALCULATIONS = False`, because the plots need the structures (for the band path) and the references.

## 4. Calculations section (`if RUN_CALCULATIONS:`)

### Step 3: load the model (once)

```python
calc = MODELS[MODEL](DEVICE)
```

The model is loaded once and reused for every material. The load is timed with `start_timer` and `performance_log` (material name `"all"`, stage `"model load"`), and the wall time is stored as `load_s`. Because it is shared, the same `load_s` value appears in every row of that run.

**Relies on:** `MODELS` and the `make_...` functions (parser file); `start_timer`, `performance_log` (processor file).

### Step 4: loop over the materials

Each material is run inside a `try / except`. A variable called `stage` is updated before each step, so if something fails the error message says where.

| Order | `stage` value | What happens | Processor function | Values produced |
|---|---|---|---|---|
| 1 | `"set up"` | Build and create the run folder; read the CIF space group; reduce to the primitive cell | `get_run_folder`, `get_primitive_cell` | `folder`, `space_group_before`, `atoms` |
| 2 | `"relaxation"` | Relax positions and cell; measure leftover force and volume | `relax_structure` | `converged`, `steps`, `relax_time`, `max_force`, `volume_per_atom` |
| 3 | `"symmetry check"` | Find the space group of the relaxed cell and compare with the CIF | `get_space_group` | `space_group_after`, `symmetry_kept` |
| 4 | `"force constants"` | Displacements, forces, force constants | `force_constant_calculator` | `phonon`, `forces_time` |
| 5 | `"band structure"` | Band path, bands, save, lowest frequency, Γ optical frequency | `get_band_path`, `band_calculator`, `save_bands`, `lowest_band_frequency`, `gamma_optical_frequency` | `bands`, `lowest_frequency`, `gamma_optical`, `imaginary_modes`, `bands_time` |
| 6 | `"reference comparison"` | Γ optical frequency of the reference and the percentage difference | `gamma_optical_frequency` | `ref_gamma_optical`, `gamma_diff` |
| 7 | `"saving results"` | Print a summary and add a row to `summary.csv` | `results_table` | one CSV row |

**Timing.** The relaxation, force constants and band structure stages are each wrapped like this:

```python
timer = start_timer()
...
relax_time = performance_log(material_name, MODEL, DEVICE, SUPERCELL, DISPLACEMENT, stage, timer)
```

Each call adds one row to `results/performance_log.csv` and returns the wall time, which goes into `summary.csv` as `relax_s`, `forces_s` or `bands_s`.

**Details worth knowing for each step:**

1. **Set up.** `space_group_before` comes from `materials[material_name].info`, which `cif_parser` filled in. `get_primitive_cell` returns a new `Atoms` object, so the original CIF structure in `materials` is left untouched.

2. **Relaxation.** `relax_structure` changes `atoms` in place, so from here on `atoms` is the relaxed structure with the calculator attached.
   - `max_force` is the largest single force **component** left on any atom (`np.abs(forces).max()`), in eV/Å.
   - `volume_per_atom` is useful for spotting the over- or under-binding of a model.

3. **Symmetry check.** Compares the space group **symbols** as text. If they differ, a warning is printed but the run carries on. `space_group_number` is returned but not used.

4. **Force constants.** Uses the relaxed `atoms`, so the phonons are calculated at this model's own equilibrium geometry.

5. **Band structure.** The band path is found from the **relaxed** cell here. (The plot section uses the **unrelaxed** CIF cell; see section 5.) The time for this stage covers all five function calls.
   - `imaginary_modes = lowest_frequency < IMAGINARY_LIMIT`.

6. **Reference comparison.** Only if the material is in `references`:

   ```
   gamma_diff = (gamma_optical - ref_gamma_optical) / ref_gamma_optical * 100
   ```

   Negative means the MLIP is softer than the reference (the known softening of uMLIPs). If there is no reference, both values are left as empty strings and the CSV cells are blank.

7. **Saving results.** The list passed to `results_table` must be in the same order as `CSV_COLUMNS` in the processor file. If you add a column in one place, add it in the other.

### If a material crashes

```python
except Exception as error:
    status = f"CRASHED in {stage}: {type(error).__name__}: {error}"
```

- The message is printed and a short row (date, material, model, supercell, displacement, status) is written to `summary.csv`.
- The loop then moves on to the next material. One failure does not stop the run.
- Files written before the crash (for example `relaxed.cif`) stay in the run folder.
- The stage that crashed gets no row in `performance_log.csv`; stages that finished before it keep theirs.

---

## 5. Plots section (`if RUN_PLOTS:`)

One figure per material, in its own `try / except` so a plotting problem with one material does not stop the others.

**Steps for each material:**

1. `phonon_loader(material_name, supercell, displacement)` loads the saved `phonopy_params.yaml` of **every model** that has results at this supercell and displacement. If there are none, it prints "no saved results to plot" and moves on.
2. Picks up the reference `Phonopy` object if the material has one, otherwise `None`.
3. Builds the band path from the primitive cell of the **CIF structure**: `get_band_path(get_primitive_cell(materials[material_name]))`. This cell is the same for every model, so it gives one shared path.
4. `plot_band_comparison(...)` draws and saves the figure.
5. Prints which models were plotted and where the figure was saved.

**Relies on:**
- `phonon_loader`, `get_primitive_cell`, `get_band_path`, `plot_band_comparison` (processor file).
- `materials` and `references` from the parse section.
- Results saved by this run or earlier runs.

## 6. What a run produces

**Printed to the terminal:**
- the materials list and the reference source of each;
- model load time;
- for each material: atoms in the primitive cell, number of displaced supercells, convergence, space group before and after, volume per atom, Γ optical frequency, difference from the reference, lowest frequency, stage times;
- where each figure was saved.

**Written to disk:**

```
results/
├── summary.csv                                   one row per material per run (appended)
├── performance_log.csv                           one row per stage per run (appended)
└── <material>/
    ├── band_comparison_sc<N>_d<disp>.png          all models + reference
    └── <MODEL>/sc<N>_d<disp>/
        ├── relax.log, relaxed.traj, relaxed.cif   stage 1
        ├── phonopy_disp.yaml, phonopy_params.yaml stage 2
        └── band.yaml                              stage 3
```
## 7. Which function each step relies on

| Runner step | Function | File | Needs on disk |
|---|---|---|---|
| Load structures | `cif_parser` | parser | `materials_cif/*.cif` |
| Load references | `reference_data_parser` | parser | `phononDB/` (Kyoto database + lookup JSON) and/or `reference_data/*.yaml` |
| Load model | `MODELS[MODEL]` → `make_mace` / `make_prophet` | parser | MACE download cache / `weights/prophet-oame-mbd.pt` |
| Stage timing | `start_timer`, `performance_log` | processor | `results/performance_log.csv` |
| Run folder | `get_run_folder` | processor | |
| Primitive cell | `get_primitive_cell` | processor | |
| Relaxation | `relax_structure` | processor | run folder |
| Symmetry check | `get_space_group` | processor | |
| Force constants | `force_constant_calculator` | processor | run folder |
| Band path | `get_band_path` | processor | |
| Bands | `band_calculator`, `save_bands` | processor | run folder |
| Checks | `lowest_band_frequency`, `gamma_optical_frequency` | processor | |
| CSV row | `results_table` | processor | |
| Load saved results | `phonon_loader` | processor | `results/<material>/<model>/.../phonopy_params.yaml` |
| Figure | `plot_band_comparison` | processor | `results/<material>/` |

## 8. Performance log

Four stages are timed: the model load (once per run, material name `"all"`), and the relaxation, force constants and band structure of each material. Each one adds a row to `results/performance_log.csv` with the wall time, CPU time, cores used, cores available and CPU usage. The functions are described in section 9 of `MLIP_processor_README.md`.

`summary.csv` keeps its four time columns (`load_s`, `relax_s`, `forces_s`, `bands_s`); they are the wall times returned by `performance_log`.
