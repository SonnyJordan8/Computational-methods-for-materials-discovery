# uMLIP phonon benchmarking workflow

Models set up so far: **MACE-MP-0 medium**, **SevenNet-Omni-i12** and **Prophet-OAME-MBD**.

```
config_file.txt ─► MLIP_config.py (settings for every script)

materials_cif/*.cif ─► primitive cell ─► relax (MLIP) ─► displacements + forces ─► force constants ─► bands, properties
                                                                                                         │
reference_data/ (Materials Project, phononDB) ─► DFT reference phonons ──────────────────────────────────┤
                                                                                                         ▼
                              results/summary.csv, performance_log.csv, <run folder>/band_structure.png
                                                                                                         │
                                       MLIP_plotter.py + plot file ─► results/<material>/comparison_<plot file>.png
```

This file covers installation, downloads and running. Each script has its own README explaining what every function does:

| File | Role | Details |
|---|---|---|
| `config_file.txt` | **All the settings**, phonopy-style `KEY = value`. The only file edited between runs | section 6 |
| `MLIP_config.py` | Reads and checks the config file; the overwrite check | section 6 |
| `MLIP_runner.py` | The script you run for calculations | `MLIP_runner_README.md` |
| `MLIP_parser.py` | Reads CIFs, finds and loads reference data, builds the MLIP calculators (`MODELS`) | `MLIP_parser_README.md` |
| `MLIP_processor.py` | Run folders, relaxation, force constants, bands, properties, CSV files, plots | `MLIP_processor_README.md` |
| `MLIP_plotter.py` | Comparison plot of several saved runs of one material | section 7 |
| `MaterialsProject_extractor.ipynb` | Downloads Materials Project phonon references | section 5.3 |
| `Kyoto_database_extractor.ipynb` | One-off set-up of the phononDB reference database | `KYOTO_database_extractor_README.md` |

## 1. Quick start

1. Install conda and git (section 3).
2. Create the environment for the model you want to run (section 4).
3. Put the files in place (section 5): CIFs, model weights, and optionally reference data.
4. Edit `config_file.txt`: at least `MATERIALS`, `MODEL`, `DEVICE` (section 6).
5. Activate the matching environment and run `python MLIP_runner.py` from the project folder.
6. To compare models: change `MODEL`, switch environment, run again; then list the runs in a plot file and run `MLIP_plotter.py` (section 7).

## 2. Folder layout

Everything sits next to the scripts. Folders marked **you add** are not created by the code. Folder names can be changed in the config (`MATERIALS_CIF_FOLDER`, `PHONON_DB_FOLDER`, `MP_REFERENCE_FOLDER`, `RESULTS_FOLDER`).

```
project folder/
├── config_file.txt
├── MLIP_config.py, MLIP_runner.py, MLIP_parser.py, MLIP_processor.py, MLIP_plotter.py
├── MaterialsProject_extractor.ipynb, Kyoto_database_extractor.ipynb
├── <plot files>.txt                         you add      one per comparison plot (section 7)
├── materials_cif/                           you add      one CIF per material
│   └── Si.cif ...
├── weights/                                 you add      only needed for Prophet
│   └── prophet-oame-mbd.pt
├── reference_data/                          you add      optional
│   ├── materials_project_reference_data/    from MaterialsProject_extractor.ipynb (section 5.3)
│   │   ├── phonopy_settings/<material>_phonopy_settings.json
│   │   ├── force_constants/<material>_force_constants
│   │   ├── band_structure/, dos/            downloaded, not used by the workflow yet
│   │   └── ...
│   └── phononDB/                            Kyoto phononDB (section 5.4)
│       ├── phonon_db/mp-149-20180417/ ...
│       └── phonon_db_lookup_df.json
└── results/                                 created by the runner
```

| Folder | Needed for | What happens if it is missing |
|---|---|---|
| `materials_cif/` | every run | `FileNotFoundError: No CIF files found` |
| `weights/prophet-oame-mbd.pt` | `MODEL = Prophet-OAME-MBD` | `FileNotFoundError` giving the download link |
| `reference_data/materials_project_reference_data/` | MP references | that source is skipped |
| `reference_data/phononDB/` | phononDB references | a message is printed and that source is skipped |
| `results/` | created automatically | |

The CIF file name (without `.cif`) is the material name used everywhere: in `MATERIALS`, `REFERENCES`, the results folders and plot files.

## 3. Before you start

- **conda** (Miniforge or Anaconda) and **git**.
- **Internet access** for the installs, the MACE and SevenNet weights (downloaded automatically on first use), the Prophet weights, the Materials Project download (needs an MP API key) and the phononDB clone.
- **Disk space:** about 3 GB for phononDB, plus a few GB per conda environment.
- A GPU is not needed: all three models run with `DEVICE = cpu`.

## 4. Conda environments

**MACE and Prophet need separate environments** (`mace-torch` 0.3.17 needs `e3nn` 0.4.4; Prophet needs `e3nn` 0.6 or later). The code is the same in every environment; activate the one that matches `MODEL`.

| `MODEL` in the config | Environment | Model package |
|---|---|---|
| `MACE-MP-0-medium` | `mlip-mace` | `mace-torch==0.3.17` |
| `SevenNet-Omni-i12` | `mlip-prophet` (where it has been run so far) | `sevenn` |
| `Prophet-OAME-MBD` | `mlip-prophet` | `prophet-mlip` from GitHub, commit `dcfdc0b` |

Every environment also needs: `ase`, `phonopy`, `pymatgen`, `monty`, `pandas`, `numpy`, `matplotlib` (plus `h5py` for `HDF5 = .TRUE.`, `tqdm` and `ipykernel` for the notebooks, `symfc` or `alm` only if `FC_CALCULATOR` asks for them).

### Environment A: MACE

```bash
conda create -n mlip-mace -c conda-forge python=3.12 pip git -y
conda activate mlip-mace
pip install "mace-torch==0.3.17" ase phonopy pymatgen monty pandas matplotlib h5py tqdm ipykernel
```

### Environment B: Prophet (and SevenNet)

```bash
conda create -n mlip-prophet -c conda-forge python=3.12 pip git -y
conda activate mlip-prophet
pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git@dcfdc0b" ase phonopy pymatgen monty pandas matplotlib h5py tqdm ipykernel
pip install sevenn
```

> **Do not run `pip install prophet`.** That installs Facebook's time-series package, which has the same import name.

> **To check:** `pip install sevenn` is how SevenNet is usually installed, but the exact command and version used in `mlip-prophet` have not been recorded here yet. Fill this in with `pip show sevenn` from that environment, and check it does not change the `e3nn`/`torch` versions Prophet needs.

### Check the install

```bash
python -c "import mace, e3nn; print(e3nn.__version__)"                    # mlip-mace: 0.4.4
python -c "from prophet import KairosCalculator; import e3nn; print(e3nn.__version__)"   # mlip-prophet: 0.6.0 or later
python -c "from sevenn.calculator import SevenNetCalculator"              # mlip-prophet
```

## 5. What to download and where it goes

### 5.1 Structures: `materials_cif/`

One CIF per material; the file name is the material name. Every CIF in the folder is read at start-up, even ones not in `MATERIALS`, so a broken CIF stops the run. Automatic reference matching uses the formula and space group found from the CIF.

### 5.2 Model weights

| Model | Where | How |
|---|---|---|
| MACE-MP-0 medium | MACE's cache | Downloaded automatically on first use |
| SevenNet-Omni-i12 | sevenn's own files | Loaded by name (`7net-omni-i12`, modal `mpa`); downloaded on first use if not bundled |
| Prophet-OAME-MBD | `weights/prophet-oame-mbd.pt` next to `MLIP_parser.py` | By hand, once (below) |

```bash
mkdir weights
curl -L -o weights/prophet-oame-mbd.pt https://huggingface.co/kairosmaterial/prophet/resolve/f9a54df874ba52b8b35e7d3f7f348b46e0135563/prophet-oame-mbd.pt
```

### 5.3 Materials Project references (optional)

Run `MaterialsProject_extractor.ipynb` for each material. It writes into `reference_data/materials_project_reference_data/`; the workflow uses two of its files:
- `phonopy_settings/<material>_phonopy_settings.json`: MP's starting unit cell (`structure`), supercell matrix and primitive matrix;
- `force_constants/<material>_force_constants`: MP's force constants.

### 5.4 Kyoto phononDB references (optional)

The Kyoto (Togo) phonon database, 2018-04-17 release, packaged by the WMD group (10,034 materials, PBEsol). From `reference_data/`:

```bash
git clone https://github.com/WMD-group/phononDB.git phononDB
cd phononDB
bash extract_files_linux.sh        # extract_files_mac.sh on macOS
python create_lookup_json.py       # writes phonon_db_lookup_df.json
```

Build the lookup table on the machine where you run the workflow (a table built on Windows can hold Windows paths). If the project folder is synced (OneDrive etc.), consider excluding `phononDB/` (about 140,000 files).

### 5.5 Which reference is used

For each material:
1. If it is listed in `REFERENCES` (e.g. `Si mp-149`, or `Si MP mp-149` / `Si phononDB mp-149` to name the source), that entry is used.
2. Otherwise it is matched automatically: Materials Project folder first, then phononDB.
3. Otherwise there is no reference.

The choice is printed at start-up, e.g. `reference for Si: MP mp-149`.

**How the reference is built.** An MP reference is built from MP's **starting unit cell**, supercell and primitive matrix (from the settings JSON), with MP's **force constants** given to phonopy directly (no force sets are involved). A phononDB reference is built from its `phonon.yaml`, with the force constants **fitted from its `FORCE_SETS`**. Neither is relaxed with the MLIP or uses the config's phonopy settings, and phonopy calculates the reference bands in the reference's own primitive cell. The band path is made from the MLIP's (runner) or the CIF's (plotter) primitive cell, so a warning is printed if the reference's primitive cell looks different. Details: section 5 of `MLIP_runner_README.md`.

## 6. Running

### The config file

All settings are in `config_file.txt`, one per line, `KEY = value`; `#` starts a comment; true/false are `.TRUE.`/`.FALSE.`. Every setting must be present and spelled correctly (unknown or missing names stop the run with an error saying which), except `IRREPS` and `MAGMOM`, which may be left out. The file is read once, by `MLIP_config.py`, and the values are checked before anything runs.

| Group | Settings |
|---|---|
| Run | `MATERIALS` (names or `ALL`), `MODEL`, `REFERENCES`, `DEVICE`, `DTYPE` (`float64`/`float32`; used by MACE only), `RUN_CALCULATIONS`, `RUN_PLOTS_RUNS`, `IMAGINARY_LIMIT` |
| Relaxation (ASE) | `OPTIMIZER` (`FIRE`, `BFGS`, `LBFGS`, `BFGSLineSearch`), `RELAX_CELL`, `RELAX_FMAX`, `ENERGY_TOLERANCE`, `MAX_STEPS`, `RELAX_SYMMETRY` |
| phonopy pre-process | `DIM` (3 or 9 numbers), `PRIMITIVE_AXES` (`AUTO`, `P` or 9 numbers), `DISPLACEMENT_DISTANCE`, `PM`, `MAGMOM` (read, not used yet), `PHONOPY_LOG_LEVEL` |
| Symmetry and force constants | `SYMMETRY`, `SYMMETRY_TOLERANCE`, `FC_CALCULATOR` (`traditional`, `symfc`, `alm`), `FC_SYMMETRY` |
| phonopy post-process | `BAND_POINTS`, `MESH`, `TPROP`, `TMIN`, `TMAX`, `TSTEP`, `IRREPS`, `SHOW_IRREPS` |
| Output | `HDF5`, `WRITE_FORCE_CONSTANTS`, `RESULTS_FOLDER`, `DISPLACEMENTS_FILE`, `FORCE_SETS_FILE` |
| Input folders | `MATERIALS_CIF_FOLDER`, `PHONON_DB_FOLDER`, `MP_REFERENCE_FOLDER` |

The tags phonopy also uses (`DIM`, `PRIMITIVE_AXES`, `PM`, `MESH`, `TPROP`, ...) are checked against phonopy's own reader as well, so they mean the same as in a phonopy `.conf` file. Do not put `=` inside a value (e.g. `REFERENCES = Si mp=149`): phonopy's reader stops with "too many values to unpack".

### Run one model

```bash
conda activate mlip-mace
python MLIP_runner.py                             # uses config_file.txt
python MLIP_runner.py --config Si_sevennet.txt    # or another config file
```

**On Windows PowerShell**, if `conda activate` is not recognised, either run `conda init powershell` once (you may also need `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`) and reopen PowerShell, or call the environment's Python directly, e.g. `& "C:\Users\<you>\miniforge3\envs\mlip-mace\python.exe" MLIP_runner.py`. Running the script with the Run button of your editor also works, once the editor uses that environment's interpreter.

### Run folders and the overwrite check

Each run is saved in `results/<material>/<MODEL>/sc<supercell>_d<displacement>_sym-on` (or `_sym-off`), with `_relaxsym-off` and/or `_fcsym-off` added when those are off, e.g. `sc2x2x2_d0.03_sym-on_relaxsym-off_fcsym-off`. The config is copied into the folder as `run_config.txt`.

If the folder already holds a run calculated with different settings (e.g. another `PM` or `FC_CALCULATOR`), the material stops with "already holds a run with different settings (...)" and the old run is kept. Delete or rename that folder to rerun it. Rerunning with the same calculation settings reuses the folder.

### What a working run prints

```
settings from .../config_file.txt
materials: ['Si']
  reference for Si: MP mp-149
loaded SevenNet-Omni-i12 in ... s

===== Si | SevenNet-Omni-i12 =====
atoms in primitive cell: 2
Displaced supercells: ...
relaxation converged: True (... steps, max force ... eV/A, last energy change ... eV/atom)
space group: Fd-3m -> Fd-3m
...
  saved to results/Si/SevenNet-Omni-i12/sc2x2x2_d0.03_sym-on_relaxsym-off_fcsym-off
  band plot: .../band_structure.png
```

## 7. Comparison plots: `MLIP_plotter.py`

The runner only plots each run against its reference. To put several runs (models, supercells, settings) of **one material** on one figure:

1. Write a plot file, e.g. `Si_models.txt`, next to the scripts:

   ```
   MATERIAL = Si
   REFERENCE = AUTO          # AUTO (default), NONE, or the path of a reference file or folder
   RUN = SevenNet-Omni-i12/sc2x2x2_d0.03_sym-on
   RUN = MACE-MP-0-medium/sc2x2x2_d0.03_sym-on
   RUN = MACE-MP-0-medium/sc3x3x3_d0.03_sym-on
   ```

   Each `RUN` is `<model folder>/<run folder>` under `results/<material>/`.

2. At the top of `MLIP_plotter.py`, set `PLOT_FILE = "Si_models.txt"` and `TITLE = "..."`, then run it (`python MLIP_plotter.py`, or the Run button). There are no command-line options.

3. The figure is saved as `results/Si/comparison_Si_models.png`. Keep the plot file to redraw the same figure after rerunning the calculations.

- Each model has its colour from `MODELS` in `MLIP_parser.py` (MACE blue, SevenNet orange, Prophet green; an unknown model is grey, with a note). Several runs of one model are told apart by line style (solid, dashed, dotted, dash-dot) in the order listed. The reference is black.
- Each run is rebuilt from its `phonopy_disp.yaml` and `FORCE_SETS`, with the force constants refitted using the settings in its own `run_config.txt`, so changing the config later does not change old runs.
- `REFERENCE = AUTO` finds the reference as the runner does (`REFERENCES` in the config, then automatic matching). A path can be a Materials Project `<material>_phonopy_settings.json`, a phonopy yaml with `FORCE_SETS` or force constants next to it, or a folder holding one (e.g. a phononDB material folder or another run). MP band-structure or DOS files are refused, because they hold no force constants.
- The band path comes from the unrelaxed CIF primitive cell, so it is the same for every run.

## 8. Outputs

```
results/
├── summary.csv                                   one row per material per run (appended)
├── performance_log.csv                           one row per stage per run (appended)
└── <material>/
    ├── comparison_<plot file>.png                from MLIP_plotter.py
    └── <MODEL>/<run folder>/
        ├── run_config.txt                        settings used
        ├── relax.log, relaxed.traj, relaxed.cif  relaxation
        ├── phonopy_disp.yaml, FORCE_SETS         displacements and forces
        ├── force_constants.hdf5                  if WRITE_FORCE_CONSTANTS (FORCE_CONSTANTS if HDF5 off)
        ├── band.hdf5                             band structure (band.yaml if HDF5 off)
        ├── thermal_properties.yaml, irreps.yaml  if TPROP / IRREPS
        └── band_structure.png                    this run against its reference
```

- Both CSV files are **appended to**. Use `date_time` to tell rows apart. If the columns change in the code, rename or delete the old CSV first.
- `summary.csv` does not yet record the run folder, so runs that differ only in symmetry settings give rows that look the same (on the fix list).
- A material that fails gets a short row with `status = CRASHED in <stage>: <error>`, and the run carries on.

## 9. What to keep in mind when reading the results

- **The phononDB reference is PBEsol, not PBE**, with no van der Waals correction. Check the functional of each MP reference.
- **No non-analytical correction** on either side: polar materials show no LO/TO splitting, and the "Γ optical frequency" is the TO mode.
- The reference is used as published, in its own cell; the MLIP run is at the MLIP's own relaxed geometry. Watch for the cell-mismatch warning on non-cubic materials.
- Imaginary modes appear as negative frequencies. Only q-points on the band path are checked.
- `DTYPE` is applied to MACE only. SevenNet and Prophet use their own precision (a question for the supervisor).
- With `RELAX_SYMMETRY` on, ASE's `FixSymmetry` uses its own tolerance (0.01), not `SYMMETRY_TOLERANCE`, and `max_force` is the symmetry-averaged force.

## 10. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `Python was not found` / `conda is not recognised` (PowerShell) | conda not set up for PowerShell | `conda init powershell`, or call the environment's `python.exe` directly (section 6) |
| `config file not found` | Wrong `--config` path, or no `config_file.txt` next to the scripts | Check the path |
| `unknown setting X` / `config_file.txt is missing these settings: ...` | Typo, or a setting left out of the config | Fix the name or add the line |
| `phonopy's own reader could not read ...: too many values to unpack` | An `=` inside a value, e.g. `mp=149` | Use `mp-149` |
| `MODEL 'X' is not in MODELS` | Name not in `MODELS` in `MLIP_parser.py` | Use one of the names listed |
| `MACE / SevenNet / Prophet is not installed` | Wrong environment active for this `MODEL` | Activate the matching one |
| `... already holds a run with different settings (...)` | The run folder has a run calculated differently | Delete or rename that folder, or change the settings back |
| `supercell_matrix shape has to be (3,) or (3, 3)` | The supercell label was passed instead of the matrix | Pass `SUPERCELL_MATRIX` (`DIM`) |
| `No CIF files found` / `No CIF file in materials_cif for ['X']` | CIF folder missing, or no `X.cif` | Section 5.1 |
| `Prophet weights not found` | Weights not downloaded or misnamed | Section 5.2 |
| `cannot import name 'KairosCalculator' from 'prophet'` | Facebook's `prophet` installed | `pip uninstall prophet`, then section 4 |
| `<file> not found - run MaterialsProject_extractor.ipynb for X` | MP reference files missing | Section 5.3 |
| `phononDB lookup table not found` | phononDB not set up | Section 5.4, or ignore |
| `reference for X: none` | No match | Add `X mp-NNN` to `REFERENCES` |
| `WARNING X: the reference primitive cell does not match the CIF primitive cell` | Different cell shape or axis order | Check the band labels before comparing |
| `plot file not found` / `unknown setting` in the plotter | Wrong `PLOT_FILE`, or a line other than `MATERIAL`, `REFERENCE`, `RUN` | Fix `PLOT_FILE` or the plot file |
| `MeshSymmetryFallbackWarning` | Even `MESH` | Harmless; an odd mesh is faster |
| CSV rows do not line up with the header | Columns changed since the file was created | Rename or delete the old CSV |
