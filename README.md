# uMLIP phonon benchmarking workflow

Models set up so far: **MACE-MP-0 medium** and **Prophet-OAME-MBD**.

```
materials_cif/*.cif ─► primitive cell ─► relax (MLIP) ─► displacements + forces ─► force constants ─► band structure
                                                                                                        │
phononDB/ or reference_data/ ─► DFT reference phonons ──────────────────────────────────────────────────┤
                                                                                                        ▼
                                                             results/summary.csv, performance_log.csv, band_comparison_*.png
```

This file covers installation, downloads and running. Each script has its own README explaining what every function does:

| File | Role | Details |
|---|---|---|
| `MLIP_runner.py` | The script you run. Holds the settings and calls the other two | `MLIP_runner_README.md` |
| `MLIP_parser.py` | Reads CIFs, finds reference data, builds the MLIP calculators | `MLIP_parser_README.md` |
| `MLIP_processor.py` | Relaxation, force constants, bands, CSV files, plots | `MLIP_processor_README.md` |
| `Kyoto_database_extractor.ipynb` | One-off setup of the DFT reference database | `KYOTO_database_extractor_README.md` |

## 1. Quick start

1. Install conda and git (section 3).
2. Create the environment for the model you want to run (section 4).
3. Put the files in place (section 5): CIFs, Prophet weights, and optionally the reference database.
4. Set `MODEL`, `DEVICE` and `MATERIALS` at the top of `MLIP_runner.py` (section 6).
5. Activate the matching environment and run `python MLIP_runner.py`.
6. To compare models, switch environment, change `MODEL`, and run again. The plots pick up every model that has saved results.

## 2. Folder layout

Everything sits next to the three scripts. Folders marked **you add** are not created by the code.

```
project folder/
├── MLIP_runner.py
├── MLIP_parser.py
├── MLIP_processor.py
├── Kyoto_database_extractor.ipynb
├── materials_cif/                 you add      one CIF per material
│   ├── Si.cif
│   ├── GaAs.cif
│   └── MgO.cif
├── weights/                       you add      only needed for Prophet
│   └── prophet-oame-mbd.pt
├── phononDB/                      you add      optional, DFT reference database (section 5.3)
│   ├── phonon_db_tarred/          from the git clone
│   ├── phonon_db/                 from the extraction step
│   │   └── mp-149-20180417/ ...
│   └── phonon_db_lookup_df.json   from create_lookup_json.py
├── reference_data/                you add      optional, your own reference files
│   └── GaAs_phonopy_params.yaml
└── results/                       created by the runner
```

| Folder | Exact name the code expects | Needed for | What happens if it is missing |
|---|---|---|---|
| `materials_cif/` | `<material>.cif`, e.g. `Si.cif` | every run | `FileNotFoundError: No CIF files found` |
| `weights/` | `prophet-oame-mbd.pt` | `MODEL = "Prophet-OAME-MBD"` | `FileNotFoundError` giving the download link |
| `phononDB/` | `phonon_db_lookup_df.json` and `phonon_db/mp-XXXX-20180417/` | reference comparison | a message is printed and only `reference_data/` is used |
| `reference_data/` | `<material>_phonopy_params.yaml` | reference for materials not in phononDB | that material runs with no reference |
| `results/` | created automatically | | |

The CIF file name (without `.cif`) is the material name used everywhere: in `MATERIALS`, in the results folders and in the reference file names.

## 3. Before you start

- **conda** (Miniforge or Anaconda) and **git**.
- **Internet access** for the installs, the MACE weights (downloaded automatically on first use), the Prophet weights and the phononDB clone.
- **Disk space:** about 3 GB for phononDB (about 300 MB clone plus about 2.6 GB unpacked, about 140,000 small files), plus a few GB per conda environment.
- A GPU is not needed. Both models run on CPU with `DEVICE = "cpu"`.

## 4. Conda environments

**MACE and Prophet need separate environments.** `mace-torch` 0.3.17 requires `e3nn` exactly 0.4.4 and Prophet requires `e3nn` 0.6 or later, so they cannot be installed together. The workflow code is identical in both; you activate the environment that matches `MODEL`.

| `MODEL` in the runner | Environment | Model package | Key constraint |
|---|---|---|---|
| `"MACE-MP-0-medium"` | `mlip-mace` | `mace-torch==0.3.17` | `e3nn==0.4.4` |
| `"Prophet-OAME-MBD"` | `mlip-prophet` | `prophet-mlip` from GitHub, commit `dcfdc0b` | `e3nn>=0.6`, `torch>=2.7`, `ase>=3.29` |

### Packages every environment needs

| Package | Used by | For |
|---|---|---|
| `ase` | parser, processor | reading CIFs, relaxation (FIRE, FrechetCellFilter) |
| `phonopy` | parser, processor | displacements, force constants, band structure |
| `pymatgen` | parser, processor, lookup script | space group, primitive cell, band path |
| `monty` | parser, lookup script | loading and saving the phononDB lookup table |
| `pandas` | lookup table | the lookup table is a DataFrame |
| `numpy`, `matplotlib` | processor, runner | numbers and the comparison figure |
| `tqdm` | `create_lookup_json.py` | progress bars (phononDB setup only) |
| `ipykernel` | the notebook | running `Kyoto_database_extractor.ipynb` (phononDB setup only) |

`numpy`, `torch` and `e3nn` are installed automatically as dependencies of the model package.

### Environment A: MACE

```bash
conda create -n mlip-mace -c conda-forge python=3.12 pip git -y
conda activate mlip-mace
pip install "mace-torch==0.3.17" ase phonopy pymatgen monty pandas matplotlib tqdm ipykernel
```

### Environment B: Prophet

```bash
conda create -n mlip-prophet -c conda-forge python=3.12 pip git -y
conda activate mlip-prophet
pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git@dcfdc0b" ase phonopy pymatgen monty pandas matplotlib tqdm ipykernel
```

> **Do not run `pip install prophet`.** That installs Facebook's time-series forecasting package, which has the same import name. The MLIP is `prophet-mlip` and is only available from GitHub.

### Check the install

```bash
# in mlip-mace
python -c "import mace, e3nn, phonopy, pymatgen, ase; print(e3nn.__version__)"      # 0.4.4

# in mlip-prophet
python -c "from prophet import KairosCalculator; import e3nn; print(e3nn.__version__)"   # 0.6.0 or later
```


## 5. What to download and where it goes

### 5.1 Structures: `materials_cif/`

Create `materials_cif/` and add one CIF per material. The file name is the material name: `Si.cif`, `GaAs.cif`, `MgO.cif` for the default `MATERIALS` list.

- Every CIF in the folder is read at start-up, even ones not listed in `MATERIALS`, so a broken CIF stops the run.
- Automatic matching to phononDB uses the formula and space group found from the CIF. Use the standard structures (diamond Si, rock-salt MgO, zinc-blende GaAs).

### 5.2 Model weights

| Model | Where the weights go | How |
|---|---|---|
| MACE-MP-0 medium | MACE's own cache (`~/.cache/mace/`) | Downloaded automatically the first time the model is loaded. Nothing to do, but it needs internet that first time |
| Prophet-OAME-MBD | `weights/prophet-oame-mbd.pt` next to `MLIP_parser.py` | Download by hand, once |

**Prophet weights.** The folder must be called `weights` and the file `prophet-oame-mbd.pt`. From the project folder:

```bash
mkdir weights
curl -L -o weights/prophet-oame-mbd.pt https://huggingface.co/kairosmaterial/prophet/resolve/f9a54df874ba52b8b35e7d3f7f348b46e0135563/prophet-oame-mbd.pt
```

Or open the link in a browser and move the downloaded file into `weights/`. The link is pinned to one revision, the checkpoint behind the Matbench Discovery entry. Licence: CC BY 4.0.

### 5.3 DFT reference data: `phononDB/` (optional)

The Kyoto (Togo) phonon database, 2018-04-17 release, packaged by the WMD group. 10,034 materials, PBEsol. Without it the workflow still runs, but with no reference line unless you supply one in `reference_data/`.

Run these from the project folder, in either environment.

**Step 1: clone**

```bash
git clone https://github.com/WMD-group/phononDB.git phononDB
```

**Step 2: extract the archives** into `phononDB/phonon_db/`

Linux, or Git Bash / WSL on Windows:

```bash
cd phononDB
bash extract_files_linux.sh        # extract_files_mac.sh on macOS
cd ..
```

Each archive unpacks to one folder, e.g. `phononDB/phonon_db/mp-149-20180417/`, containing `phonon.yaml`, `FORCE_SETS`, `BORN` (about half the materials) and the DFT input files.

**Step 3: build the lookup table**

```bash
cd phononDB
python create_lookup_json.py
cd ..
```

This writes `phononDB/phonon_db_lookup_df.json`, which is the file `MLIP_parser.py` reads. It takes a few minutes. Steps 1 and 3 can also be run from `Kyoto_database_extractor.ipynb` (Cells 2 and 4).

**Build the lookup table on the machine where you run the workflow.** Do not copy a lookup table made on Windows to Linux: depending on the script version it can store Windows-style paths, which the parser cannot read on Linux.

**If the project folder is synced (OneDrive etc.):** `phononDB/` is about 140,000 files. Consider excluding it from sync.

### 5.4 Your own reference files: `reference_data/` (optional)

For a material that is not in phononDB, add `reference_data/<material>_phonopy_params.yaml`. It must be a phonopy file that contains force constants or forces, so `phonopy.load` can use it directly.

Order of preference for each material: phononDB, then `reference_data/`, then none. The source chosen is printed at start-up, e.g. `reference for Si: phononDB mp-149`.

## 6. Running

### Settings (top of `MLIP_runner.py`)

| Setting | Example | Meaning |
|---|---|---|
| `MODEL` | `"MACE-MP-0-medium"` or `"Prophet-OAME-MBD"` | Must match the active conda environment |
| `DEVICE` | `"cpu"` or `"cuda"` | |
| `SUPERCELL` | `2` | N for an N×N×N supercell |
| `DISPLACEMENT` | `0.03` | Finite displacement in Å |
| `MATERIALS` | `["Si", "GaAs", "MgO"]` | CIF names to run. `None` runs every CIF |
| `REFERENCE_MP_IDS` | `{}` or `{"Si": "mp-149"}` | Manual phononDB id when automatic matching fails |
| `RUN_CALCULATIONS` | `True` | Relax and calculate phonons with `MODEL` |
| `RUN_PLOTS` | `True` | Draw the comparison figures from all saved results |
| `IMAGINARY_LIMIT` | `-0.05` | THz. A lower frequency than this counts as an imaginary mode |

Settings that stay the same for every run (`FMAX`, `MAX_STEPS`, `IS_PLUSMINUS`, `BAND_POINTS`) are at the top of `MLIP_processor.py`.

### Run one model

```bash
conda activate mlip-mace
python MLIP_runner.py
```

Run it from the project folder so Python finds `MLIP_parser.py` and `MLIP_processor.py`.

### Compare two models

```bash
conda activate mlip-mace          # MODEL = "MACE-MP-0-medium"
python MLIP_runner.py

conda activate mlip-prophet       # change MODEL to "Prophet-OAME-MBD"
python MLIP_runner.py
```

The second run's figures contain both models, because the plot section loads every model saved at the current `SUPERCELL` and `DISPLACEMENT`.

To redraw figures without loading a model, set `RUN_CALCULATIONS = False`.

### What a working run prints

```
materials: ['Si', 'GaAs', 'MgO']
  reference for Si: phononDB mp-149
  reference for GaAs: none
  reference for MgO: phononDB mp-1265
loaded MACE-MP-0-medium in ... s

===== Si | MACE-MP-0-medium =====
atoms in primitive cell: 2
Displaced supercells: 2
relaxation converged: True (...)
space group: Fd-3m -> Fd-3m
...
```
## 7. Outputs

```
results/
├── summary.csv                                   one row per material per run
├── performance_log.csv                           one row per stage per run (wall time, CPU time, cores)
└── <material>/
    ├── band_comparison_sc<N>_d<disp>.png          all models + reference
    └── <MODEL>/sc<N>_d<disp>/
        ├── relax.log, relaxed.traj, relaxed.cif   relaxation
        ├── phonopy_disp.yaml, phonopy_params.yaml force constants
        └── band.yaml                              band structure
```

- Both CSV files are **appended to**, never overwritten. Re-running the same settings adds new rows; use `date_time` to tell them apart.
- If you change the columns in the code, rename or delete the old CSV first, otherwise new rows will not line up with the old header.
- A material that fails gets a short row with `status = CRASHED in <stage>: <error>` and the run carries on with the next material.

---

## 8. What to keep in mind when reading the results

- **The phononDB reference is PBEsol, not PBE**, with no van der Waals correction.
- **No non-analytical correction** on either side: the reference is loaded with `is_nac=False` and the MLIP phonons have no Born charges. Polar materials (MgO, GaAs) show no LO/TO splitting, and the "Γ optical frequency" is the TO mode.
- Imaginary modes appear as negative frequencies. Only q-points on the band path are checked.
- The reference source is printed at start-up but is not written to `summary.csv`.


## 9. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `No CIF files found in .../materials_cif` | Folder missing or empty | Section 5.1 |
| `No CIF file in materials_cif for ['X']` | A name in `MATERIALS` has no `X.cif` | Add the CIF or fix the name |
| `Prophet weights not found at .../weights/prophet-oame-mbd.pt` | Weights not downloaded, or wrong folder or file name | Section 5.2 |
| `MACE is not installed` / `Prophet is not installed` | Wrong environment active for this `MODEL` | `conda activate` the matching one |
| `cannot import name 'KairosCalculator' from 'prophet'` | Facebook's `prophet` is installed instead of `prophet-mlip` | `pip uninstall prophet`, then section 4 |
| `kernels require a CUDA device` | Prophet called with its GPU kernel on CPU | Keep `use_kernel=False` in `make_prophet` |
| `phononDB lookup table not found` | phononDB not set up | Section 5.3, or ignore if you do not need it |
| `FileNotFoundError: 'phonon_db'` in `create_lookup_json.py` | Archives not extracted | Section 5.3, step 2 |
| `X: more than one phononDB match [...]` | Two database entries share the formula and space group | Add the right id to `REFERENCE_MP_IDS` |
| `reference for X: none` when X should be in phononDB | The CIF's space group does not match the database entry | Add the id to `REFERENCE_MP_IDS` |
| `X: mp-NNN is not in phononDB` | The id in `REFERENCE_MP_IDS` is not in this release | Check the id; use `reference_data/` instead |
| Error loading `phonon.yaml` right after `materials: [...]`, on Linux | Lookup table was built on Windows | Rebuild it on this machine (section 5.3, step 3) |
| CSV rows do not line up with the header | Columns changed since the file was created | Rename or delete the old CSV |
