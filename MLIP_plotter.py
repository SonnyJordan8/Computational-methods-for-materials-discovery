from pathlib import Path

from MLIP_config import CODE_FOLDER
from MLIP_parser import MATERIALS_CIF_FOLDER, PHONON_DB_FOLDER, MP_REFERENCE_FOLDER, MODELS
from MLIP_parser import cif_parser, reference_data_parser
from MLIP_processor import RESULTS_FOLDER, load_run_phonon, get_primitive_cell, get_band_path, plot_bands

# Settings - change these before each plot
INPUT_PLOT_FILE = "mlip_plotter_input.txt"      # the plot file, relative to the code folder
TITLE = "Si: SevenNet-Omni-i12 vs MACE"                    # the figure title

# Line styles for the runs of one model, in order
LINE_STYLES = ["-", "--", ":", "-."]


def read_input_plot_file(path):
    """
    Reads a plot file. Returns a dictionary with MATERIAL, REFERENCE and RUNS (a list).
    Any line it can't read, an unknown setting, or a missing MATERIAL or RUN raises an error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"plot file not found: {path}")

    plot = {"MATERIAL": None, "REFERENCE": "AUTO", "RUNS": []}
    with open(path) as file:
        for line_number, line in enumerate(file, start=1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            if "=" not in text:
                raise ValueError(f"{path.name} line {line_number}: expected KEY = value, got {text!r}")
            key, value = text.split("=", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "RUN":
                plot["RUNS"].append(value)      # RUN may appear many times, once per run
            elif key in ("MATERIAL", "REFERENCE"):
                plot[key] = value
            else:
                raise ValueError(f"{path.name} line {line_number}: unknown setting {key} "
                                 f"(use MATERIAL, REFERENCE or RUN; the title is set at the top of MLIP_plotter.py)")

    if not plot["MATERIAL"]:
        raise ValueError(f"{path.name}: MATERIAL is missing")
    if not plot["RUNS"]:
        raise ValueError(f"{path.name}: no RUN lines - add one per run to plot")
    return plot


def requested_reference(material_name, reference):
    """
    Turns the plot file's REFERENCE into the form reference_data_parser takes:
    AUTO -> {} (matched automatically), NONE -> None (no reference),
    "<mp-id>" -> {material: mp-id}, "<source> <mp-id>" -> {material: (source, mp-id)}.
    """
    parts = reference.split()
    if len(parts) == 1 and parts[0].upper() == "AUTO":
        return {}
    if len(parts) == 1 and parts[0].upper() == "NONE":
        return None
    if len(parts) == 1:
        return {material_name: parts[0]}
    if len(parts) == 2:
        return {material_name: (parts[0], parts[1])}
    raise ValueError(f"REFERENCE must be AUTO, NONE, <mp-id> or <source> <mp-id>, got {reference!r}")


def run_styles(runs):
    """
    The colour and line style of each run: each model in its colour from MODELS, and a different
    line style for each run of the same model, in the order they are listed.
    A model folder that is not in MODELS is drawn in grey, with a note.
    runs: list of "<model>/<run folder>". Returns {run: {"color": ..., "linestyle": ...}}.
    """
    runs_per_model = {}     # model -> how many of its runs have been given a style so far
    styles = {}
    for run in runs:
        model = run.split("/")[0]
        if model not in runs_per_model:
            runs_per_model[model] = 0
            if model not in MODELS:
                print(f"note: {model} is not in MODELS (MLIP_parser.py), so it is drawn in grey")
        colour = MODELS[model]["colour"] if model in MODELS else "grey"
        styles[run] = {"color": colour,
                       "linestyle": LINE_STYLES[runs_per_model[model] % len(LINE_STYLES)]}
        runs_per_model[model] = runs_per_model[model] + 1
    return styles


def make_comparison_plot(plot_file, title):
    """
    Makes the figure described by one plot file and returns the path it was saved to.
    """
    plot_file = Path(plot_file)
    plot = read_input_plot_file(plot_file)
    material_name = plot["MATERIAL"]

    # The material's structure, for the band path
    materials = cif_parser(MATERIALS_CIF_FOLDER)
    if material_name not in materials:
        raise ValueError(f"No CIF file in {MATERIALS_CIF_FOLDER.name} for {material_name}. "
                         f"Available: {list(materials.keys())}")
    structure = materials[material_name]
    band_path, band_labels = get_band_path(get_primitive_cell(structure))

    # The runs, each rebuilt with the settings it was calculated with
    datasets = {}
    not_found = []
    for run in plot["RUNS"]:
        run = run.replace("\\", "/").strip("/")
        phonon = load_run_phonon(RESULTS_FOLDER / material_name / run)
        if phonon is None:
            not_found.append(run)
        else:
            datasets[run] = phonon
    if not_found:
        raise ValueError(f"no saved run (phonopy_disp.yaml and FORCE_SETS) in "
                         f"{RESULTS_FOLDER / material_name} for: {', '.join(not_found)}")

    # The reference
    reference_phonon = None
    reference_label = "none"
    requested = requested_reference(material_name, plot["REFERENCE"])
    if requested is not None:
        references, sources = reference_data_parser({material_name: structure}, PHONON_DB_FOLDER,
                                                    MP_REFERENCE_FOLDER, requested=requested)
        reference_phonon = references.get(material_name)
        reference_label = sources[material_name]

    figure_file = RESULTS_FOLDER / material_name / f"comparison_{plot_file.stem}.png"
    return plot_bands(datasets, reference_phonon, band_path, band_labels,
                      title=title, figure_file=figure_file, reference_label=reference_label,
                      styles=run_styles(list(datasets.keys())))


# Make the plot
figure_file = make_comparison_plot(CODE_FOLDER / INPUT_PLOT_FILE, TITLE)
print(f"comparison plot: {figure_file}")