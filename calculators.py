# MLIP calculators for the phonon workflow (version 3, simplified)
#
# Each model has one function that builds its ASE calculator.
# The MODELS dictionary at the bottom links each model name to its function.
# phonon_workflow.py uses it like this:  calc = MODELS["MACE-MP-0-medium"]("cpu")
#
# To add a model: write one make_... function and add one line to MODELS.
#
# The mace and prophet imports are inside the functions on purpose. The two
# packages cannot be installed in the same conda environment (mace-torch needs
# e3nn 0.4.4, Prophet needs e3nn 0.6 or later). Importing inside the function
# means this file still loads when only one of them is installed.
#
# Conda environments on my laptop:
#   MACE-MP-0-medium  -> quantum-materials
#   Prophet-OAME-MBD  -> mlip-prophet

# Handles file paths as objects, used to find the Prophet weights file
from pathlib import Path


# Benchmark model: MACE-MP-0 medium - Batatia et al. 2025
def make_mace(device):
    # mace_mp - loads a MACE-MP-0 model and returns an ASE calculator
    from mace.calculators import mace_mp

    # model="medium" is downloaded by mace-torch the first time it is used.
    # float64 because finite-difference forces are sensitive to precision - Orb-v3 paper Fig. 8
    return mace_mp(model="medium", default_dtype="float64", device=device)


# Novel model: Prophet-OAME-MBD
# The weights file was downloaded by hand into the "checkpoints" folder of
# Week_7_phonon_workflow, which is one folder above this V3 folder.
PROPHET_WEIGHTS = Path(__file__).resolve().parent.parent / "checkpoints" / "prophet-oame-mbd.pt"
# Where the weights file was downloaded from (pinned to one exact version)
PROPHET_URL = "https://huggingface.co/kairosmaterial/prophet/resolve/f9a54df874ba52b8b35e7d3f7f348b46e0135563/prophet-oame-mbd.pt"


def make_prophet(device):
    # Stop with a clear message if the weights file is missing
    if not PROPHET_WEIGHTS.exists():
        raise FileNotFoundError(f"Prophet weights not found at {PROPHET_WEIGHTS}. Download them from {PROPHET_URL}")

    # KairosCalculator - the ASE calculator for Prophet
    from prophet import KairosCalculator

    # use_kernel=False skips Prophet's GPU kernel, so it also runs on CPU - Prophet README
    return KairosCalculator(model_path=str(PROPHET_WEIGHTS), use_kernel=False, device=device)


# Model name -> function that builds its calculator
MODELS = {
    "MACE-MP-0-medium": make_mace,
    "Prophet-OAME-MBD": make_prophet,
}
