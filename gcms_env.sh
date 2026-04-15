#!/bin/bash
# setup_gcms_env.sh - Create a Python virtual environment for GCMS IFU analysis
# with all required dependencies for gcms_reduce.py

set -e  # exit on error

ENV_NAME="gcms_ifu_env"
PYTHON="python3"

echo "Creating virtual environment: $ENV_NAME"
$PYTHON -m venv "$ENV_NAME"

echo "Activating environment"
source "$ENV_NAME/bin/activate"

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing core scientific packages"
pip install numpy scipy matplotlib

echo "Installing astronomy and FITS packages"
pip install astropy

echo "Installing fitting and kinematics packages"
pip install lmfit

echo "Installing astroquery (for SIMBAD, NED, SDSS queries)"
pip install astroquery

echo "Installing reproject (for SDSS image alignment)"
pip install reproject

echo "Installing optional but useful packages"
pip install jupyter notebook seaborn tqdm

echo "Verifying key imports..."
$PYTHON -c "import numpy, scipy, matplotlib, astropy, lmfit, astroquery, reproject; print('All packages imported successfully.')"

echo ""
echo "Setup complete. To activate the environment later, run:"
echo "  source $ENV_NAME/bin/activate"

chmod +x setup_gcms_env.sh
./setup_gcms_env.sh
source gcms_ifu_env/bin/activate
