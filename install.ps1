# ============================================================
#  install.ps1  –  PowerShell SSL-safe installer
#  Usage:  .\install.ps1
# ============================================================

Write-Host ""
Write-Host "  F1 ML PROJECT - POWERSHELL INSTALLER" -ForegroundColor Red
Write-Host "  ======================================" -ForegroundColor Red
Write-Host ""

$trusted = @(
    "--trusted-host", "pypi.org",
    "--trusted-host", "files.pythonhosted.org",
    "--trusted-host", "pypi.python.org"
)

$packages = @(
    "scikit-learn",
    "numpy",
    "pandas",
    "joblib",
    "streamlit",
    "plotly",
    "dvc",
    "pyyaml",
    "pytest",
    "matplotlib",
    "seaborn"
)

# Upgrade pip
Write-Host "  [1/3] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip @trusted

# Install packages
Write-Host ""
Write-Host "  [2/3] Installing packages..." -ForegroundColor Yellow
foreach ($pkg in $packages) {
    Write-Host "        Installing $pkg ..." -NoNewline
    python -m pip install $pkg @trusted -q
    Write-Host " done" -ForegroundColor Green
}

# Verify
Write-Host ""
Write-Host "  [3/3] Verifying..." -ForegroundColor Yellow
python -c "
import sklearn, numpy, pandas, joblib, streamlit, plotly, yaml
print('  sklearn  :', sklearn.__version__)
print('  numpy    :', numpy.__version__)
print('  pandas   :', pandas.__version__)
print('  joblib   :', joblib.__version__)
print('  streamlit:', streamlit.__version__)
print()
print('  All OK - ready to run!')
"

Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    python src/train_models.py"
Write-Host "    streamlit run app.py"
Write-Host ""
