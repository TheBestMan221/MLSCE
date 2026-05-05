@echo off
REM ============================================================
REM  install.bat  –  Windows SSL-safe installer for F1 ML Project
REM  Run this instead of: pip install -r requirements.txt
REM  Double-click or run in PowerShell:  .\install.bat
REM ============================================================

echo.
echo  F1 ML PROJECT - WINDOWS INSTALLER
echo  ===================================
echo.

REM Upgrade pip first with SSL bypass
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

echo.
echo  Installing all dependencies (SSL bypass enabled)...
echo.

python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org ^
    scikit-learn ^
    numpy ^
    pandas ^
    joblib ^
    streamlit ^
    plotly ^
    dvc ^
    pyyaml ^
    pytest ^
    matplotlib ^
    seaborn

echo.
echo  Verifying installation...
python -c "import sklearn, numpy, pandas, joblib, streamlit, plotly, yaml, pytest; print('  All packages installed successfully!')"

echo.
echo  Done! You can now run:
echo    python src/train_models.py
echo    streamlit run app.py
echo.
pause
