@echo off
echo ============================================
echo  Setting up virtual environment...
echo ============================================

cd /d "%~dp0"

python -m venv venv

echo Activating venv...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo To activate the venv in future, run:
echo   venv\Scripts\activate
echo.
echo To verify decompositions:
echo   python tests\test_decompositions.py
echo.
echo To run the full benchmark:
echo   python benchmarks\run_benchmark.py
echo.
pause
