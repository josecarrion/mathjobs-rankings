@echo off
REM Build script for MathJobs Rankings App
REM Run this on Windows to create the standalone executable

echo Installing build dependencies...
pip install pyinstaller "flet==0.28.3" pandas numpy scikit-surprise openpyxl

echo.
echo Building executable with flet pack...
flet pack mathjobs_app.py --name "MathJobsRankings" --add-data "mathjobs_rankings_new.py;." -y

echo.
if exist "dist\MathJobsRankings.exe" (
    echo Build complete!
    echo The executable is in the 'dist' folder: dist\MathJobsRankings.exe
) else (
    echo Build may have failed. Check the output above for errors.
)
echo.
pause
