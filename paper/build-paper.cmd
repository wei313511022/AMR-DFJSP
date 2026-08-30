@echo off
setlocal

rem Build the paper without latexmk/Perl.
rem Usage:
rem   build-paper.cmd
rem   build-paper.cmd path\to\another.tex

if "%~1"=="" (
    set "TEX_FILE=%~dp0main.tex"
) else (
    set "TEX_FILE=%~f1"
)

if not exist "%TEX_FILE%" (
    echo [ERROR] TeX file not found: "%TEX_FILE%"
    exit /b 2
)

for %%F in ("%TEX_FILE%") do set "DOC_DIR=%%~dpF"
for %%F in ("%TEX_FILE%") do set "TEX_NAME=%%~nxF"
for %%F in ("%TEX_FILE%") do set "JOB_NAME=%%~nF"

set "PDFLATEX="
set "BIBTEX="

if exist "%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" set "PDFLATEX=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
if not defined PDFLATEX if exist "%ProgramFiles%\MiKTeX\miktex\bin\x64\pdflatex.exe" set "PDFLATEX=%ProgramFiles%\MiKTeX\miktex\bin\x64\pdflatex.exe"
if not defined PDFLATEX for %%I in (pdflatex.exe) do set "PDFLATEX=%%~$PATH:I"

if exist "%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\bibtex.exe" set "BIBTEX=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\bibtex.exe"
if not defined BIBTEX if exist "%ProgramFiles%\MiKTeX\miktex\bin\x64\bibtex.exe" set "BIBTEX=%ProgramFiles%\MiKTeX\miktex\bin\x64\bibtex.exe"
if not defined BIBTEX for %%I in (bibtex.exe) do set "BIBTEX=%%~$PATH:I"

if not defined PDFLATEX (
    echo [ERROR] pdflatex.exe was not found. Install MiKTeX or add it to PATH.
    exit /b 3
)

if not defined BIBTEX (
    echo [ERROR] bibtex.exe was not found. Install MiKTeX or add it to PATH.
    exit /b 3
)

pushd "%DOC_DIR%" || exit /b 4

echo.
echo [1/4] pdflatex: initial pass
"%PDFLATEX%" -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error "%TEX_NAME%"
if errorlevel 1 goto :build_failed

echo.
echo [2/4] bibtex: references
"%BIBTEX%" "%JOB_NAME%"
if errorlevel 1 goto :build_failed

echo.
echo [3/4] pdflatex: citations and references
"%PDFLATEX%" -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error "%TEX_NAME%"
if errorlevel 1 goto :build_failed

echo.
echo [4/4] pdflatex: final pass
"%PDFLATEX%" -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error "%TEX_NAME%"
if errorlevel 1 goto :build_failed

echo.
echo [OK] Build completed: "%DOC_DIR%%JOB_NAME%.pdf"
popd
exit /b 0

:build_failed
set "BUILD_EXIT=%ERRORLEVEL%"
echo.
echo [ERROR] Build stopped. Review "%DOC_DIR%%JOB_NAME%.log" above for details.
popd
exit /b %BUILD_EXIT%
