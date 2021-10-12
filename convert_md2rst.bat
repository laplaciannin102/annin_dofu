@echo off
rem convert markdown to rst
rem author: laplaciannin102
rem usage: Drag and drop the markdown file onto this batch file.

cd /d %~dp0

rem args
set arg1=%1
set markdown_fname=%arg1%

for /f %%i in ('echo %markdown_fname%') do set fname=%%~ni

rem set rst file name
set rst_fname=%fname%.rst

echo rst_fname is
echo %rst_fname%
echo

rem convert
pandoc --standalone --from markdown --to rst %markdown_fname% > %rst_fname%

rem pause

