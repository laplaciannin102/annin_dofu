@echo off
rem test python modules
rem author: laplaciannin102(Kosuke Asada)

echo _________________________________________________________________________________________
echo this bat file tests python modules.
echo _________________________________________________________________________________________
echo.

cd /d %~dp0

rem unit test
echo ========================================
echo unit test.
echo ^>^>python ^-m unittest discover
echo ========================================
echo.

call python -m unittest discover

echo.
echo _________________________________________________________________________________________
echo.

pause

