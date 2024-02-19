
for /r %%i in (*.py) do (
    py "%%i" 1>"salida_%%~ni.txt" 2>"error_%%~ni.txt"
)

pause