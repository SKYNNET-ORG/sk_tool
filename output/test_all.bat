
for /r %%i in (*.py) do (
    rem py "%%i" 1>"salida_%%~ni.txt" 2>"error_%%~ni.txt"
    py "%%i" 1>"salida_%%~ni.txt" 
)

pause