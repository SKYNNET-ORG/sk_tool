echo off
echo | set /p=Hora de comienzo: 
time /t

for /r %%i in (*.py) do (
    echo Procesando archivo: %%i
    py "%%i" 1>"salida_%%~ni.txt" 2>"error_%%~ni.txt"
    rem py "%%i" 1>"salida_%%~ni.txt" 
)

echo | set /p=Hora de finalizacion: 
time /t

pause