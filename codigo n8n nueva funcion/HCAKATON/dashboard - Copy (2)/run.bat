@echo off
echo ========================================
echo   EcoCampus UPTC - Iniciando Dashboard
echo   Hackathon IAMinds 2026
echo ========================================
echo.

cd /d "%~dp0"

echo Instalando dependencias...
pip install -r requirements.txt -q

echo.
echo Iniciando servidor Streamlit...
echo Abra su navegador en: http://localhost:8501
echo.
echo Presione Ctrl+C para detener el servidor
echo.

streamlit run app.py --server.port 8501

pause
