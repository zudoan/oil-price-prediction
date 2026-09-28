@echo off
title PetroForecast AI Server
chcp 65001 > nul
cls
echo ===================================================
echo     PETROFORECAST AI - SINGAPORE ENERGY SYSTEM     
echo ===================================================
echo.
echo Đang khởi động máy chủ Web AI với GPU NVIDIA RTX 4060 Ti...
"C:\Users\AD\AppData\Local\Programs\Python\Python311\python.exe" -X utf8 run_server.py
pause
