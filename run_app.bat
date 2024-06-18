@echo off
REM Run the FastAPI application

REM Navigate to the directory containing the FastAPI app
cd /d %~dp0

REM Start the FastAPI application in a new command window and keep it running
start cmd /k "uvicorn main:app --reload"

REM Wait for a few seconds to ensure the server starts
timeout /t 5 /nobreak

REM Open the default web browser and navigate to the application URL
start http://127.0.0.1:8000

pause
