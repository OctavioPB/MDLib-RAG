@echo off
rem Double-click to install what is missing and open Biblioteca (ADR-021).
rem Keep this window open while you use the app; close it or press Ctrl+C to stop.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
if errorlevel 1 pause
