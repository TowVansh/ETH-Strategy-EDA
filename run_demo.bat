@echo off
cd /d "%~dp0code"
python -m pip install -q -r requirements.txt
echo.
echo === Baseline backtest ===
python main.py --data ..\data\eth_usdt_perp_30m.csv --preset baseline --out ..\output\trades_with_indicators.csv
echo.
echo === Optimized backtest ===
python main.py --data ..\data\eth_usdt_perp_30m.csv --preset optimized --out ..\output\optimized_trades.csv
echo.
echo === EDA (figures saved to the figures folder) ===
python eda.py
pause
