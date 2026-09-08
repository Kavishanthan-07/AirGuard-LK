# AirGuard-LK

AI Early-Warning System for Industrial Compressor Fault Risk.

## Competition
OctWave 3.0 — Industrial AI & Intelligence

## Main Research Question
Can historical compressor sensor behaviour identify documented fault events early enough to provide useful maintenance warnings?

## Dataset
MetroPT-3

## Main Evaluation
- PR-AUC
- Precision
- Recall
- F1
- Event detection rate
- Median early-warning time
- False alarms per day

## Experimental Rules
- Chronological splitting only
- No random time-series split
- No future information in features
- Test data remains untouched during development
- Event-level evaluation required
- All reported results must come from executed experiments

## Project Structure
- data/ — datasets and processed data
- notebooks/ — competition notebooks
- src/ — reusable project code
- artifacts/ — models, metrics and figures
- report/ — competition report
- presentation/ — competition presentation
- submission/ — final submission package
