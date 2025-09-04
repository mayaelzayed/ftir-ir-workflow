# FTIR Carroucell Data Processing (Pyridine/Colidine on Bifunctional Catalysts)

This repository contains a compact, reproducible workflow to organize raw Thermo `.spa` spectra, perform baseline-corrected integration of diagnostic FTIR bands (e.g., 1545 and 1474 cm⁻¹), build analysis workbooks (conversion and μmol·g₍cat₎⁻¹), and extract initial rates and reaction orders.

## Textual workflow overview (no figure)
1. **Organize raw data** — `scripts/01_manage_carroucell.py`  
   - Sort `.spa` files into per-experiment folders inferred from filename prefixes.

2. **Process & integrate bands** — `scripts/02_extract_integrate_pyridine.py`  
   - Read OMNIC `.spa` with SpectroChemPy.  
   - Subtract the first spectrum, apply multivariate polynomial baseline (PCHIP), and integrate user-defined windows (e.g., 1505–1580 cm⁻¹).  
   - Export CSV of time vs area and diagnostic plots.

3. **Build analysis workbooks** — `scripts/03_build_excel_and_umol.py`  
   - Merge per-sample series; compute **Conversion (%)** = `1 − A(t)/A_max`.  
   - Compute **Amount (μmol·gcat⁻¹)** = `[(1 − A/A_max) · A_max · 2] / (m_cat/1000 · ε)` with ε = 1.13 (1545 cm⁻¹) or 1.28 (1474 cm⁻¹); `m_cat` in mg (persisted in JSON).

4. **Kinetics** — `scripts/04_initial_rates_and_order.py`  
   - Determine **r₀** from an early-time linear segment constrained through the origin.  
   - Regress `ln r₀` vs `ln P(H₂)` (Pa) to estimate the apparent reaction order *n*; export publication-ready figures.

5. **Reproducibility**  
   - Python ≥ 3.10; dependencies in `requirements.txt`.  
   - Use relative paths (`data/raw`, `data/processed`, `figures`) and commit the per-sample mass JSON.

## Repository layout
```
ftir-ir-workflow/
├─ scripts/
│  ├─ 01_manage_carroucell.py
│  ├─ 02_extract_integrate_pyridine.py
│  ├─ 03_build_excel_and_umol.py
│  └─ 04_initial_rates_and_order.py
├─ requirements.txt
├─ LICENSE
└─ CITATION.cff
```

## Quickstart
```bash
python -m venv .venv
. .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Run the scripts in order (1 → 4).

## Citation
If you use this workflow, please cite using `CITATION.cff`.
