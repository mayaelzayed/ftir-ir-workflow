# Annex X — Python Workflow for FTIR Data Extraction and Kinetic Analysis (Text Only)

This annex documents the in-house Python workflow used to (i) organize raw Thermo OMNIC `.spa` spectra, (ii) perform multivariate baseline correction and numerical integration of diagnostic pyridine (and colidine) bands, (iii) convert band areas into *conversion* and **μmol·g₍cat₎⁻¹**, and (iv) extract initial rates and the apparent reaction order with respect to H₂.

## Textual overview
1. **Data organization** — `01_manage_carroucell.py`: sort `.spa` into per-experiment folders.
2. **Processing & integration** — `02_extract_integrate_pyridine.py`: subtract first spectrum, baseline (PCHIP), integrate windows (e.g., 1505–1580 cm⁻¹), export CSV.
3. **Workbooks & μmol·g₍cat₎⁻¹** — `03_build_excel_and_umol.py`: compute conversion and μmol·g₍cat₎⁻¹ using ε and mass JSON.
4. **Kinetics** — `04_initial_rates_and_order.py`: fixed-origin r₀, ln–ln vs P(H₂) in Pa for apparent order *n*.
5. **Environment** — Python ≥3.10; SpectroChemPy, NumPy, pandas, Matplotlib, scikit-learn, openpyxl.

**Formulae**  
Conversion (%) = `1 − A(t)/A_max`  
Amount (μmol·g₍cat₎⁻¹) = `[(1 − A/A_max) · A_max · 2] / (m_cat/1000 · ε)`
