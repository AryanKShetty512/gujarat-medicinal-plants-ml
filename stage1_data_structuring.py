"""
STAGE 1 — DATA STRUCTURING
Gujarat Medicinal Plants ML Project

WHAT THIS SCRIPT DOES:
1. Builds a district x plant x year PANEL (long format) from the existing
   district x plant master dataset + the yearly yield/climate time series.
2. Computes stress indices using EACH YEAR'S ACTUAL season climate (more
   precise than the old master file, which used one multi-year average).
3. Standardizes (z-scores) each stress feature WITHIN each plant, so all
   5 plants become comparable on the same scale for pooled modeling later
   (this is what Stage 2's Bayesian model and Stage 3's XGBoost need).

INPUT FILES (must be in the same folder as this script):
 - Gujarat_5Plant_MASTER_FULL.csv
 - Gujarat_YieldClimate_TimeSeries_4Plants.csv   (Isabgol/Fennel/Cumin/Castor)
 - Fenugreek_STEP1_Yield_2019_2022_DOCUMENTED_ONLY.csv
 - Gujarat_Monthly_Weather_TimeSeries_2015_2026.csv

OUTPUT:
 - Stage1_District_Plant_Year_Panel.csv   <- hand this to Person 2 (Aryan)
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# STEP 0: district name standardization (same mapping used throughout the
# project, kept here so this script runs standalone without depending on
# earlier notebook state)
# ---------------------------------------------------------------------------
NAME_MAP = {
    'Kachchh': 'Kutch', 'Kutch': 'Kutch', 'Ahmadabad': 'Ahmedabad', 'Ahmedabad': 'Ahmedabad',
    'Arvalli': 'Aravalli', 'Aravalli': 'Aravalli', 'Banas Kantha': 'Banaskantha', 'Banaskantha': 'Banaskantha',
    'Chhotaudepur': 'Chhota Udepur', 'Chhota Udepur': 'Chhota Udepur',
    'Dangs': 'Dang', 'The Dangs': 'Dang', 'Dang': 'Dang',
    'Devbhoomi Dwarka': 'Devbhumi Dwarka', 'Devbhumi Dwarka': 'Devbhumi Dwarka',
    'Dohad': 'Dahod', 'Dahod': 'Dahod', 'Mahesana': 'Mehsana', 'Mehsana': 'Mehsana',
    'Panch Mahals': 'Panchmahal', 'Panchmahal': 'Panchmahal',
    'Sabar Kantha': 'Sabarkantha', 'Sabarkantha': 'Sabarkantha',
}


def std_district(name):
    return NAME_MAP.get(str(name).strip(), str(name).strip())


def gap(actual, lo, hi):
    """Gap-based stress: 0 if actual is within [lo, hi], else distance outside it."""
    return np.where(
        pd.isna(lo) | pd.isna(hi) | pd.isna(actual), np.nan,
        np.where(actual < lo, lo - actual, np.where(actual > hi, actual - hi, 0))
    )


def build_panel():
    # -----------------------------------------------------------------
    # STEP 1: load the static district+plant info (soil, requirement
    # ranges, biotic risk) from the existing master file
    # -----------------------------------------------------------------
    master = pd.read_csv('Gujarat_5Plant_MASTER_FULL.csv')
    master['District'] = master['District'].apply(std_district)

    static_cols = [
        'District', 'Plant', 'Latitude', 'Longitude',
        'Dominant_Soil_Texture', 'SLUSI_pH_Category', 'SLUSI_Salinity',
        'SLUSI_OC_Class', 'SLUSI_N_Class', 'SLUSI_P_Class', 'SLUSI_K_Class',
        'District_pH_Estimate', 'Optimal_Temp_Min_C', 'Optimal_Temp_Max_C',
        'Optimal_Rainfall_Min_mm', 'Optimal_Rainfall_Max_mm',
        'Optimal_pH_Min', 'Optimal_pH_Max',
        'N_Pests_Documented', 'N_Diseases_Documented', 'Fungal_Disease_Present',
    ]
    static = master[static_cols].copy()

    # -----------------------------------------------------------------
    # STEP 2: load yearly yield + season climate for the 4 crops that
    # have 3 real years of data
    # -----------------------------------------------------------------
    yc4 = pd.read_csv('Gujarat_YieldClimate_TimeSeries_4Plants.csv')
    yc4['District'] = yc4['District'].apply(std_district)

    # -----------------------------------------------------------------
    # STEP 3: build Fenugreek's single year (2019-20) the same way, by
    # aggregating its growing-season months from the monthly weather file
    # -----------------------------------------------------------------
    monthly = pd.read_csv('Gujarat_Monthly_Weather_TimeSeries_2015_2026.csv')
    monthly['District'] = monthly['District'].apply(std_district)

    def agri_year(row):
        y, m = row['Year'], row['Month']
        return f"{y}-{str(y + 1)[2:]}" if m >= 7 else f"{y - 1}-{str(y)[2:]}"

    monthly['Agri_Year'] = monthly.apply(agri_year, axis=1)
    season_climate = monthly.groupby(['District', 'Agri_Year']).agg(
        Season_Avg_Temp_C=('Avg_Temp_C', 'mean'),
        Season_Total_Rainfall_mm=('Total_Rainfall_mm', 'sum'),
        Season_Avg_Humidity_pct=('Avg_Humidity_pct', 'mean'),
    ).reset_index()

    fen = pd.read_csv('Fenugreek_STEP1_Yield_2019_2022_DOCUMENTED_ONLY.csv')
    fen['District'] = fen['District'].apply(std_district)
    fen = fen[['District', 'Average_Yield_kg_ha']].rename(columns={'Average_Yield_kg_ha': 'Yield_kg_ha'})
    fen['Plant'] = 'Fenugreek'
    fen['Agri_Year'] = '2019-20'
    fen = fen.merge(season_climate, on=['District', 'Agri_Year'], how='left')
    fen = fen.dropna(subset=['Yield_kg_ha'])  # keep only districts with a real documented yield

    # -----------------------------------------------------------------
    # STEP 4: combine into one long panel: District x Plant x Year
    # -----------------------------------------------------------------
    panel = pd.concat([
        yc4[['District', 'Plant', 'Agri_Year', 'Yield_kg_ha',
             'Season_Avg_Temp_C', 'Season_Total_Rainfall_mm', 'Season_Avg_Humidity_pct']],
        fen[['District', 'Plant', 'Agri_Year', 'Yield_kg_ha',
             'Season_Avg_Temp_C', 'Season_Total_Rainfall_mm', 'Season_Avg_Humidity_pct']],
    ], ignore_index=True)

    panel = panel.merge(static, on=['District', 'Plant'], how='left')

    # -----------------------------------------------------------------
    # STEP 5: compute stress indices using THAT YEAR'S actual climate
    # (more precise than the multi-year average used in the master file)
    # -----------------------------------------------------------------
    panel['Heat_Stress_C'] = gap(panel['Season_Avg_Temp_C'], panel['Optimal_Temp_Min_C'], panel['Optimal_Temp_Max_C'])
    panel['Rainfall_Stress_mm'] = gap(panel['Season_Total_Rainfall_mm'], panel['Optimal_Rainfall_Min_mm'], panel['Optimal_Rainfall_Max_mm'])
    panel['pH_Stress'] = gap(panel['District_pH_Estimate'], panel['Optimal_pH_Min'], panel['Optimal_pH_Max'])

    # Biotic risk, humidity-modulated per year (same rule as before, now
    # applied per actual yearly humidity instead of a multi-year average)
    panel['Biotic_Risk_Base'] = panel['N_Pests_Documented'] + panel['N_Diseases_Documented']
    panel['Biotic_Risk_Adjusted'] = np.where(
        (panel['Fungal_Disease_Present']) & (panel['Season_Avg_Humidity_pct'] > 75),
        panel['Biotic_Risk_Base'] + 1, panel['Biotic_Risk_Base']
    )

    # -----------------------------------------------------------------
    # STEP 6: z-score each stress feature WITHIN each plant (this is the
    # key Stage 1 deliverable — makes the 5 plants comparable for pooled
    # modeling in Stage 2/3, since each plant has a totally different
    # ideal range otherwise)
    # -----------------------------------------------------------------
    for col in ['Heat_Stress_C', 'Rainfall_Stress_mm', 'pH_Stress', 'Biotic_Risk_Adjusted']:
        panel[f'{col}_zscore'] = panel.groupby('Plant')[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
        )

    return panel


def main():
    panel = build_panel()
    panel.to_csv('Stage1_District_Plant_Year_Panel.csv', index=False)

    # ---- validation printout, so you can confirm it worked correctly ----
    print("Panel shape:", panel.shape)
    print("\nRows per plant:")
    print(panel.groupby('Plant').size())
    print("\nYears covered per plant:")
    print(panel.groupby('Plant')['Agri_Year'].unique())
    print("\nNulls per column (should only be in expected places):")
    nulls = panel.isna().sum()
    print(nulls[nulls > 0])
    print("\nZ-score sanity check (each plant's z-scored columns should have mean ~0):")
    print(panel.groupby('Plant')[['Heat_Stress_C_zscore', 'Rainfall_Stress_mm_zscore', 'pH_Stress_zscore']].mean().round(3))
    print("\nSaved: Stage1_District_Plant_Year_Panel.csv")


if __name__ == '__main__':
    main()
