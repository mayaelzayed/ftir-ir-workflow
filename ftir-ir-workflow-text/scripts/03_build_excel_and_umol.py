# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:09:26 2024

@author: Maya
"""

import pandas as pd
import re
import os
import json

file_name = 'Carroucell-Pyridine-H2-Cycle2 - Python.xlsx'
# Load the Excel file
file_path = f'C:/Users/Maya/OneDrive - ensicaen.fr/Desktop/Philippe/{file_name}'
output_dir = 'C:/Users/Maya/OneDrive - ensicaen.fr/Desktop/Philippe'
mass_data_file = os.path.join(output_dir, f'{file_name}_mass_samples.json')

# Define the band variable
#band = 'A1474'
band = 'A1545'

# Epsilon values based on the band
epsilon_values = {
    'A1474': 1.28,
    'A1545': 1.13
}

# Extract the numeric part from the band for pattern matching
band_number = re.findall(r'\d+', band)[0]

# Function to load or prompt for sample masses
def load_or_prompt_for_masses(sample_names):
    # Load existing mass data if available
    if os.path.exists(mass_data_file):
        with open(mass_data_file, 'r') as f:
            mass_data = json.load(f)
    else:
        mass_data = {}

    # Prompt user for missing sample masses
    for sample_name in sample_names:
        if sample_name not in mass_data:
            mass = float(input(f"Enter the mass for sample '{sample_name}': "))
            mass_data[sample_name] = mass

    # Save the updated mass data
    with open(mass_data_file, 'w') as f:
        json.dump(mass_data, f)

    return mass_data

# Load the Excel file and filter the sheets
def load_and_filter_sheets(file_path):
    xls = pd.ExcelFile(file_path)
    pattern = rf"^\d+°C-\d+Torr-{band_number}"
    sheet_names = [sheet for sheet in xls.sheet_names if re.match(pattern, sheet)]
    return xls, sheet_names

# Function to process and save data across multiple sheets
def process_and_save_data(file_path):
    xls, sheet_names = load_and_filter_sheets(file_path)
    all_sample_data = {}
    all_sample_names = set()

    # Collect all sample names across sheets to prompt for masses
    for sheet_name in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        sample_names = df.iloc[3, :].dropna().values.tolist()
        sample_names = [name for name in sample_names if isinstance(name, str)]
        all_sample_names.update(sample_names)

    # Load or prompt for sample masses
    sample_masses = load_or_prompt_for_masses(all_sample_names)

    # Process each sheet and sample
    for sheet_name in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        df = df.dropna(how='all', axis=1).dropna(how='all')
        # Extract sample names from the 4th row (index 3)
        sample_names = df.iloc[3, :].dropna().values.tolist()
        sample_names = [name for name in sample_names if isinstance(name, str)]
        # Dynamically detect the columns for 'Tps / min' and the band (detect the start and end of each sample)
        tps_min_columns = df.iloc[4, :].eq('Tps / min').to_numpy().nonzero()[0]
        band_columns = df.iloc[4, :].eq(band).to_numpy().nonzero()[0]
        # Ensure equal number of 'Tps / min' and band columns
        print(tps_min_columns,band_columns)
        assert len(tps_min_columns) == len(band_columns)+1, "Mismatch between 'Tps / min' and band columns"


        for i, sample_name in enumerate(sample_names):
            # Dynamically detect the correct columns for this sample
            tps_min_col = tps_min_columns[i]
            band_col = band_columns[i]
            # Extract relevant columns for this sample
            relevant_data = df.iloc[5:, [tps_min_col, band_col]].copy()
            relevant_data.columns = pd.MultiIndex.from_product([[sheet_name], ['Tps / min', band]])
            # Ensure data is numeric and handle missing values
            relevant_data = relevant_data.apply(pd.to_numeric, errors='coerce')
            relevant_data.dropna(subset=[(sheet_name, 'Tps / min'), (sheet_name, band)], inplace=True)

            # Calculate max value for conversion % calculation
            max_band_value = relevant_data[(sheet_name, band)].max()
            #print(f"Max {band} value for {sample_name} in {sheet_name}: {max_band_value}")
            # Calculate conversion %
            
            relevant_data[(sheet_name, 'conversion %')] = (1 - (relevant_data[(sheet_name, band)] / max_band_value)) * 100
            # Check if epsilon is found for the band; if not, print a warning message
            if band in epsilon_values:
                epsilon = epsilon_values[band]
            else:
                print(f"Epsilon value for {band} not found. Please check the band name.")

            # Calculate the new value using the formula band * 1.6 / (mass * epsilon)
            mass = sample_masses[sample_name]
            relevant_data[(sheet_name, 'umol/gcat')] = (1 - (relevant_data[(sheet_name, band)] / max_band_value)) * max_band_value * 2 / (mass/1000 * epsilon)

            if sample_name not in all_sample_data:
                all_sample_data[sample_name] = relevant_data
            else:
                all_sample_data[sample_name] = pd.concat([all_sample_data[sample_name], relevant_data], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    
    for sample_name, data in all_sample_data.items():
        # Create a dedicated folder for each sample
        sample_dir = os.path.join(output_dir, re.sub(r'[\\/*?:"<>|]', "_", sample_name.strip()))
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save the Excel file in the sample's folder
        safe_sample_name = re.sub(r'[\\/*?:"<>|]', "_", sample_name.strip())
        data.to_excel(f"{sample_dir}/{safe_sample_name}_{band}.xlsx")

# Run the function to process the file and save the data
process_and_save_data(file_path)
