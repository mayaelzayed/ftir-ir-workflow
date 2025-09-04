# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:13:07 2024

@author: Maya
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import re

# Define the sample variable
sample = 'Pt.Si+ZSM5(11) (50-50) (Col + Pyr)'
# Define the band variable
#band = '1474'
band = '1545'
file_name = f'{sample}_A{band}.xlsx'
# Load the Excel file
file_path = f'C:/Users/Maya/OneDrive - ensicaen.fr/Desktop/Philippe/{sample}/{file_name}'
save_path = f'C:/Users/Maya/OneDrive - ensicaen.fr/Desktop/Philippe/{sample}/{band}'

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Dynamically detect the sheet names
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
# Automatically choose the first sheet, or you could prompt the user to select one
# For now, we'll select the first sheet by default
sheet_name = sheet_names[0]  # You can change this if you want a different sheet

# Load the sheet without specifying headers to retain both rows
data_sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

# Remove the first column (assuming it's an index or unnecessary)
data_sheet = data_sheet.drop(columns=[0])

# Detect the pattern in the first row and create a new row with only pressure values
# Extract the numeric part from the band for pattern matching
# Define the band variable
band_number = re.findall(r'\d+', band)[0]
pattern = rf"^(\d+°C)-(\d+Torr)-{band_number}"
new_header_level_1 = []
new_header_level_2 = []

for i, cell in enumerate(data_sheet.iloc[0]):
    match = re.search(pattern, str(cell))
    if match:
        new_header_level_1.append(match.group(2))  # Extract the Torr value
    else:
        new_header_level_1.append(None)

# The second row will be the actual column names (time, conversion, etc.)
new_header_level_2 = data_sheet.iloc[1].tolist()

# Replace NaN values in the header with the previous non-NaN Torr value
for i in range(1, len(new_header_level_1)):
    if new_header_level_1[i] is None:
        new_header_level_1[i] = new_header_level_1[i - 1]

# Create a multi-level header
data_sheet.columns = pd.MultiIndex.from_arrays([new_header_level_2, new_header_level_1])

# Drop the first two rows now that we've extracted the headers
data_cleaned = data_sheet.drop(index=[0,1,2])

def identify_pressure_conditions(data):
    conditions = {}
    for i in range(0, len(data.columns), 4):
        time_col = data.columns[i]
        conversion_col = data.columns[i + 2]
        umol_col = data.columns[i + 3]
        # Extract pressure value from the header, it's the second element of the tuple
        pressure_value = time_col[1]  # Extracting the '100Torr', '15Torr', etc. from the second part of the tuple
        # Store the condition for this pressure
        conditions[pressure_value] = (time_col, conversion_col, umol_col)
#    print(conditions)
    return conditions

def determine_initial_rate_fixed_origin(data, time_col, conversion_col, umol_col, pressure):
    """
    Determine the initial rate of reaction by fitting the initial linear portion of the conversion and umol/gcat data
    starting from t=0 min, fixed to the origin (0,0), and providing the best R² value with at least 5 data points.
    
    Parameters:
    data (DataFrame): The cleaned data containing the relevant columns.
    time_col (str): The name of the time column.
    conversion_col (str): The name of the conversion column.
    pressure (str): The pressure condition for labeling.
    umol_col (str, optional): The name of the 'umol/gcat' column for additional fitting and plotting.
    
    Returns:
    initial_rate (float): The initial rate of reaction based on conversion.
    umol_rate (float, optional): The initial rate based on umol/gcat if umol_col is provided.
    """

    # Filter the data to ensure it starts from t = 0 min (for conversion)
    data_condition = data[[time_col, conversion_col]].dropna()
    data_condition = data_condition[data_condition[time_col] >= 0]
    data_condition = data_condition.reset_index(drop=True)

    best_r2 = -np.inf
    best_model = None
    best_n = 0
    
    # Iterate over different end points to find the best R² with at least 5 data points for conversion
    for n in range(5, len(data_condition)):  # Start with at least 5 points
        X = data_condition[time_col][:n].values.reshape(-1, 1)
        y = data_condition[conversion_col][:n].values
        
        # Perform linear regression with the intercept fixed to zero
        model = LinearRegression(fit_intercept=False).fit(X, y)
        r2 = model.score(X, y)
        
        if r2 > best_r2:
            best_r2 = r2
            best_n = n
            best_model = model
    
    initial_rate = best_model.coef_[0]
    
    # Plot the conversion data and the best fit
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(data_condition[time_col], data_condition[conversion_col], 'o', color='black', markersize=4 )#, label='Conversion Data')
    
    # Plot the best fit line
    X_fit = data_condition[time_col][:best_n].values.reshape(-1, 1)
    y_fit = best_model.predict(X_fit)
    plt.plot(X_fit, y_fit, '-', color='red', linewidth=1.5, label=f'Best Fit (Fixed Origin): R² = {best_r2:.4f}\nRate = {initial_rate:.4f} min$^{-1}$')

    plt.xlabel('Time (min)', fontsize=12, fontname='Arial')
    plt.ylabel('Conversion (%)', fontsize=12, fontname='Arial')
    plt.title(f'Initial Rate Determination for {pressure} of H$_2$', fontsize=14, fontname='Arial')

    plt.legend(loc='best', fontsize=10, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(save_path, f'r0_{pressure}_min-1.tif')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(plot_filename, format='tiff', dpi=300)
    
    plt.show()

    # If umol/gcat column is provided, perform the same fitting for umol/gcat vs. time
    if umol_col:
        data_umol = data[[time_col, umol_col]].dropna()
        data_umol = data_umol[data_umol[time_col] >= 0]
        data_umol = data_umol.reset_index(drop=True)

        best_r2_umol = -np.inf
        best_model_umol = None
        best_n_umol = 0
        
        # Iterate over different end points to find the best R² with at least 5 data points for umol/gcat
        for n in range(2, 5):
            X_umol = data_umol[time_col][:n].values.reshape(-1, 1)
            y_umol = data_umol[umol_col][:n].values
            
            # Perform linear regression with the intercept fixed to zero
            model_umol = LinearRegression(fit_intercept=False).fit(X_umol, y_umol)
            r2_umol = model_umol.score(X_umol, y_umol)
            
            if r2_umol > best_r2_umol:
                best_r2_umol = r2_umol
                best_n_umol = n
                best_model_umol = model_umol
        
        umol_rate = best_model_umol.coef_[0]

        # Plot the umol/gcat data and the best fit
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(data_umol[time_col], data_umol[umol_col], 'o', color='black', markersize=4)#, label='umol.min-1.gcat-1 Data')
        
        # Plot the best fit line for umol/gcat
        X_fit_umol = data_umol[time_col][:best_n_umol].values.reshape(-1, 1)
        y_fit_umol = best_model_umol.predict(X_fit_umol)
        plt.plot(X_fit_umol, y_fit_umol, '-', color='red', linewidth=1.5, label=f'Best Fit (Fixed Origin): R² = {best_r2_umol:.4f}\nRate = {umol_rate:.4f} min$^{-1}$')

        plt.xlabel('Time (min)', fontsize=12, fontname='Arial')
        plt.ylabel('$\\mu$mol.min$^{-1}$.g$_{cat}^{-1}$', fontsize=12, fontname='Arial')
        plt.title(f'Initial Rate Determination for {pressure} of H$_2$', fontsize=14, fontname='Arial')
        
        plt.legend(loc='best', fontsize=10, frameon=False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(save_path, f'r0_{pressure}_umol.min-1.gcat-1.tif')
        plt.savefig(plot_filename, format='tiff', dpi=300)
        
        plt.show()

    return initial_rate, umol_rate

# Analyze all pressure conditions
pressure_conditions_dict = identify_pressure_conditions(data_cleaned)

r0_min = {}
r0_umol_gcat_min = {}

for pressure, cols in pressure_conditions_dict.items():
    time_col, conversion_col, umol_col = cols
    initial_rate, umol_rate = determine_initial_rate_fixed_origin(data_cleaned, time_col, conversion_col, umol_col, pressure=pressure)
    r0_min[pressure] = initial_rate
    r0_umol_gcat_min[pressure] = umol_rate
    
    # Optionally store umol_rate if needed

# Display the initial rates
print("Initial Reaction Rates:")
for pressure, rate in r0_min.items():
    print(f"{pressure}: {rate:.4f} min-1")
    
# Display the initial rates
print("Initial Reaction Rates:")
for pressure, rate in r0_umol_gcat_min.items():
    print(f"{pressure}: {rate:.4f} umol.gcat.min")

# Convert the second element of the tuple (pressure) to numeric values for sorting
pressure_conditions = sorted([float(key.replace('Torr', '')) for key in r0_min.keys()])

# Now you can proceed to sort and extract the initial rates accordingly
r0_min_sorted = [r0_min[key] for key in sorted(r0_min.keys(), key=lambda x: float(x.replace('Torr', '')))]
r0_umol_gcat_min_sorted = [r0_umol_gcat_min[key] for key in sorted(r0_umol_gcat_min.keys(), key=lambda x: float(x.replace('Torr', '')))]

# Print the sorted rates
print(r0_min_sorted)
print(r0_umol_gcat_min_sorted)

# Function to determine the reaction order
def determine_reaction_order(pressure_conditions, initial_rates, umol_rates):
    """
    Automatically determine the reaction order from provided pressures and initial rates.
    
    Parameters:
    pressures (list or np.array): The partial pressures of H2 in Torr.
    initial_rates (list or np.array): The corresponding initial rates in units/min.

    Returns:
    slope (float): The determined reaction order (slope of log-log plot).
    """
    pressure_conditions = np.array(pressure_conditions)*133.322
    initial_rates = np.array(initial_rates)
    umol_rates = np.array(umol_rates)

    # Convert pressures and initial rates to log scale
    ln_pressures = np.log(pressure_conditions)
    ln_initial_rates = np.log(initial_rates)
    ln_umol_rates = np.log(umol_rates)

    # Perform linear regression to find the slope (reaction order n)
    model = LinearRegression()
    ln_pressures_reshaped = ln_pressures.reshape(-1, 1)
    model.fit(ln_pressures_reshaped, ln_initial_rates)
    slope = model.coef_[0]  # The slope n
    
    model_umol = LinearRegression()
    model_umol.fit(ln_pressures_reshaped, ln_umol_rates)
    slope_umol = model_umol.coef_[0]  # The slope n
    
    # Plot the log-log graph
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(ln_pressures, ln_initial_rates, 'o', color='black', markersize=5)#, label='Data points')
    plt.plot(ln_pressures, model.predict(ln_pressures_reshaped), '-', color='red', linewidth=1.5, label=f'Fit: slope = {slope:.4f}')

    plt.xlabel(r'Ln(P (H$_2$))', fontsize=12, fontname='Arial')
    plt.ylabel('Ln(r$_0$) (min)$^{-1}$)', fontsize=12, fontname='Arial')
    plt.title('Ln-Ln Plot: Initial Rate vs. Partial Pressure of H$_2$ in Pa', fontsize=14, fontname='Arial')

    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Save the plot as a high-resolution TIFF image
    plot_filename = os.path.join(save_path, 'Reaction_Order_Ln_Ln_Plot_r0_min_Pa.tif')
    plt.savefig(plot_filename, format='tiff', dpi=300)

    plt.show()
    
    # Plot the log-log graph
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(ln_pressures, ln_umol_rates, 'o', color='black', markersize=5)#, label='Data points')
    plt.plot(ln_pressures, model_umol.predict(ln_pressures_reshaped), '-', color='red', linewidth=1.5, label=f'Fit: slope = {slope_umol:.4f}')

    plt.xlabel(r'Ln(P (H$_2$))', fontsize=12, fontname='Arial')
    plt.ylabel(r'Ln(r$_0$) ($\mu$mol.min$^{-1}$.g$_{cat}^{-1}$)', fontsize=12, fontname='Arial')
    plt.title(r'Ln-Ln Plot: Initial Rate vs. Partial Pressure of H$_2$ in Pa', fontsize=14, fontname='Arial')

    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Save the plot as a high-resolution TIFF image
    plot_filename = os.path.join(save_path, 'Reaction_Order_Ln_Ln_r0_min_umol_gcat_Pa.tif')
    plt.savefig(plot_filename, format='tiff', dpi=300)
        
    plt.show()
    
    pressure_conditions_reshapes = pressure_conditions.reshape(-1, 1)
    model_raw = LinearRegression()
    model_raw.fit(pressure_conditions_reshapes, umol_rates)
    slope_raw = model_raw.coef_[0]  # The slope n
    
    # Plot the normal graph
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(pressure_conditions, umol_rates, 'o', color='black', markersize=5)#, label='Data points')
    plt.plot(pressure_conditions, model_raw.predict(pressure_conditions_reshapes), '-', color='red', linewidth=1.5, label=f'Fit: slope = {slope_raw:.4f}')

    plt.xlabel(r'(P (H$_2$) (Pa))', fontsize=12, fontname='Arial')
    plt.ylabel('(r$_0$) (min)$^{-1}$)', fontsize=12, fontname='Arial')
    plt.title('Plot: Initial Rate in µmol.g-1.min-1 vs. Partial Pressure of H$_2$ in Pa', fontsize=14, fontname='Arial')

    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Save the plot as a high-resolution TIFF image
    plot_filename = os.path.join(save_path, 'Reaction_Order_Plot_r0_min_Pa.tif')
    plt.savefig(plot_filename, format='tiff', dpi=300)

    plt.show()

    # Return the determined reaction order n
    return slope, slope_umol, slope_raw

reaction_order = determine_reaction_order(pressure_conditions, r0_min_sorted, r0_umol_gcat_min_sorted)
# Assuming reaction_order is a tuple containing two values: slope and slope_umol
slope, slope_umol, slope_raw = reaction_order

# Print the slopes with the correct formatting
print(f"Determined reaction order n (conversion %): {slope:.4f}")
print(f"Determined reaction order n (umol/gcat): {slope_umol:.4f}")
print(f"Constante de vitesse (umol/gcat_sans_log): {slope_raw:.4f}")
