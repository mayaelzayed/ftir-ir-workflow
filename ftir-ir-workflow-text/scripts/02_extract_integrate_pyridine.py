# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:51:54 2024

@author: mlaluc
"""
import spectrochempy as scp
import numpy as np
import pandas as pd
import os as os
import matplotlib.pyplot as plt
import time as tps
from matplotlib import cm

serie= "Maya"
folder_source = "Pyridine_Maya"
sample = "BEA11+Pt-Si_50-50"
sonde = "Pyridine"
#activation = "Activation_fresh_samples_2"
main_path = 'C:/Users/mlaluc/OneDrive - ensicaen.fr/Pièces jointes/Maya'
folder_path = f'C:/Users/mlaluc/OneDrive - ensicaen.fr/Pièces jointes/Maya/Bilan des manipes CARROUCELL analysées/cycle 2/{folder_source}/{sample}'
#folder_path_ref = f'C:/Users/mlaluc/OneDrive - ensicaen.fr/In situ/Carroucell/IR_data_for_treatment/{activation}/{sample}'
save_csv_path = f'C:/Users/mlaluc/OneDrive - ensicaen.fr/Pièces jointes/Maya/Maya/Bilan des manipes CARROUCELL analysées/cycle 2/{folder_source}/{sample}/Data'

os.makedirs(save_csv_path, exist_ok=True)
# Define your ranges and parameters
experiments = [ # First spectra for 'equi'
    #{'x_range': (3700.0, 3800.0), 'spectra_range': (1, 101), 'region': 'Si-OH','exp_type': 'equi'},
    #{'x_range': (3625.0, 3700.0), 'spectra_range': (1, 101), 'region': 'Fe-OH','exp_type': 'equi'},
    #{'x_range': (3570.0, 3625.0), 'spectra_range': (1, 101), 'region': 'HF-OH','exp_type': 'equi'},
    #{'x_range': (3550.0, 3800.0), 'spectra_range': (1, 101), 'region': 'Si_OH_Fe','exp_type': 'equi'},
    #{'x_range': (1582.0, 1675.0), 'spectra_range': (1, 101), 'region': 'Lewis','exp_type': 'equi'},
    ##{'x_range': (1505.0, 1575.0), 'spectra_range': (1, 101), 'region': '1545','exp_type': 'equi'},
    ##{'x_range': (1465.0, 1482.0), 'spectra_range': (1, 101), 'region': '1474','exp_type': 'equi'},
    #{'x_range': (1425.0, 1469.0), 'spectra_range': (1, 101), 'region': 'LAS Fe2+ pyr','exp_type': 'equi'},
    {'x_range': (1505.0, 1580.0), 'spectra_range': (1, 101), 'region': 'pyr','exp_type': 'equi'},
    #{'x_range': (880.0, 980.0), 'spectra_range': (1, 101), 'region': 'TOT','exp_type': 'equi'},
    
]

blc_ranges = [([1405.0, 1410.0], [1700.0, 2000.0], [3800.0, 6000.0], [1505.0, 1506.0])]  # Add more as needed
blc_ranges_2 = [([1565.0, 1579.9], [1506.0, 1508.0])]  # Add more as needed

peak_reference_positions = [1636, 1620, 1610, 1575, 1545, 1490, 1474, 1455]


def read_and_preprocess(folder_path, x_range, spectra_range):
    data_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Select a subset of files based on the desired range of spectra
    #selected_data_list = data_list[spectra_range[0]:spectra_range[1]]
    dataset = scp.read_omnic(data_list, directory=folder_path)
    dataset.y -= dataset.y[0]  # baseline correction
    dataset.y.ito("minute")  # unit conversion
    dataset = dataset[:, :]  # keeping specific region
    return dataset

def process_sub_dataset(dataset, x_range, spectra_range):
    subtracted_spectra = dataset-dataset[0]
    subtracted_spectra = subtracted_spectra[spectra_range[0]:spectra_range[1],:]
    #subtracted_spectra = subtracted_spectra[spectra_range[0]:spectra_range[1], x_range[1]:x_range[0]]
    #subtracted_spectra.plot()
    return subtracted_spectra
        
def apply_baseline_correction(subtracted_spectra, blc_ranges, blc_ranges_2, x_range, spectra_range):
    subtracted_spectra.y -= subtracted_spectra.y[1]  # baseline correction
    blc = scp.Baseline(log_level="INFO",
    multivariate=True,  # use a multivariate baseline correction approach
    model="polynomial",  # use a polynomial model
    order="pchip",  # with a pchip interpolation method
    n_components=8)
    blc.ranges = blc_ranges
    dataset_corr = blc.fit(subtracted_spectra)
    dataset_corr = blc.transform()
    dataset_corr.plot()
    dataset_corr = dataset_corr[spectra_range[0]:spectra_range[1],:]
    blc.ranges = blc_ranges_2
    dataset_corr = blc.fit(subtracted_spectra)
    dataset_corr = blc.transform()
    dataset_corr = dataset_corr[spectra_range[0]:spectra_range[1], x_range[1]:x_range[0]]
    #dataset_corr = dataset_corr[spectra_range[0]:spectra_range[1], :]
    return dataset_corr

def plot_and_save(dataset_corr, dpi, save_path, filename, x_range):
    fig, ax = plt.subplots(figsize=(20, 10)) #size of the plot
    fig.patch.set_facecolor('white')  # This sets the entire figure background to white
    # Increase font size globally for the plot
    plt.rcParams.update({'font.size': 20, 'font.family': 'monospace'})  # You can change '12' to your desired font size
    # Replace 'min_x_data' and 'max_x_data' with the actual range of your x data
    ax.set_xlim([x_range[1],x_range[0]])
    # For a PowerPoint presentation, you might scale up:
    #ax.set_title('Your Title', fontsize=32)  # Use larger for PowerPoint
    ax.set_xlabel('Wavenumbers (cm$^{-1}$)', fontsize=24)  # Use larger for PowerPoint
    ax.set_ylabel('Absorbance (a.u.)', fontsize=24)  # Use larger for PowerPoint
    ax.tick_params(axis='both', which='major', labelsize=20, pad=10)  # Use larger for PowerPoint
    # Label the axes correctly
    # Optional: Adjust label padding and tick parameters if needed
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.tick_params(axis='x', which='major', pad=10)
    ax.tick_params(axis='y', which='major', pad=10)
    ax.ticklabel_format(style='plain', axis='x')
    # Reverse the x-axis
    #ax.invert_xaxis()
    ax.set_facecolor('1.0')
    # Plot spectra with different colors but without a color bar
    colors = cm.tab10(np.linspace(0, 1, len(dataset_corr)))
    for i, ds in enumerate(dataset_corr):
        ax.plot(ds.x.data, ds.data[0], color=colors[i % 10])  # Loop over 10 different colors
    # Use tight layout
    ax.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()
            
    ax.autoscale(True)
    ax.autoscale_view(True, True, True)
    # Save the plot
    full_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
    #plt.close(fig)  # Close the plot to free up memory
    

def calculate_area_and_export(dataset_corr,experiments, time_label, area_label):
    area = dataset_corr.trapezoid(dim="x")
    areas = np.array(area)
    areas[areas < 0] = 0  # Replace negative areas with 0
    data_area  = pd.DataFrame({time_label: np.array(dataset_corr.y), area_label: areas})
    return data_area
    
# For exporting results
all_data = []

# Create and plot datasets
for experiment in experiments:
    x_range = experiment['x_range']
    spectra_range = experiment['spectra_range']
    region = experiment['region']
    exp_type= experiment['exp_type']
    
    # Reading and initial preprocessing
    dataset = read_and_preprocess(folder_path, x_range, spectra_range)
    # Sub differences
    subtracted_spectra = process_sub_dataset(dataset, x_range,spectra_range)
    # Baseline correction (assuming the same baseline correction parameters for simplicity)
    dataset_corr = apply_baseline_correction(subtracted_spectra, blc_ranges[0], blc_ranges_2[0], x_range, spectra_range )
    # Assuming dataset_corr.y contains time data in minutes
    time_data = dataset_corr.y.data
    
    #Saving parameters
    current_time = tps.strftime("%Y%m%d-%H%M%S")
    foldername = f'{folder_source}_{sample}'
    filename = f'{foldername}_{current_time}_{region}_{exp_type}.png'
    save_path = f'C:/Users/mlaluc/OneDrive - ensicaen.fr/Pièces jointes/Maya/Bilan des manipes CARROUCELL analysées/{foldername}'
    plot_and_save(dataset_corr, 300, save_path, filename, x_range)

    # Call the multi-peak integration function
    data_area = calculate_area_and_export(dataset_corr, experiments,'Time', f'Area_{region}_{x_range}')
    all_data.append(data_area)
    
# Concatenate all data into a single DataFrame
all_data_df = pd.concat(all_data, ignore_index=True)
csvname=f'{foldername}.csv'
# Combine the path and filename
full_csv_path = os.path.join(save_csv_path, csvname)
all_data_df.to_csv(full_csv_path, index=False)










