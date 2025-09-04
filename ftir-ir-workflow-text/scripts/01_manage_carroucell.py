# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:21:38 2024

@author: Maya
"""

import os
import shutil

sonde = "Colidine" # à changer selon dossier d'expérience
folder_exp_name = "Expérience 25-02-20 (10h58)" # à changer 
exp_name = "25 Torr 110°C(3d)" # à changer selon le type de cata analysé/n° d'exp

#Source is the folder with all the spa from the computer of the carroucell
source_path = f'C:/Users/Maya/OneDrive - ensicaen.fr/Desktop/Python Analysis/Experiences Spectres Brut/cycle 3/{folder_exp_name}'

#folder of the experiments
sonde_folder_path = f'C:/Users/Maya/OneDrive - ensicaen.fr/Desktop/Python Analysis/Experiences Spectres Brut/Spectres Triés/{sonde}_{exp_name}'

def organize_spa_files(source_path):
    # The main folder where subfolders will be created and files will be copied
    sonde_folder = f'{sonde}_{exp_name}'
    os.makedirs(sonde_folder, exist_ok=True)

# Scan the source folder for .spa files
    for file in os.listdir(source_path):
        if file.endswith(".spa"):
            # Extract the part before the last underscore
            prefix = "_".join(file.split("_")[:-1])

            # Determine the path for the subfolder
            subfolder_path = os.path.join(sonde_folder_path, prefix)
            
            # Check if the subfolder exists, if not, create it and log the creation
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                print(f"Created folder: {subfolder_path}")
                
            # Determine the source and destination paths for the file
            source_file_path = os.path.join(source_path, file)
            destination_file_path = os.path.join(subfolder_path, file)
            
            # Copy the file to the destination
            shutil.copy(source_file_path, destination_file_path)
            print(f"Copied {file} to {subfolder_path}")

organize_spa_files(source_path)