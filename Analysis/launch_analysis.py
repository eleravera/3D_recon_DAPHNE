import subprocess
import os
import re
import time

#DONE
"""
#400 pixels, @53 cm, 8 detector
    {"directory": "2024-12-21_12-43-27", "criteria": [3, 3, 0.1]}, 
    {"directory": "2024-12-21_12-43-27", "criteria": [3.0, 5.0, 0.1]}, 
    {"directory": "2024-12-21_12-43-27", "criteria": [5.0, 3.0, 0.1]},
    {"directory": "2024-12-21_12-43-27", "criteria": [5.0, 5.0, 0.1]}    
    {"directory": "2024-12-21_12-43-27", "criteria": [3.0, 10.0, 0.1]}

"""


#GOING ON DIFFERENT CORES 
"""
"""


##############################################################################################
#To do the gamma analisi

configurations = [
{"directory": "2025-01-21_09-53-49/", "criteria": [3.0, 3.0, 0.1]}, #500 pixels, @53 cm, 8 det
{"directory": "2025-01-21_09-53-49/", "criteria": [3.0, 5.0, 0.1]}, 
{"directory": "2025-01-21_09-53-49/", "criteria": [5.0, 3.0, 0.1]}, 
{"directory": "2025-01-21_09-53-49/", "criteria": [5.0, 5.0, 0.1]}
] 

path_to_search = "/home/eleonora/3D_recon_DAPHNE/Reconstruction/"
input_file = "/home/eleonora/eFLASH_3D_Sim-build/Dose_Map_30mm/spettro_9MeV/doseDistribution.npz" 

for config in configurations:
    # Separiamo i criteri in singoli argomenti
    criteria_args = [str(c) for c in config["criteria"]]  # Trasformiamo i criteri in stringhe
    command = [
        "python3",
        "process_reco_files.py",
        input_file, 
        os.path.join(path_to_search, config["directory"]), 
        "--do_Gamma", 
        "--criteria"] + criteria_args  
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
    

###########################################
#Only for output file     
"""
   
path_to_search = "/home/eleonora/3D_recon_DAPHNE/Reconstruction/"

#directories = [d for d in os.listdir(path_to_search) if os.path.isdir(os.path.join(path_to_search, d))]

directories = ['2025-01-21_09-53-49', '2025-01-21_15-42-55', '2025-01-23_13-15-31', '2025-01-23_14-17-15', '2025-01-23_17-44-21', '2025-01-23_19-18-25']


input_file = "/home/eleonora/eFLASH_3D_Sim-build/Dose_Map_30mm/spettro_9MeV/doseDistribution.npz"  # "../Data/cylindrical_phantom.npz"

for dir in directories:
    # Separiamo i criteri in singoli argomenti
    command = [
        "python3", 
        "process_reco_files.py",
        input_file, 
        os.path.join(path_to_search, dir)+'/']          
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")"""




#per quelli con smoothing
"""
directories = ["2025-01-15_10-01-27", "2025-01-15_18-50-34", "2025-01-15_21-07-25", "2025-01-15_23-28-45"]

configurations = [
    {"directory": "2025-01-15_10-01-27", "sigma": 2}, 
    {"directory": "2025-01-15_18-50-34", "sigma": 4},    
    {"directory": "2025-01-15_21-07-25", "sigma": 6}, 
    {"directory": "2025-01-15_23-28-45", "sigma": 8},            
]


input_file = "/home/eleonora/3D_recon_DAPHNE/Data/cylindrical_phantom.npz"

for config in configurations:
    directory = config["directory"]
    sigma = config["sigma"]

    # Creazione del comando
    command = [
        "python3",
        "process_reco_files.py",
        input_file,
        os.path.join(path_to_search, directory) + '/',
        "--sigma", str(sigma)  # Convertiamo sigma in stringa
    ]
    
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command for directory {directory}: {e}")

"""

#python -i process_reco_files.py /home/eleonora/3D_recon_DAPHNE/Data/cylindrical_phantom.npz /home/eleonora/3D_recon_DAPHNE/Reconstruction/2024-12-21_12-43-27/ --do_Gamma --criteria 1. 1. 1.
