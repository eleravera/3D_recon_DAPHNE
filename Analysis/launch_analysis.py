import subprocess
import os
import re
import time

##############################################################################################
#To do the gamma analisi

configurations = [
#{"directory": "2025-01-30_04-29-08/", "criteria": [3.0, 3.0, 0.0]} #doing
{"directory": "2025-03-07_15-37-12/", "criteria": [3.0, 3.0, 0.0]},
{"directory": "2025-03-07_13-21-04/", "criteria": [3.0, 3.0, 0.0]}
] 


path_to_search = "/home/eleonora/3D_recon_DAPHNE/Reconstruction/"
input_file = "/home/eleonora/3D_recon_DAPHNE/Data/SheppLogan3D.npz"  # /home/eleonora/3D_recon_DAPHNE/Data/doseDistribution_flip.npz

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

directories = ['2024-12-20_11-22-47', '2024-12-20_12-01-39', '2024-12-21_12-43-27', '2024-12-21_14-57-21', '2024-12-21_17-44-43', '2024-12-21_21-01-11', '2025-01-04_12-59-04', '2025-01-04_16-37-48', '2025-01-04_20-17-09', '2025-01-05_00-34-48', '2025-01-05_05-22-16', '2025-01-07_18-44-49', '2025-01-08_00-42-46', '2025-01-11_00-12-31', '2025-01-11_03-59-48', '2025-01-11_04-38-54', '2025-01-11_04-59-18', '2025-01-11_05-22-05', '2025-01-11_06-31-38', '2025-01-11_08-04-39', '2025-01-11_09-46-29', '2025-01-11_11-38-31', '2025-01-13_10-30-31', '2025-01-13_16-50-40', '2025-01-13_19-11-12', '2025-01-13_21-55-32', '2025-01-14_01-06-20', '2025-01-14_04-32-26', '2025-01-14_13-35-29', '2025-01-14_13-57-30', '2025-01-14_14-37-58', '2025-01-14_16-00-43', '2025-01-14_18-30-29', '2025-01-14_20-16-04', '2024-12-20_12-50-58', '2024-12-21_10-12-42', '2024-12-21_11-00-27', '2024-12-21_11-50-31']#gli ultimi quattro sono quelli con pitch >1


input_file = "/home/eleonora/3D_recon_DAPHNE/Data/cylindrical_phantom.npz" 

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
        print(f"Error executing command: {e}")
        
"""        
        


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
