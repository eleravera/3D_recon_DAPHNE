import sys
sys.path.append("../")
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def read_log_file(log_file_path):
    base_directory = os.path.dirname(log_file_path)

    # Array per ogni colonna
    iterations = []
    pitch = []
    views = []
    pixels = []
    input_files = []
    output_directories = []

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            # Salta righe vuote o commenti
            if line.strip() and not line.startswith("#"):
                # Divide la riga in colonne usando spazi e tabulazioni come separatori
                columns = line.split()

                # Assicurati che la riga abbia almeno 6 colonne
                if len(columns) >= 6:
                    # Aggiungi i valori alle rispettive liste
                    iterations.append(int(columns[5]))
                    pitch.append(float(columns[4]))
                    views.append(int(columns[2]))
                    pixels.append(int(columns[3]))
                    input_files.append(columns[1])
                    output_directories.append(os.path.join(base_directory, columns[0]))
                    
    iterations = np.array(iterations)
    pitch = np.array(pitch)
    views = np.array(views)
    pixels = np.array(pixels)
    input_files = np.array(input_files)
    output_directories = np.array(output_directories)
                                
    return iterations, pitch, views, pixels, input_files, output_directories


    
def read_output_statistics(folder_path):
    file_path = os.path.join(folder_path, "output_statistics.txt")

    if not os.path.exists(file_path):
        print(f"File non trovato: {file_path}")
        return None

    # Inizializza un dizionario per ogni colonna
    statistics = {
        "Iteration": [],
        "Diff Mean": [],
        "Diff Std Dev": [],
        "Rel Diff Mean": [],
        "Rel Diff Std Dev": [],
        "5%_Cnt": [],
        "5%_Tot": [],
        "5%_Percentage": [],
        "10%_Cnt": [],
        "10%_Tot": [],
        "10%_Percentage": [],
        "Spatial_res": []
    }

    # Leggi il file
    with open(file_path, 'r') as stats_file:
        for line in stats_file:
            if line.strip() and not line.startswith("#"):
                columns = line.split()
                
                # Aggiungi i valori nelle rispettive colonne del dizionario
                statistics["Iteration"].append(int(columns[0]))
                statistics["Diff Mean"].append(float(columns[1]))
                statistics["Diff Std Dev"].append(float(columns[2]))
                statistics["Rel Diff Mean"].append(float(columns[3]))
                statistics["Rel Diff Std Dev"].append(float(columns[4]))
                statistics["5%_Cnt"].append(int(columns[5]))
                statistics["5%_Tot"].append(int(columns[6]))
                statistics["5%_Percentage"].append(float(columns[7]))
                statistics["10%_Cnt"].append(int(columns[8]))
                statistics["10%_Tot"].append(int(columns[9]))
                statistics["10%_Percentage"].append(float(columns[10]))
                statistics["Spatial_res"].append(float(columns[11]))

    return statistics
    
    
    
    
    
    
def plot_vs_iteration(statistics, folder_path, metric, ylabel, label_array, label='', title=''):
    """
    Funzione per creare un plot di una metrica in funzione delle iterazioni.
    
    Parameters:
    - statistics: lista di dizionari contenenti i dati da plottare
    - folder_path: lista dei percorsi delle cartelle per le etichette delle curve
    - metric: il nome della metrica da plottare (come stringa)
    - ylabel: etichetta per l'asse delle ordinate (come stringa)
    - label_array: lista di etichette per ciascun dato (una per ogni cartella)
    - label: prefisso da aggiungere all'etichetta (stringa vuota per default)
    """
    plt.figure(figsize=(10, 6))
    
    # Per ogni cartella, plottiamo la metrica specificata
    for i, stats in enumerate(statistics):
        # Concatenate il prefisso `label` con l'etichetta corrispondente in `label_array[i]`
        current_label = label + "{:.2f}".format(label_array[i])
        plt.plot(stats["Iteration"], stats[metric], label=current_label)
    
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Iteration - {title}")
    plt.legend()
    plt.grid(True)



def plot_last_metrics(label_array, metrics, xlabel='', ylabel='', title=''):
    """
    Funzione per fare il plot di una metrica a partire dai valori estratti per ogni cartella.
    
    Parameters:
    - label_array: array contenente le etichette per le cartelle (ad esempio, pitch)
    - metrics: lista dei valori da plottare (es. last_5_percentages, last_10_percentages, etc.)
    - ylabel: etichetta per l'asse y
    - title: titolo del grafico
    - xlabel: etichetta per l'asse x (default è "Folder Index")
    """
    plt.figure(figsize=(10, 6))
    plt.plot(label_array, metrics, 'o')
   
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)



def calculate_sad(pixels, pitch):
    if len(pixels)==1:
        pixels = np.ones(len(pitch))*pixels
    
    l = 100 # mm = FOV
    return (l*l)/(pixels*pitch - np.sqrt(2)*l)


###################################################################################################################


PLOT_ITER = True
PLOT_ITER_SAD = False
PIXELS_LIST = [200, 400, 500]
VIEWS_LIST = [4, 8]
ITER = 50
COMPARE_EDGE = False
INPUT_FILE = '/home/eleonora/eFLASH_3D_Sim-build/Dose_Map_30mm/spettro_9MeV/doseDistribution.npz'

log_file_path = "/home/eleonora/3D_recon_DAPHNE/Reconstruction/log.txt"




last_5_percentages = []
last_10_percentages = []
last_spatial_res = []
last_diff_mean = []
source_axis_distance = []
legend = []


all_5_percentages = []
all_10_percentages = []

for VIEWS in VIEWS_LIST:
    for PIXEL in PIXELS_LIST: 

        iterations, pitch, views, pixels, input_files, folder_path = read_log_file(log_file_path)

        mask = (pixels == PIXEL) * (views == VIEWS) * (input_files == INPUT_FILE)
        iterations = iterations[mask]
        pitch = pitch[mask]
        views = views[mask]
        pixels = pixels[mask]
        folder_path = folder_path[mask]
        input_files = input_files[mask]
        sad = calculate_sad(pixels, pitch)

        statistics = []
        for folder in folder_path:
            stats = read_output_statistics(folder)
            if stats==None: 
                print(stats)
                continue
            else:
                statistics.append(stats)


        percentages_5 = [stats['5%_Percentage'][-1] for stats in statistics]
        percentages_10 = [stats['10%_Percentage'][-1] for stats in statistics]
        sp_res = [stats['Spatial_res'][-1] for stats in statistics]
        diff_mean = [stats['Rel Diff Mean'][-1] for stats in statistics]


        last_5_percentages.append(np.array(percentages_5))
        last_10_percentages.append(np.array(percentages_10))
        last_spatial_res.append(np.array(sp_res))
        last_diff_mean.append(np.array(diff_mean))
        source_axis_distance.append(sad)
        legend.append(f'{PIXEL} pixels, {VIEWS} views')
        
        
        all_5_percentages.append(np.array(stats['5%_Percentage']))
        all_10_percentages.append(np.array(stats['10%_Percentage']))
        

        if PLOT_ITER_SAD: 
            plot_vs_iteration(statistics, folder_path, "Rel Diff Mean", "Rel Diff Mean", calculate_sad(pixels, pitch), 'sad [mm] ', f'{PIXEL} pixels, {VIEWS} views')
            plot_vs_iteration(statistics, folder_path, "5%_Percentage", "5% Percentage", calculate_sad(pixels, pitch), 'sad [mm] ', f'{PIXEL} pixels, {VIEWS} views')
            plot_vs_iteration(statistics, folder_path, "10%_Percentage", "10% Percentage", calculate_sad(pixels, pitch), 'sad [mm] ', f'{PIXEL} pixels, {VIEWS} views')
            plot_vs_iteration(statistics, folder_path, "Spatial_res", "Spatial Resolution", calculate_sad(pixels, pitch), 'sad [mm] ', f'{PIXEL} pixels, {VIEWS} views')

            plot_last_metrics(pitch, percentages_10, 'Pitch [mm]', '10% Percentage', f'{PIXEL} pixels, {VIEWS} views, {ITER} Iter')
            plot_last_metrics(pitch, sp_res, 'Pitch [mm]', 'Spatial resolution', f'{PIXEL} pixels, {VIEWS} views, {ITER} Iter')
            plot_last_metrics(pitch, diff_mean, 'Pitch [mm]', '$\mu$ relative', f'{PIXEL} pixels, {VIEWS} views, {ITER} Iter')
            
            plot_last_metrics(sad, percentages_5, 'sad [mm]', '5% Percentage', f'{PIXEL} pixels, {VIEWS} views, {ITER} Iter')
            plot_last_metrics(sad, percentages_10, 'sad [mm]', '10% Percentage', f'{PIXEL} pixels, {VIEWS} views, {ITER} Iter')
            plot_last_metrics(sad, sp_res, 'sad [mm]', 'Spatial resolution', f'{PIXEL} pixels, {VIEWS} views, {ITER} Iter')
            plot_last_metrics(sad, diff_mean, 'sad [mm]', '$\mu$ relative', f'{PIXEL} pixels, {VIEWS} views, {ITER} Iter')



if PLOT_ITER:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(0, len(all_5_percentages)):
        plt.plot(np.linspace(0,49, 50), all_5_percentages[i], label=legend[i])
    plt.legend()
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("5% Percentage")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(0, len(all_10_percentages)):
        plt.plot(np.linspace(0,49, 50), all_10_percentages[i], label=legend[i])
    plt.legend()
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("10% Percentage")
    plt.show()


fig_5, ax_5 = plt.subplots(figsize=(10, 6))
fig_10, ax_10 = plt.subplots(figsize=(10, 6))
fig_res, ax_res = plt.subplots(figsize=(10, 6))
fig_mu, ax_mu = plt.subplots(figsize=(10, 6))

for i in range(0, len(source_axis_distance)):
    sorted_indices = np.argsort(source_axis_distance[i])
    source_axis_distance[i] = source_axis_distance[i][sorted_indices]
    last_5_percentages[i] = last_5_percentages[i][sorted_indices]
    last_10_percentages[i] = last_10_percentages[i][sorted_indices]
    last_spatial_res[i] = last_spatial_res[i][sorted_indices]        
    last_diff_mean[i] = last_diff_mean[i][sorted_indices]        
    

for i in range(0, len(last_5_percentages)):
    ax_5.plot(source_axis_distance[i], last_5_percentages[i], '-o', label = legend[i])
    ax_10.plot(source_axis_distance[i], last_10_percentages[i], '-o', label = legend[i])    
    ax_res.plot(source_axis_distance[i], last_spatial_res[i], '-o', label = legend[i])
    ax_mu.plot(source_axis_distance[i], last_diff_mean[i], '-o', label = legend[i])
ax_5.set(xlabel='sad [mm]', ylabel='5%', title='')
ax_5.legend()
ax_5.grid(True)

ax_10.set(xlabel='sad [mm]', ylabel='10%', title='')
ax_10.legend()
ax_10.grid(True)

ax_res.set(xlabel='sad [mm]', ylabel='sp resl', title='')
ax_res.legend()
ax_res.grid(True)

ax_mu.set(xlabel='sad [mm]', ylabel='mu', title='')
ax_mu.legend()
ax_mu.grid(True)





if COMPARE_EDGE == True: 
    data = [
        {"directory": "/2025-01-15_10-01-27", "sigma": 2},
        {"directory": "/2025-01-15_18-50-34", "sigma": 4},
        {"directory": "/2025-01-15_21-07-25", "sigma": 6},
        {"directory": "/2025-01-15_23-28-45", "sigma": 8}
    ]

    statistics = []
    sigma = []
    for entry in data:  
        folder = os.path.dirname(log_file_path) + entry["directory"]
        s = entry["sigma"]
        print(f"Processing folder: {folder}, with sigma: {s}")
        stats = read_output_statistics(folder) 
        statistics.append(stats)
        sigma.append(s)


    percentages_5 = [stats['5%_Percentage'][-1] for stats in statistics]
    percentages_10 = [stats['10%_Percentage'][-1] for stats in statistics]
    sp_res = [stats['Spatial_res'][-1] for stats in statistics]
    diff_mean = [stats['Rel Diff Mean'][-1] for stats in statistics]
    
    
    fig_5, ax_5 = plt.subplots(figsize=(10, 6))
    fig_10, ax_10 = plt.subplots(figsize=(10, 6))
    fig_res, ax_res = plt.subplots(figsize=(10, 6))
    fig_mu, ax_mu = plt.subplots(figsize=(10, 6))

    ax_5.plot(sigma, percentages_5, '-o')
    ax_5.set(xlabel='sigma', ylabel='5%', title='')
    ax_5.grid(True)

    ax_10.plot(sigma, percentages_10, '-o')
    ax_10.set(xlabel='sigma', ylabel='10%', title='')
    ax_10.grid(True)

    ax_res.plot(sigma, sp_res, '-o')
    ax_res.set(xlabel='sigma', ylabel='sp resl', title='')
    ax_res.grid(True)

    ax_mu.plot(sigma, diff_mean, '-o')
    ax_mu.set(xlabel='sigma', ylabel='mu', title='')
    ax_mu.grid(True)
    plt.show()
    
    



plt.show()    
