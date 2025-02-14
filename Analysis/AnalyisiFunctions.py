import os
import numpy as np
from matplotlib import pyplot as plt
from Misc.Preview import Visualize3dImage
import time

def find_latest_recon_file(recon_dir):
    """
    Trova il file di ricostruzione più recente nella directory specificata.
    """
    recon_files = [f for f in os.listdir(recon_dir) if f.endswith('.npz')]
    if not recon_files:
        raise FileNotFoundError("Nessun file .npz trovato nella directory specificata.")
    recon_files.sort(key=lambda f: os.path.getmtime(os.path.join(recon_dir, f)), reverse=True)
    latest_file = os.path.join(recon_dir, recon_files[0])
    return latest_file

def load_highest_iteration_matrix(recon_file):
    """
    Carica il file di ricostruzione e trova la matrice con l'iterazione più alta.
    """
    with np.load(recon_file) as data:
        iter_keys = [k for k in data.keys() if k.startswith("iter")]
        if not iter_keys:
            raise KeyError("Nessuna chiave iterativa ('iterN') trovata nel file.")
        iter_keys.sort(key=lambda k: int(k[4:]), reverse=True)
        highest_iter_key = iter_keys[0]
        matrix = data[highest_iter_key]
        return matrix

def load_specific_iteration_matrix(recon_file, iteration_number):
    """
    Carica una matrice di una specifica iterazione da un file di ricostruzione.

    Args:
        recon_file (str): Percorso al file di ricostruzione (.npz).
        iteration_number (int): Numero dell'iterazione desiderata.

    Returns:
        numpy.ndarray: Matrice dell'iterazione specificata.

    Raises:
        KeyError: Se la chiave dell'iterazione specificata non esiste nel file.
    """
    with np.load(recon_file) as data:
        iter_key = f"iter{iteration_number}"
        if iter_key not in data:
            raise KeyError(f"La chiave per l'iterazione {iteration_number} ('{iter_key}') non è presente nel file.")
        matrix = data[iter_key]
        return matrix


def save_matrix_to_npz(file_path, matrix, key):
    """
    Salva una matrice in un file .npz con una certa chiave.
    Se il file esiste, aggiunge la matrice al file.
    Se il file non esiste, lo crea.

    :param file_path: Percorso del file .npz
    :param matrix: Matrice da salvare
    :param key: Chiave con cui salvare la matrice
    """
    # Verifica se il file esiste
    if os.path.exists(file_path):
        # Carica i dati esistenti
        existing_data = np.load(file_path)
        data_dict = {k: existing_data[k] for k in existing_data.keys()}
        existing_data.close()  # Chiude il file dopo il caricamento
    else:
        # Se il file non esiste, inizializza un dizionario vuoto
        data_dict = {}

    # Aggiungi la matrice con la chiave specificata
    if key in data_dict:
        print(f"Avviso: La chiave '{key}' esiste già nel file. Sovrascrivendo.")
    data_dict[key] = matrix

    # Salva tutti i dati nel file (sovrascrivendo se necessario)
    np.savez(file_path, **data_dict)
    print(f"Matrice salvata con chiave '{key}' nel file '{file_path}'")
    


def check_signal_equality(matrix1, matrix2):
    """
    Controlla se la somma del segnale nelle due distribuzioni è uguale.
    """
    signal1 = np.sum(matrix1)
    signal2 = np.sum(matrix2)
    tolerance = 1.e-4
    print('signal1, signal2: ', signal1, signal2)
    print('Differenza: ', signal1 - signal2)
    
    # Confronto con tolleranza relativa
    is_equal = np.isclose(signal1, signal2, rtol=tolerance)
    print('Sono uguali secondo np.isclose con tolleranza', tolerance, '?:', is_equal)
    
    return is_equal
    
    
    
def normalize_matrix(matrix, per_slice=False):
    """
    Normalizza una matrice tridimensionale.

    Args:
        matrix (numpy.ndarray): La matrice da normalizzare.
        per_slice (bool): Se True, normalizza ogni slice indipendentemente. 
                          Altrimenti, normalizza globalmente.

    Returns:
        numpy.ndarray: La matrice normalizzata.
    """
    if matrix.ndim != 3:
        raise ValueError("La matrice deve essere tridimensionale.")

    if per_slice:
        # Normalizzazione per slice
        normalized_matrix = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):  # Considero la prima dimensione per le slices
            slice = matrix[i]
            total_signal = np.sum(slice)
            if total_signal == 0:
                print(f"Attenzione: la slice {i} ha un segnale totale pari a zero. Normalizzazione non eseguita.")
                continue
            normalized_matrix[i] = slice / total_signal
        # Gestire i NaN dopo la normalizzazione
        normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.0)  # Impostiamo NaN a 0
        return normalized_matrix
    else:
        # Normalizzazione globale
        total_signal = np.sum(matrix)
        if total_signal == 0:
            raise ValueError("Il segnale totale della matrice è zero. Normalizzazione impossibile.")
        normalized_matrix = matrix / total_signal
        # Gestire i NaN dopo la normalizzazione
        normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.0)  # Impostiamo NaN a 0
        return normalized_matrix


def preprocess_for_visualization(matrix):
    """
    Pre-elabora la matrice per la visualizzazione, sostituendo NaN e Inf con zero.
    
    Args:
        matrix (numpy.ndarray): La matrice da pre-processare.
    
    Returns:
        numpy.ndarray: La matrice pre-elaborata.
    """
    matrix = np.nan_to_num(matrix, nan=0.0)  # Sostituisce NaN con 0
    matrix[np.isinf(matrix)] = 0  # Sostituisce Inf con 0
    return matrix


def calculate_difference_stats(difference):
    """
    Calcola la media e la deviazione standard della differenza tra due immagini,
    ignorando i valori NaN.
    
    Args:
        difference (np.ndarray): La differenza tra due immagini come array NumPy,
                                 che può contenere valori NaN.
    
    Returns:
        tuple: Una tupla contenente (media, deviazione standard) della differenza,
               oppure None se la differenza contiene solo NaN.
    """
    # Verifica se ci sono valori validi nella differenza
    if np.any(~np.isnan(difference)):
        mean_diff = np.nanmean(difference)  # Calcola la media ignorando i NaN
        std_diff = np.nanstd(difference)   # Calcola sigma ignorando i NaN
        return mean_diff, std_diff
    else:
        print("Attenzione: la differenza contiene solo valori NaN!")
        return None



def calculate_sigma_per_slice(matrix):
    """
    Calcola la deviazione standard per ogni slice lungo i tre assi della matrice tridimensionale.
    Restituisce un array 3xN dove N è il numero di slice per ogni asse.
    """
    # Verifica che la matrice sia tridimensionale
    if matrix.ndim != 3:
        raise ValueError("La matrice deve essere tridimensionale.")
    
    # Calcola la deviazione standard lungo ciascun asse
    sigma_0 = np.std(matrix, axis=(1, 2))  # Deviatore lungo l'asse 0
    sigma_1 = np.std(matrix, axis=(0, 2))  # Deviatore lungo l'asse 1
    sigma_2 = np.std(matrix, axis=(0, 1))  # Deviatore lungo l'asse 2
    
    # Organizza il risultato in un array 3xN
    sigma = np.array([sigma_0, sigma_1, sigma_2])
    
    return sigma

def plot_sigma(sigma):
    """
    Crea un plot con le deviazioni standard calcolate lungo i tre assi.
    """
    # Crea un grafico
    plt.figure(figsize=(10, 6))
    
    # Etichette per gli assi
    axes = ['Asse 0', 'Asse 1', 'Asse 2']
    
    # Plot per ciascun asse
    for i in range(3):
        plt.plot(sigma[i], '-', label=f'Sigma {axes[i]}', lw=2)
    
    # Aggiungi etichette e titolo
    plt.xlabel('Indice dello slice')
    plt.ylabel('Deviazione standard')
    plt.title('Deviazione standard lungo i tre assi')
    plt.legend()
    plt.grid(True)





def plot_signal_histograms(img1, img2, bins=50, labels=("Image 1", "Image 2"), title="Signal Histogram", colors=("blue", "orange")):
    """
    Plots histograms of the total signal for two 3D images.

    Parameters:
        img1 (ndarray): First 3D image array.
        img2 (ndarray): Second 3D image array.
        bins (int): Number of bins for the histogram.
        labels (tuple): Labels for the two images (default: "Image 1", "Image 2").
        title (str): Title of the plot.
        colors (tuple): Colors for the two histograms (default: "blue", "orange").
    """
    # Flatten the images to calculate total signals per slice
    signal1 = np.concatenate(np.concatenate(img1))  # Signal per slice for img1
    signal2 = np.concatenate(np.concatenate(img2)) # Signal per slice for img2
    
    # Plot histograms
    plt.figure(figsize=(8, 6))
    plt.hist(signal1, bins=bins, color=colors[0], alpha=0.7, label=labels[0])
    plt.hist(signal2, bins=bins, color=colors[1], alpha=0.7, label=labels[1])
    #threshold = 0.1 * signal2.max()
    #plt.axvline(x=threshold, color='red', linestyle='--', label="Threshold 10%")
    plt.xlabel("Signal")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()



def plot_difference_histograms(difference, relative_difference, bins=100):
    """
    Crea un plot con due istogrammi affiancati:
    uno per la difference e uno per la relative difference.
    
    Args:
        difference (np.ndarray): Array 3D della differenza assoluta.
        relative_difference (np.ndarray): Array 3D della differenza relativa.
        bins (int): Numero di bin per gli istogrammi. Default: 100.
    """
    # Flatten degli array per creare gli istogrammi (escludendo i NaN)
    diff_flattened = difference[~np.isnan(difference)].flatten()
    rel_diff_flattened = relative_difference[~np.isnan(relative_difference)].flatten()
    
    # Maschera della relative_difference (escludere NaN in rel_diff)
    #masked_diff = difference[~np.isnan(relative_difference)].flatten()
    
    
    # Creazione del plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Istogramma della difference
    axes[0].hist(diff_flattened, bins=bins, color='blue', histtype='step', label='Difference')
    #axes[0].hist(masked_diff, bins=bins, color='orange', histtype='step', label='Masked Difference')
    axes[0].set_title("Histogram of Difference")
    axes[0].set_xlabel("Difference")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Istogramma della relative difference
    axes[1].hist(rel_diff_flattened, bins=bins, color='green', histtype='step', label='Relative Difference')
    #axes[1].hist(masked_diff, bins=bins, color='orange', histtype='step', label='Masked Relative Difference')    
    axes[1].set_title("Histogram of Relative Difference")
    axes[1].set_xlabel("Relative Difference")
    axes[1].legend()

    # Mostra il plot
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[1].grid(True, linestyle="--", alpha=0.7)    
    plt.tight_layout()


def ComputeMatrixRatio(matrix1, matrix2, zero_division=np.nan):
    """
    Compute the element-wise ratio of two matrices with robust handling of division by zero.
    
    Parameters:
        matrix1 (ndarray): Numerator matrix.
        matrix2 (ndarray): Denominator matrix.
        zero_division: Value to assign when division by zero occurs (default is np.nan).
        
    Returns:
        ndarray: A matrix representing the element-wise ratio of matrix1 to matrix2.
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("The two matrices must have the same dimensions.")
    
    # Initialize the ratio matrix
    ratio = np.empty_like(matrix1, dtype=np.float64)
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(matrix1, matrix2)
        ratio[~np.isfinite(ratio)] = np.nan  # Assign NaN where division by zero occurs
    
    return ratio


def calculate_spatial_resolution(matrix, i, side, index_limit, zero_func):
    """
    Calcola la risoluzione spaziale basata sul profilo in una matrice.
    
    Parameters:
        matrix (np.ndarray): Matrice di input.
        i (int): Indice della prima dimensione.
        side (int): Indice di inizio del profilo.
        index_limit (int): Limite superiore per il profilo.
        zero_func (callable): Funzione per calcolare l'indice dello zero.

    Returns:
        int: Risoluzione spaziale del profilo.
    """
    profile = matrix[i, 50, side:index_limit]
    profile[profile < 0.005] = 0.0
    max_idx = np.argmax(profile)
    zero_idx = zero_func(np.where(profile == 0.0)[0])
    return max_idx - zero_idx
    
    
def range_statistics(matrix, lower_bound, upper_bound):
    """
    Calcola e stampa il numero di elementi in un intervallo specificato e la loro percentuale.

    Args:
        matrix (numpy.ndarray): Matrice da analizzare.
        lower_bound (float): Limite inferiore dell'intervallo.
        upper_bound (float): Limite superiore dell'intervallo.
    """
    count_in_range = np.sum((lower_bound <= matrix) & (matrix <= upper_bound))
    total_events = np.sum(~np.isnan(matrix))
    percentage = (count_in_range / total_events) * 100
    return count_in_range, total_events, percentage




def gamma_analysis_3d(reference, verification, spacing, dose_crit, dist_crit, dose_threshold):
    """
    Implementazione della gamma analysis per distribuzioni 3D.
    
    Parameters:
    - reference: matrice 3D della dose di riferimento.
    - verification: matrice 3D della dose da verificare.
    - spacing: dimensione del voxel (es. [dx, dy, dz] in mm).
    - dose_crit: criterio di dose (es. 3% della dose massima).
    - dist_crit: criterio di distanza (es. 3 mm).
    - dose_threshold: soglia di dose come frazione (es. 0.10 per 10% della dose massima).
    
    Returns:
    - gamma_map: mappa 3D dell'indice gamma.
    - passing_rate: percentuale di punti conformi (\( \gamma \leq 1 \)).
    """
    # Calcolo dei parametri
    max_dose = np.max(reference)
    dose_threshold *= max_dose
    dose_crit_abs = dose_crit * max_dose / 100

    # Crea una griglia di coordinate per la distribuzione
    x = np.arange(reference.shape[0]) * spacing[0]
    y = np.arange(reference.shape[1]) * spacing[1]
    z = np.arange(reference.shape[2]) * spacing[2]
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')

    # Esclude i punti sotto la soglia
    mask_reference = reference >= dose_threshold
    mask_verification = verification >= dose_threshold
    valid_mask = mask_verification  # Maschera per punti validi
    
    #sum_valid_mask= np.sum(valid_mask)
    #print('THE SUM OF VALID MASK IS: ', sum_valid_mask)

    # Inizializza la mappa gamma
    gamma_map = np.full(verification.shape, np.inf)
    
    dose_contribution_map = np.full(verification.shape, np.inf)
    spatial_contribution_map = np.full(verification.shape, np.inf)    
    
    # Itera sui punti di verifica per mantenere la logica della funzione originale
    for i in range(verification.shape[0]):
        start_time = time.time()
        
        for j in range(verification.shape[1]):
            for k in range(verification.shape[2]):
                # Esclude i punti sotto la soglia
                if reference[i, j, k] < dose_threshold or verification[i, j, k] < dose_threshold:
                    continue

                # Differenza di dose e distanza
                delta_dose = np.abs(reference - verification[i, j, k])
                delta_x = xv - xv[i, j, k]
                delta_y = yv - yv[i, j, k]
                delta_z = zv - zv[i, j, k]
                delta_dist = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)


                # Calcola gamma per ogni punto
                gamma = np.sqrt((delta_dose / dose_crit_abs)**2 + (delta_dist / dist_crit)**2)
                min_value = np.min(gamma) # Prendi il minimo solo sui punti validi     
                gamma_map[i, j, k] = min_value 
                
                min_index = np.argwhere(gamma == min_value)[0]

                
                #if delta_dist[tuple(min_index)] > 0: 
                #   print(f'Spatial contribution is !=0')
                #print(f'delta_dose[tuple(min_index)]: {delta_dose[tuple(min_index)]}')
                #print(f'delta_dist[tuple(min_index)]: {delta_dist[tuple(min_index)]}')                
                dose_contribution_map[i, j, k] = delta_dose[tuple(min_index)] / dose_crit_abs
                spatial_contribution_map[i, j, k] = delta_dist[tuple(min_index)] / dist_crit
        print('slice i/100, time: ', i, time.time()-start_time)
                     

    # Calcola la percentuale di punti conformi
    passing_rate = np.sum(gamma_map <= 1) / np.sum(valid_mask) * 100

    return gamma_map, passing_rate, dose_contribution_map, spatial_contribution_map

