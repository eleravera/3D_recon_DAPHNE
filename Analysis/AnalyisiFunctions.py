import os
import numpy as np
from matplotlib import pyplot as plt
from Misc.Preview import Visualize3dImage

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

def check_signal_equality(matrix1, matrix2):
    """
    Controlla se la somma del segnale nelle due distribuzioni è uguale.
    """
    signal1 = np.sum(matrix1)
    signal2 = np.sum(matrix2)
    
    print('signal1, signal2: ', signal1, signal2)
    return np.isclose(signal1, signal2)
    
    

def normalize_matrix(matrix):
    """
    Normalizza la matrice rispetto al suo segnale totale.
    """
    total_signal = np.sum(matrix)
    if total_signal == 0:
        raise ValueError("Il segnale totale della matrice è zero. Normalizzazione impossibile.")
    return matrix / total_signal


def normalize_slices(matrix):
    """
    Normalizza ogni slice della matrice tridimensionale in modo che il segnale totale di ciascuna slice sia 1.
    """
    if matrix.ndim != 3:
        raise ValueError("La matrice deve essere tridimensionale.")
    
    normalized_matrix = np.zeros_like(matrix)
    
    for i in range(matrix.shape[2]):
        slice = matrix[i]
        total_signal = np.sum(slice)
        
        if total_signal == 0:
            print(f"Attenzione: la slice {i} ha un segnale totale pari a zero. Normalizzazione non eseguita.")
            continue
        
        normalized_matrix[i] = slice / total_signal
        
    return normalized_matrix


def calculate_sigma(matrix):
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





def PlotTotalSignalHistograms(img1, img2, bins=50, labels=("Image 1", "Image 2"), title="Signal Histogram", colors=("blue", "orange")):
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
    plt.xlabel("Signal")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
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
        ratio[~np.isfinite(ratio)] = zero_division  # Replace inf and NaN with the specified value
    
    return ratio
