#python -i 3D_distributions.py /home/eleonora/3D_recon_DAPHNE/Data/cylindrical_phantom.npz /home/eleonora/3D_recon_DAPHNE/Reconstruction/2024-12-21_12-43-27/ --sigma 2

import sys
sys.path.append("../")
import argparse
import ast
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from Misc.Preview import Visualize3dImage, Visualize3dImageWithProfile, Visualize3dImageWithProfileAndHistogram
import Analysis.AnalyisiFunctions as Analysis


def process_and_visualize(input_matrix, reconstructed_matrix, threshold, normalization_type='slice', description="", do_plot=True):
    """
    Elabora e visualizza due matrici tridimensionali: una input e una ricostruita.

    Args:
        input_matrix (numpy.ndarray): Matrice di input originale.
        reconstructed_matrix (numpy.ndarray): Matrice ricostruita.
        normalization_type (str): Tipo di normalizzazione ('slice' o 'global').
        description (str): Descrizione per i titoli delle visualizzazioni.
        do_plot (bool): Se True, genera i plot; altrimenti li salta.
    """
    # Normalizzazione
    if normalization_type == 'slice':
        normalized_input = Analysis.normalize_matrix(input_matrix, per_slice=True)
        normalized_reconstructed = Analysis.normalize_matrix(reconstructed_matrix, per_slice=True)
    elif normalization_type == 'global':
        normalized_input = Analysis.normalize_matrix(input_matrix, per_slice=False)
        normalized_reconstructed = Analysis.normalize_matrix(reconstructed_matrix, per_slice=False)
    else:
        raise ValueError(f"Tipo di normalizzazione '{normalization_type}' non riconosciuto.")

    # Pre-elaborazione per gestire NaN e Inf
    normalized_input = Analysis.preprocess_for_visualization(normalized_input)
    normalized_reconstructed = Analysis.preprocess_for_visualization(normalized_reconstructed)

    Analysis.check_signal_equality(normalized_input, normalized_reconstructed)

    # Calcolo differenza e rapporto relativo
    difference = normalized_reconstructed - normalized_input
    stats = Analysis.calculate_difference_stats(difference)
    print(f"\nMedia della differenza: {stats[0]}")
    print(f"Deviazione standard della differenza: {stats[1]}")
    
    
    rel_diff_ = Analysis.ComputeMatrixRatio(difference, normalized_input)
    rel_diff = Analysis.preprocess_for_visualization(rel_diff_)
    stats = Analysis.calculate_difference_stats(rel_diff_)
    print(f"\nMedia della differenza relativa: {stats[0]}")
    print(f"Deviazione standard della differenza relativa: {stats[1]}")

    # Statistiche
    stats = Analysis.range_statistics(rel_diff_, -0.05, 0.05)
    print(f"\nEventi nel range [-5%, 5%]: {stats[0]} su {stats[1]} = {stats[2]:.2f}%")

    stats = Analysis.range_statistics(rel_diff_, -0.1, 0.1)
    print(f"\nEventi nel range [-10%, 10%]: {stats[0]} su {stats[1]} = {stats[2]:.2f}%")
    
    # Visualizzazione (se do_plot è True)
    if do_plot:
        Visualize3dImage(normalized_input, slice_axis=0, title=f"Input {description}")
        Visualize3dImage(normalized_reconstructed, slice_axis=0, title=f"Ricostruita {description}")
        Visualize3dImageWithProfile(normalized_reconstructed, slice_axis=2, profile_axis=1, title='')
        Visualize3dImageWithProfile(difference, slice_axis=0, profile_axis=0, symmetric_colorbar=True, title=f"Differenza {description}")
        Visualize3dImageWithProfile(difference, slice_axis=2, profile_axis=0, symmetric_colorbar=True, title=f"Differenza {description}")
        Visualize3dImageWithProfile(rel_diff, slice_axis=0, profile_axis=0, symmetric_colorbar=True, title=f"Rapporto Relativo {description}")
        Visualize3dImageWithProfile(rel_diff, slice_axis=2, profile_axis=0, symmetric_colorbar=True, title=f"Rapporto Relativo {description}")

        Analysis.plot_signal_histograms(normalized_input, normalized_reconstructed, bins=100, labels=("input", "recon"), title="Signal Histogram", colors=("blue", "orange"))

        Visualize3dImageWithProfileAndHistogram(difference, slice_axis=0, profile_axis=0, symmetric_colorbar=True, title=f"Difference {description}")
        Visualize3dImageWithProfileAndHistogram(rel_diff_, slice_axis=0, profile_axis=0, symmetric_colorbar=True, title=f"Rapporto Relativo {description}")
        threshold= normalized_reconstructed.max()*0.1
        masked_rel_diff = normalized_reconstructed > threshold
        print('Unmasked/Total = ', np.sum(masked_rel_diff)/10000)
        
        Analysis.plot_difference_histograms(difference, rel_diff_)

        threshold = normalized_reconstructed.max() * threshold
        mask_10percent = normalized_reconstructed > threshold
        rel_diff[~mask_10percent] = np.nan 
        Visualize3dImageWithProfile(rel_diff, slice_axis = 0, profile_axis = 0, symmetric_colorbar=True, title=f"Rapporto Relativo dove reco > 10% {description}")
        Visualize3dImage(mask_10percent, slice_axis=0)
        
        
        #plot pdd and profile
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.linspace(0.5, 99.5, 100),  normalized_input[:, 50, 50], label = 'Reference')
        ax.plot(np.linspace(0.5, 99.5, 100),  normalized_reconstructed[:, 50, 50], label = 'Reconstructed')
        ax.set(xlabel='Depth [mm]', ylabel='Signal [a.u.]', title ='')
        ax.grid()
        ax.legend()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.linspace(-49.5, 49.5, 100),  normalized_input[12, :, 50], label = 'Reference')
        ax.plot(np.linspace(-49.5, 49.5, 100),  normalized_reconstructed[12, :, 50], label = 'Reconstructed')
        ax.set(xlabel='Profile [mm]', ylabel='Signal [a.u.]', title ='')
        ax.grid()
        ax.legend()        

        
        
        return normalized_input, normalized_reconstructed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elabora i dati di ricostruzione e salva le statistiche.")
    parser.add_argument('input_file', type=str, help="Percorso del file di input contenente la matrice originale (ad esempio, 'cylindrical_phantom.npz').")
    parser.add_argument('recon_dir', type=str, help="Directory contenente i file di ricostruzione.")
    parser.add_argument('--sigma', type=int, default=None, help="Sigma for the Gaussian filter (must be an integer).")
    args = parser.parse_args()    

    input_matrix = np.load(args.input_file)['doseDistr'].astype(np.float64)
    if args.sigma is not None: 
        input_matrix = gaussian_filter(input_matrix, sigma = args.sigma/6, order = 0)

    latest_recon_file = os.path.join(args.recon_dir, 'reco.npz') if os.path.isfile(os.path.join(args.recon_dir, 'reco.npz')) else None
    reconstructed_matrix = Analysis.load_highest_iteration_matrix(latest_recon_file).astype(np.float64)
        
    _ = process_and_visualize(input_matrix, reconstructed_matrix, threshold=0.1, normalization_type='global', description="")
    

plt.show()
