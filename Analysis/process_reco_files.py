#python -i process_reco_files.py /home/eleonora/3D_recon_DAPHNE/Data/cylindrical_phantom.npz /home/eleonora/3D_recon_DAPHNE/Reconstruction/2024-12-21_12-43-27/ --do_Gamma --criteria 1. 1. 1.

import sys
sys.path.append("../")
import argparse
import os
import numpy as np
import time 
import Analysis.AnalyisiFunctions as Analysis


def process_data(input_matrix, reconstructed_matrix, normalization_type='slice'):
    """
    Elabora due matrici tridimensionali: una input e una ricostruita, senza generare plot.

    Args:
        input_matrix (numpy.ndarray): Matrice di input originale.
        reconstructed_matrix (numpy.ndarray): Matrice ricostruita.
        normalization_type (str): Tipo di normalizzazione ('slice' o 'global').
    Returns:
        dict: Un dizionario contenente le statistiche calcolate e le matrici di output.
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

    # Calcolo differenza
    difference = normalized_reconstructed - normalized_input
    difference_stats = Analysis.calculate_difference_stats(difference)

    # Calcolo rapporto relativo
    rel_diff_ = Analysis.ComputeMatrixRatio(difference, normalized_input)
    rel_diff = Analysis.preprocess_for_visualization(rel_diff_)
    relative_difference_stats = Analysis.calculate_difference_stats(rel_diff_)

    # Statistiche dei range
    stats_range_5 = Analysis.range_statistics(rel_diff_, -0.05, 0.05)
    stats_range_10 = Analysis.range_statistics(rel_diff_, -0.1, 0.1)

    # Output
    results = {
        "difference_stats": {
            "mean": difference_stats[0],
            "std_dev": difference_stats[1],
        },
        "relative_difference_stats": {
            "mean": relative_difference_stats[0],
            "std_dev": relative_difference_stats[1],
        },
        "range_statistics": {
            "5_percent": {
                "count": stats_range_5[0],
                "total": stats_range_5[1],
                "percentage": stats_range_5[2],
            },
            "10_percent": {
                "count": stats_range_10[0],
                "total": stats_range_10[1],
                "percentage": stats_range_10[2],
            },
        },
    }

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elabora i dati di ricostruzione e salva le statistiche.")
    parser.add_argument('input_file', type=str, help="Percorso del file di input contenente la matrice originale (ad esempio, 'cylindrical_phantom.npz').")
    parser.add_argument('recon_dir', type=str, help="Directory contenente i file di ricostruzione.")
    parser.add_argument('--do_Gamma', action='store_true', help="Lancia la gamma analisi. (Default: False).")
    parser.add_argument('--criteria', nargs=3, type=float, metavar=('DOSE', 'DISTANCE', 'THRESHOLD'),
                        help="Specifica i criteri come 3 valori: percentuale in dose, distanza in mm, soglia di dose.")
    args = parser.parse_args()    

    input_matrix = np.load(args.input_file)['doseDistr']  
    latest_recon_file = os.path.join(args.recon_dir, 'reco.npz') if os.path.isfile(os.path.join(args.recon_dir, 'reco.npz')) else None
    
    iteration_number = 0
    output_file = args.recon_dir + "output_statistics.txt"
    
    with open(output_file, 'w') as file:
        file.write("#Iteration \tDiff Mean \tDiff Std Dev \tRel Diff Mean \tRel Diff Std Dev\t5%_Cnt\t5%_Tot\t5%_Percentage\t10%_Cnt\t10%_Tot\t10%_Percentage\tSpatial_res\n")
    
    while True:
        try:
            reconstructed_matrix = Analysis.load_specific_iteration_matrix(latest_recon_file, iteration_number)
            
            # Esegui il processo e visualizza i risultati
            print(f"\nElaborazione per l'iterazione {iteration_number}")
            results = process_data(
                input_matrix,
                reconstructed_matrix,
                normalization_type='global'
            )

            try:
                sp_res = []
                i_list = []
                for i in range(10, 90):
                    sp_res.append(Analysis.calculate_spatial_resolution(reconstructed_matrix, i, 15, 25, np.max))
                    sp_res.append(-Analysis.calculate_spatial_resolution(reconstructed_matrix, i, 75, 85, np.min))
                    i_list.append(i)
                    i_list.append(i)
                spatial_resolution = np.mean(sp_res)
            except Exception as e:
                print(f"Errore durante il calcolo della risoluzione spaziale: {e}")
                spatial_resolution = -1

            # Estrai le statistiche dai risultati
            difference_stats = results["difference_stats"]
            relative_difference_stats = results["relative_difference_stats"]
            range_stats_5 = results["range_statistics"]["5_percent"]
            range_stats_10 = results["range_statistics"]["10_percent"]
                    
            with open(output_file, 'a') as file:
                # Scrivi i risultati nel file
                file.write(f"{iteration_number}\t\t"
                           f"{difference_stats['mean']:.2e}\t\t{difference_stats['std_dev']:.2e}\t\t"
                           f"{relative_difference_stats['mean']:.2e}\t\t{relative_difference_stats['std_dev']:.2e}\t\t"
                           f"{range_stats_5['count']}\t\t{range_stats_5['total']}\t\t{range_stats_5['percentage']:.2f}\t\t"
                           f"{range_stats_10['count']}\t\t{range_stats_10['total']}\t\t{range_stats_10['percentage']:.2f}\t"
                           f"{spatial_resolution:.2f}\n")
                    
            iteration_number += 1

        except KeyError:
            # Esci dal ciclo quando non ci sono più iterazioni
            print(f"Iterazione {iteration_number} non trovata. Fine del ciclo.")
            break



    if args.do_Gamma:
        if not hasattr(args, 'criteria') or args.criteria is None:
            # Usa i valori di default se --criteria non è specificato
            criteria = [3, 3, 0.1]  # Default
            print(f"--do_Gamma è abilitato con criteria di default: {criteria}.")
        else:
            # Controlla se criteria ha esattamente 3 valori
            if len(args.criteria) == 3:
                criteria = args.criteria
                print(f"--do_Gamma è abilitato con criteria specificati: {criteria}.")
            else:
                print("Errore: criteria deve avere esattamente 3 valori.")
            
        
        spacing = [1.0, 1.0, 1.0]  # Spaziatura in mm (es. voxel isotropici)
        
        gamma_map, passing_rate = Analysis.gamma_analysis_3d(input_matrix, reconstructed_matrix, spacing, dose_crit=criteria[0], dist_crit=criteria[1], dose_threshold=criteria[2])
        print("Percentuale di punti conformi:", passing_rate, "%")

        output_file = args.recon_dir + 'gamma_analysis.npz'   
        Analysis.save_matrix_to_npz(output_file, gamma_map, str(criteria))

