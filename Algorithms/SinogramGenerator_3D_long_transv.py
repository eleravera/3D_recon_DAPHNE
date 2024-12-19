import numpy as np
from Algorithms.SiddonProjector import SiddonProjector
from Misc.Sinogram import Sinogram
from Geometry.ExperimentalSetup import Mode
from matplotlib import pyplot as plt

class SinogramGenerator_3D:
    """!@brief
       This class  generates a Sinogram of an image using the projection_extrema contained in the detector
    """

    def __init__(self, experimental_setup):
        self._experimental_setup = experimental_setup

    def GenerateObjectSinogram(self, img, transponse_image=0):
        """!@brief 
            Generate the Sinogram of a 2D image. The binning parameters are taken from the experimental setup  
        """

        sinogram_list = []
        self.sinogram = Sinogram(self._experimental_setup.pixels_per_slice_nb, 1,
                                 self._experimental_setup.detector_slice_nb)
        sinogram_list.append(self.sinogram)
        self.sinogram = Sinogram(self._experimental_setup.pixels_per_slice_nb, 
                                 self._experimental_setup.gantry_angles_nb,
                                 self._experimental_setup.detector_slice_nb)
        sinogram_list.append(self.sinogram)

        _img = img
        if transponse_image == 1:
            _img = np.transpose(img, axes=(1, 0, 2))
        else:
            _img = img

        Siddon = SiddonProjector(
            self._experimental_setup.image_matrix_size_mm,
            self._experimental_setup.voxel_size_mm,
        )
        StartPoints = self._experimental_setup._projections_extrema["p0"]
        EndPoints =   self._experimental_setup._projections_extrema["p1"]

        sino = sinogram_list[0]
        sino.image_matrix_size_mm = self._experimental_setup.image_matrix_size_mm
        sino.voxel_size_mm = self._experimental_setup.voxel_size_mm
        sino._radial_step_mm=self._experimental_setup._radial_step_mm
        sino.mode=self._experimental_setup.mode
        sino._voxel_nb=self._experimental_setup._voxel_nb
        sino._angular_step_deg=self._experimental_setup._angular_step_deg # that should be = angular_range_deg for gantry_angles_nb = 1
        sino.gantry_angles_nb=1
        sino.detector_slice_nb=self._experimental_setup.detector_slice_nb
        sino.slice_pitch_mm=self._experimental_setup.slice_pitch_mm
        if self._experimental_setup.mode != Mode.PARALLELBEAM:
              sino.sad_mm= self._experimental_setup.sad_mm
              sino.sdd_mm= self._experimental_setup.sdd_mm
        sino.detector_pitch_mm= self._experimental_setup._detector_pitch_mm
        sino.pixels_per_slice_nb=self._experimental_setup.pixels_per_slice_nb


        first_detector = int(sino.detector_slice_nb*sino.pixels_per_slice_nb) 

        print('Projection on first detector')

        my_list = [1702, 1762, 1822, 1882]
        perc0=perc1=0
        for i in range(0, first_detector):
              perc0=perc1
              perc1=np.trunc((i+1)/len(StartPoints)*100).astype(np.int32)
              if perc1 != perc0:
                  print("Projecting data, " + str(perc1) + "% done...", end="\r")
              TOR = Siddon.CalcIntersection(StartPoints[i], EndPoints[i])        
              mask = (TOR["vx"] >= 100) | (TOR["vy"] >= 100) | (TOR["vz"] >= 100)
            
              try:
                if np.any(mask):  # Se esistono valori sopra il limite
                    raise ValueError("Indici fuori dai limiti rilevati, impossibile calcolare proj.")
                
                # Calcola `proj` solo se non ci sono indici fuori dai limiti
                proj = np.sum(_img[TOR["vx"], TOR["vy"], TOR["vz"]] * TOR["prob"])
                sino.AddItem(self._experimental_setup._bins['s'][i],self._experimental_setup._bins[i]['theta'],self._experimental_setup._bins[i]['slice'], proj)
              
              except ValueError as e:
                if np.any(TOR["prob"][mask] > 1.e-13):
                    print(f"Errore durante il calcolo di proj: {e}")
                    print("Valori con indici >= 100:")
                    print("vx:", TOR["vx"][mask])
                    print("vy:", TOR["vy"][mask])
                    print("vz:", TOR["vz"][mask])
                    print("prob:", TOR["prob"][mask])
                    with open("/home/eleonora/3D_recon_DAPHNE/Reconstruction/TOR.txt", "a") as log_file:
                        log_file.write(f"TOR: {TOR}\n")
                        log_file.write(f"{_}")
                        log_file.write('\n')

        
        print('shape sino 0: ' , sinogram_list[0]._data.shape)


        sino = sinogram_list[1]
        sino.image_matrix_size_mm = self._experimental_setup.image_matrix_size_mm
        sino.voxel_size_mm = self._experimental_setup.voxel_size_mm
        sino._radial_step_mm=self._experimental_setup._radial_step_mm
        sino.mode=self._experimental_setup.mode
        sino._voxel_nb=self._experimental_setup._voxel_nb
        sino._angular_step_deg=self._experimental_setup._angular_step_deg # that should be = angular_range_deg for gantry_angles_nb = 1
        sino.gantry_angles_nb= self._experimental_setup.gantry_angles_nb
        sino.detector_slice_nb=self._experimental_setup.detector_slice_nb
        sino.slice_pitch_mm=self._experimental_setup.slice_pitch_mm
        if self._experimental_setup.mode != Mode.PARALLELBEAM:
              sino.sad_mm= self._experimental_setup.sad_mm
              sino.sdd_mm= self._experimental_setup.sdd_mm
        sino.detector_pitch_mm= self._experimental_setup._detector_pitch_mm
        sino.pixels_per_slice_nb=self._experimental_setup.pixels_per_slice_nb

        print('Projection on second detector')

        first_detector = int(sino.detector_slice_nb*sino.pixels_per_slice_nb) 
        perc0=perc1=0
        for i in range(0, int(first_detector*sino.gantry_angles_nb)):
              perc0=perc1
              perc1=np.trunc((i+1)/len(StartPoints)*100).astype(np.int32)
              if perc1 != perc0:
                  print("Projecting data, " + str(perc1) + "% done...", end="\r")
              TOR = Siddon.CalcIntersection(StartPoints[i+first_detector], EndPoints[i+first_detector])
            
              mask = (TOR["vx"] >= 100) | (TOR["vy"] >= 100) | (TOR["vz"] >= 100)

              try:
                if np.any(mask):  # Se esistono valori sopra il limite
                    raise ValueError("Indici fuori dai limiti rilevati, impossibile calcolare proj.")
                
                # Calcola `proj` solo se non ci sono indici fuori dai limiti
                proj = np.sum(_img[TOR["vx"], TOR["vy"], TOR["vz"]] * TOR["prob"])
                sino.AddItem(self._experimental_setup._bins['s'][i],self._experimental_setup._bins[i]['theta'],self._experimental_setup._bins[i]['slice'], proj)
                
              except ValueError as e:
                print(f"Errore durante il calcolo di proj: {e}")
                with open("/home/eleonora/3D_recon_DAPHNE/Reconstruction/error_log.txt", "a") as log_file:
                    log_file.write(f"Errore durante il calcolo di proj: {e}\n")
                    log_file.write("Valori con indici >= 100:\n")
                    log_file.write(f"vx: {TOR['vx'][mask]}\n")
                    log_file.write(f"vy: {TOR['vy'][mask]}\n")
                    log_file.write(f"vz: {TOR['vz'][mask]}\n")
                    log_file.write(f"prob: {TOR['prob'][mask]}\n\n")

        print('shape sino 1: ' , sinogram_list[1]._data.shape)


        return sinogram_list
