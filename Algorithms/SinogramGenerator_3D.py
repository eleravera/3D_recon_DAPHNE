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
        for i in range(0, self._experimental_setup._detector_number):
            # copy some of the parameters inside the sinogram
            self.sinogram = Sinogram(self._experimental_setup.pixels_per_slice_nb,
                                      self._experimental_setup.gantry_angles_nb,
                                      self._experimental_setup.detector_slice_nb)
            sinogram_list.append(self.sinogram)

        _img = img
        if transponse_image == 1:
            _img = np.transpose(img, axes=(1, 0, 2))
        else:
            _img = img

        #print('The image shape is: ', _img.shape)
        Siddon = SiddonProjector(
            self._experimental_setup.image_matrix_size_mm,
            self._experimental_setup.voxel_size_mm,
        )
        StartPoints = self._experimental_setup._projections_extrema["p0"]
        EndPoints =   self._experimental_setup._projections_extrema["p1"]

        for j in range(0, self._experimental_setup._detector_number):
          sino = sinogram_list[j]
          # for convenience we copy the usel attributes of the geometry inside the sinogram object
          sino.image_matrix_size_mm = self._experimental_setup.image_matrix_size_mm
          sino.voxel_size_mm = self._experimental_setup.voxel_size_mm
          sino._radial_step_mm=self._experimental_setup._radial_step_mm
          sino.mode=self._experimental_setup.mode
          sino._voxel_nb=self._experimental_setup._voxel_nb
          sino._angular_step_deg=self._experimental_setup._angular_step_deg
          sino.gantry_angles_nb=self._experimental_setup.gantry_angles_nb
          sino.detector_slice_nb=self._experimental_setup.detector_slice_nb
          sino.slice_pitch_mm=self._experimental_setup.slice_pitch_mm
  #       print(sino.geometry)
          if self._experimental_setup.mode != Mode.PARALLELBEAM:
              sino.sad_mm= self._experimental_setup.sad_mm
              sino.sdd_mm= self._experimental_setup.sdd_mm
          sino.detector_pitch_mm= self._experimental_setup._detector_pitch_mm
          sino.pixels_per_slice_nb=self._experimental_setup.pixels_per_slice_nb

          #print('StartPoints and EndPoints shapes:', StartPoints.shape, EndPoints.shape)
          #print('Loop done on: int(len(StartPoints)/self._experimental_setup._detector_number) ', int(len(StartPoints)/self._experimental_setup._detector_number))
            
          index = int(len(StartPoints)/self._experimental_setup._detector_number) *j 
          print('index: ', index)
          perc0=perc1=0
          for i in range(0, int(len(StartPoints)/self._experimental_setup._detector_number)):
              perc0=perc1
              perc1=np.trunc((i+1)/len(StartPoints)*100).astype(np.int32)
              if perc1 != perc0:
                  print("Projecting data, " + str(perc1) + "% done...", end="\r")
              TOR = Siddon.CalcIntersection(StartPoints[i+index], EndPoints[i+index])

              mask = (TOR["vx"] >= 100) | (TOR["vy"] >= 100) | (TOR["vz"] >= 100)
            
              try:
                if np.any(mask):  # Se esistono valori sopra il limite
                    raise ValueError("Indici fuori dai limiti rilevati, impossibile calcolare proj.")
                
                # Calcola `proj` solo se non ci sono indici fuori dai limiti
                proj = np.sum(_img[TOR["vx"], TOR["vy"], TOR["vz"]] * TOR["prob"])
                sino.AddItem(self._experimental_setup._bins['s'][i],self._experimental_setup._bins[i]['theta'],self._experimental_setup._bins[i]['slice'], proj)
              
              except ValueError as e:
                print(f"Errore durante il calcolo di proj: {e}")
                print("Valori con indici >= 100:")
                print("vx:", TOR["vx"][mask])
                print("vy:", TOR["vy"][mask])
                print("vz:", TOR["vz"][mask])
                print("prob:", TOR["prob"][mask])                  

        return sinogram_list
