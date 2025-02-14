import subprocess
"""
    # FATTI A DICEMBRE
    {"pixel_nb": 200, "pixel_pitch": 0.8, "long_views": 8},
    {"pixel_nb": 200, "pixel_pitch": 1., "long_views": 8},
    {"pixel_nb": 200, "pixel_pitch": 1.1, "long_views": 8},
    {"pixel_nb": 200, "pixel_pitch": 1.2, "long_views": 8},
    {"pixel_nb": 200, "pixel_pitch": 1.3, "long_views": 8},
    {"pixel_nb": 200, "pixel_pitch": 1.4, "long_views": 8},
    {"pixel_nb": 400, "pixel_pitch": 0.4, "long_views": 8},
    {"pixel_nb": 400, "pixel_pitch": 0.5, "long_views": 8}, 
    {"pixel_nb": 400, "pixel_pitch": 0.6, "long_views": 8},
    {"pixel_nb": 400, "pixel_pitch": 0.7, "long_views": 8},

    #FATTI A GENNAIO    
     {"pixel_nb": 400, "pixel_pitch": 0.8, "long_views": 8},
     {"pixel_nb": 500, "pixel_pitch": 0.35, "long_views": 8},
     {"pixel_nb": 500, "pixel_pitch": 0.4, "long_views": 8},    
     {"pixel_nb": 500, "pixel_pitch": 0.5, "long_views": 8},  
     {"pixel_nb": 500, "pixel_pitch": 0.6, "long_views": 8},
     {"pixel_nb": 500, "pixel_pitch": 0.7, "long_views": 8},     
     {"pixel_nb": 500, "pixel_pitch": 0.8, "long_views": 8}
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 8},
     # stessa cosa con 4 views. 
    {"pixel_nb": 200, "pixel_pitch": 0.9, "long_views": 8},     
    {"pixel_nb": 200, "pixel_pitch": 0.8, "long_views": 4},
    {"pixel_nb": 200, "pixel_pitch": 0.9, "long_views": 4},
    {"pixel_nb": 400, "pixel_pitch": 0.4, "long_views": 4},
    {"pixel_nb": 400, "pixel_pitch": 0.5, "long_views": 4}, 
    {"pixel_nb": 400, "pixel_pitch": 0.6, "long_views": 4},
    {"pixel_nb": 400, "pixel_pitch": 0.7, "long_views": 4},
    {"pixel_nb": 400, "pixel_pitch": 0.8, "long_views": 4},   
    {"pixel_nb": 500, "pixel_pitch": 0.35, "long_views": 4},
    {"pixel_nb": 500, "pixel_pitch": 0.4, "long_views": 4},    
    {"pixel_nb": 500, "pixel_pitch": 0.5, "long_views": 4},  
    {"pixel_nb": 500, "pixel_pitch": 0.6, "long_views": 4},
    {"pixel_nb": 500, "pixel_pitch": 0.7, "long_views": 4},     
    {"pixel_nb": 500, "pixel_pitch": 0.8, "long_views": 4}, 
    {"pixel_nb": 200, "pixel_pitch": 0.83, "long_views": 4},
    {"pixel_nb": 200, "pixel_pitch": 0.83, "long_views": 8},
    {"pixel_nb": 400, "pixel_pitch": 0.42, "long_views": 4},
    {"pixel_nb": 400, "pixel_pitch": 0.42, "long_views": 8},         
    {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 4},         
    {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 8}

    {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 8, "sigma": 2}, #4
    {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 8, "sigma": 4}, #6
    {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 8, "sigma": 6}, #7-8
    {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 8, "sigma": 8} #
    

    con dose distribution non ruotata "/home/eleonora/eFLASH_3D_Sim-build/Dose_Map_30mm/spettro_9MeV/doseDistribution.npz"
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 8},
     {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 8}, 
     {"pixel_nb": 200, "pixel_pitch": 0.80, "long_views": 8},
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 4},
     {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 4}, 
     {"pixel_nb": 200, "pixel_pitch": 0.80, "long_views": 4},
    
"""


"""
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 8},
     {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 8}, 
     {"pixel_nb": 200, "pixel_pitch": 0.80, "long_views": 8},
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 4},
     {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 4}, 
     {"pixel_nb": 200, "pixel_pitch": 0.80, "long_views": 4}
"""


configurations = [
     {"pixel_nb": 200, "pixel_pitch": 0.80, "long_views": 4},
]

input_file = "../Data/cylindrical_phantom.npz"#"../Data/doseDistribution_flip.npz" #"../Data/cylindrical_phantom.npz"
iterations = 50

# Ciclo attraverso le configurazioni e eseguo lo script
for config in configurations:
    command = [
        "python3", "pinholeMLEM.py",
        "--input_file", input_file,
        "--iterations", str(iterations),
        "--long_views", str(config["long_views"]),
        "--pixel_nb", str(config["pixel_nb"]),
        "--pixel_pitch", str(config["pixel_pitch"])
        #"--sigma", str(config["sigma"])         #######da commentare in caso usassi pinholeMLEM.py
    ]
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

