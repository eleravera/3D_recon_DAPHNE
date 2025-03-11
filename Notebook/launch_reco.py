import subprocess
"""
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 8},
     {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 8}, 
     {"pixel_nb": 200, "pixel_pitch": 0.80, "long_views": 8},
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 4},
     {"pixel_nb": 400, "pixel_pitch": 0.40, "long_views": 4}, 
     {"pixel_nb": 200, "pixel_pitch": 0.80, "long_views": 4}
"""


configurations = [
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 4},
     {"pixel_nb": 500, "pixel_pitch": 0.32, "long_views": 8},
]

input_file = "../Data/SheppLogan3D.npz"  #"../Data/cylindrical_phantom.npz"#"../Data/doseDistribution_flip.npz" #"../Data/cylindrical_phantom.npz"
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

