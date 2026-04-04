import subprocess
import os

# Base directory
BASE_DIR = "output/awgn"
ORIGINAL_DIR = os.path.join(BASE_DIR, "original")

# List of dB levels
DB_LEVELS = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1]
# Process each dB level
for db in DB_LEVELS:
    compare_dir = os.path.join(BASE_DIR, f"reconstructed_{db}dB")
    
    print(f"Processing FID for {db}dB...")

    # Run the FID command
    cmd = [
        "python", "-m", "pytorch_fid",
        ORIGINAL_DIR,
        compare_dir,
        "--device", "cuda:0"
    ]

    # Capture the output and display it
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    fid_output = result.stdout.strip() 
    print(fid_output)
    print("-----------------------------------")

