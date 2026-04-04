import subprocess
import os

# Base directory
BASE_DIR = "output/awgn"
ORIGINAL_DIR = os.path.join(BASE_DIR, "original")

# List of dB levels
DB_LEVELS = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1]

# Function to calculate mean from LPIPS file
def calculate_mean(file_path):
    with open(file_path, 'r') as f:
        scores = [float(line.split()[1]) for line in f.readlines()]
    return sum(scores) / len(scores) if scores else 0

# Process each dB level
for db in DB_LEVELS:
    compare_dir = os.path.join(BASE_DIR, f"reconstructed_{db}dB")
    output_file = os.path.join(compare_dir, "lpips.txt")
    
    print(f"Processing {db}dB...")

    # Run the LPIPS command
    cmd = [
        "python", "lpips_2dirs.py",
        "-d0", ORIGINAL_DIR,
        "-d1", compare_dir,
        "-o", output_file,
        "--use_gpu"
    ]
    subprocess.run(cmd, check=True)

    # Calculate and display the mean
    mean_score = calculate_mean(output_file)
    print(f"Mean LPIPS score for qam_{db}dB: {mean_score:.4f}")
    print("-----------------------------------")

