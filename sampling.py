import os
import yaml
import json
import torch
from tqdm import tqdm
import torchvision.utils as vutils
import numpy as np
from inference.utils import *
from train import WurstCoreB
from modules.modulation import tensor2bin, bin2tensor, qam_modulate, qam_demodulate, normalize, denormalize, ls_channel_estimation
from modules.modulation import awgn_channel, rayleigh_fading_channel, rician_fading_channel, estimate_snr, lmmse_equalization
from modules.denoiser import EmbeddingDenoiseModel

snr_values = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1]    # SNR values to consider
fp_bits = 8                                             # bits for float-to-nbit conversion
M_qam = 4                                               # Modulation order
max_batches = 25                                        # Max number of images to process for each SNR

# Check GPU access
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load configuration file
config_file = 'configs/inference/sampling.yaml'
with open(config_file, "r", encoding="utf-8") as file:
    config_file = yaml.safe_load(file)

# Load the normalization factors that were obtained from the dataset used to train the denoiser model
with open("configs/normalization_stats.json", "r") as f:
    stats = json.load(f)

# Setup models (Effnet and Finetuned Stage B), data and pre-processors
core = WurstCoreB(config_dict=config_file, device=device, training=False)
extras = core.setup_extras_pre()
data = core.setup_data(extras)
models = core.setup_models(extras)
models.generator.bfloat16().eval()
print("\nFinetuned Stage B Ready")
extras.sampling_configs['cfg'] = 1.1
extras.sampling_configs['shift'] = 1
extras.sampling_configs['timesteps'] = 10
extras.sampling_configs['t_start'] = 1.0

# Set up the denoiser model 
denoiser_model = EmbeddingDenoiseModel().to(device, dtype=torch.bfloat16).eval()
denoiser_model.load_state_dict(torch.load('models/denoiser/v3/denoiser_v3.pth', map_location=device))
torch.cuda.empty_cache()

# Specify output directories
output_dir_original = "output/awgn/original"
output_dirs = {snr: f"output/awgn/reconstructed_{snr}dB" for snr in snr_values}
os.makedirs(output_dir_original, exist_ok=True)
for snr_dir in output_dirs.values():
    os.makedirs(snr_dir, exist_ok=True)

# Process Images
for batch_index, batch in enumerate(tqdm(data.iterator, desc="Processing Batches")):
    if batch_index >= max_batches:
        print(f"Reached the limit of {max_batches} batches. Stopping.")
        break

    # Downscale images (higher overall compression)
    factor = 0.75       # 0.75 gives good balance between compression and quality
    scaled_image = downscale_images(batch['images'], factor)

    # Extract EffNet embeddings
    effnet_latents = models.effnet(extras.effnet_preprocess(scaled_image.to(device)))
    batch_size = effnet_latents.shape[0]

    # Save the original images
    for i, image in enumerate(batch['images']):
        output_filename = f"00{batch_index}.png"
        output_path_original = os.path.join(output_dir_original, output_filename)
        vutils.save_image(image, output_path_original)
        print(f"Saved: {output_path_original}")

    # Process the image for different SNRs
    for snr in snr_values:
        print(f"\nSimulating with SNR = {snr} dB")

        # Float tensor to n-bit conversion
        bit_list = tensor2bin(effnet_latents.cpu(), bit_width=fp_bits)

        # QAM modulation
        symbols = qam_modulate(bit_list, M=M_qam)

        # Append pilot symbols (considers 4-QAM)
        constellation = [1+1j, 1-1j, -1+1j, -1-1j] / np.sqrt(2)
        pilot_symbols = np.array([constellation[np.random.randint(0, 4)] for _ in range(200)])
        symbols_with_pilots = np.concatenate((symbols, pilot_symbols))

        # Channel transmission (AWGN, Rayleigh, Rician)
        noisy_symbols_with_pilots = awgn_channel(symbols_with_pilots, snr_db=snr)
        #noisy_symbols_with_pilots = rician_fading_channel(symbols_with_pilots, snr_db=snr)
        #noisy_symbols_with_pilots = rayleigh_fading_channel(symbols_with_pilots, snr_db=snr)
        
        # Channel state estimation using pilot symbols
        h_estimated = ls_channel_estimation(noisy_symbols_with_pilots[-200:], pilot_symbols, device="cpu")
        snr_estimated = estimate_snr(noisy_symbols_with_pilots[-200:], pilot_symbols, h_estimated, device="cpu")
        h_real = h_estimated.real.item()
        h_imag = h_estimated.imag.item()
        
        # LMMSE equalization (exclude pilots)
        equalized_symbols = lmmse_equalization(noisy_symbols_with_pilots[:-200], h_estimated, snr_estimated, device="cpu")
        
        # Demodulation and Noisy tensor reconstruction
        bit_list_noisy = qam_demodulate(equalized_symbols, M=M_qam, bit_width=fp_bits)
        bit_list_noisy = bit_list_noisy[:effnet_latents.numel()]
        noisy_effnet_latents = bin2tensor(bit_list_noisy, bit_width=fp_bits)
        noisy_effnet_latents = noisy_effnet_latents.reshape(effnet_latents.shape)
        
        # Normalize and pre-process for denoiser
        noisy_normalized = normalize(noisy_effnet_latents.numpy(), stats["noisy_min"], stats["noisy_max"])
        h_real_normalized = normalize(h_real, stats["h_min"], stats["h_max"])
        h_imag_normalized = normalize(h_imag, stats["h_min"], stats["h_max"])
        snr_normalized = normalize(snr_estimated, stats["snr_min"], stats["snr_max"], target_range=(0,1))
        noisy_normalized = np.expand_dims(noisy_normalized, axis=1)
        h_real_tile = np.full((batch_size, 1, 16, 24, 24), h_real_normalized, dtype=np.float32)
        h_imag_tile = np.full((batch_size, 1, 16, 24, 24), h_imag_normalized, dtype=np.float32)
        snr_tile = np.full((batch_size, 1, 16, 24, 24), snr_normalized, dtype=np.float32)
        input_tensor = np.concatenate((noisy_normalized, h_real_tile, h_imag_tile, snr_tile), axis=1)
        input_tensor = torch.tensor(input_tensor, device=device, dtype=torch.float32)
        del noisy_normalized, h_real_normalized, h_imag_normalized, snr_normalized
        torch.cuda.empty_cache()

        # Denoise the received corrupted effnet latents
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            denoised_latents = denoiser_model(input_tensor).to(dtype=torch.float32)
            denoised_latents = denormalize(denoised_latents.cpu().numpy(), stats["clean_min"], stats["clean_max"])
            denoised_latents = torch.FloatTensor(denoised_latents).reshape(effnet_latents.shape).to(device, dtype=torch.bfloat16)

        # Prepare CSI in conditioning format for finetuned stage B
        h_real_tile = torch.tensor(h_real_tile, device=device, dtype=torch.bfloat16)
        h_imag_tile = torch.tensor(h_imag_tile, device=device, dtype=torch.bfloat16)
        snr_tile = torch.tensor(snr_tile, device=device, dtype=torch.bfloat16)

        # Conditioning information for finetuned Stage B 
        conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False)
        unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True)      
        conditions['effnet'] = denoised_latents
        conditions['h_real'] = h_real_tile
        conditions['h_imag'] = h_imag_tile
        conditions['snr'] = snr_tile
        unconditions['effnet'] = torch.zeros_like(denoised_latents)
        unconditions['h_real'] = torch.zeros_like(h_real_tile)
        unconditions['h_imag'] = torch.zeros_like(h_imag_tile)
        unconditions['snr'] = torch.zeros_like(snr_tile)

        # Image reconstruction
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Stage B reconstruction (finetuned diffusion model)
            sampling_b = extras.gdf.sample(
                models.generator, conditions, (batch['images'].size(0), 4, batch['images'].size(-2)//4, batch['images'].size(-1)//4),
                unconditions, device=device, **extras.sampling_configs
            )
            for (sampled_b, _, _) in tqdm(sampling_b, total=extras.sampling_configs['timesteps']):
                sampled_b = sampled_b
                torch.cuda.empty_cache()
            
            # Stage A reconstruction 
            sampled = models.stage_a.decode(sampled_b).float()

        # Save reconstructed images
        for i, image in enumerate(sampled):
            output_filename = f"00{batch_index}.png"
            output_path = os.path.join(output_dirs[snr], output_filename)
            vutils.save_image(image, output_path)
            print(f"Saved: {output_path}")


