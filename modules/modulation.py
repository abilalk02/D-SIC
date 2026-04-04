import numpy as np
import matplotlib as plt
import random
import torch
import math 

# Generalized float to n-bit binary conversion
def float_to_nbit_bin(f, bit_width=8):
    if bit_width not in [8, 16, 32]:
        raise ValueError("bit_width must be 8, 16 or 32")
    
    sign = '0' if f >= 0 else '1'
    abs_f = abs(f)
    float32_bits = np.float32(abs_f).view(np.uint32)
    bin_str = format(float32_bits, '032b')
    
    exp_32 = int(bin_str[1:9], 2) - 127
    mantissa_32 = bin_str[9:]
    
    if bit_width == 8:
        exp_bits = 4
        mantissa_bits = 3
        bias = 7
    elif bit_width == 16:
        exp_bits = 5
        mantissa_bits = 10
        bias = 15
    elif bit_width == 32:
        exp_bits = 8
        mantissa_bits = 23
        bias = 127
    
    new_exp = exp_32 + bias
    max_exp = (1 << exp_bits) - 1
    if new_exp < 0:
        new_exp = 0
    elif new_exp > max_exp:
        new_exp = max_exp
    
    new_mantissa = mantissa_32[:mantissa_bits]
    return f"{sign}{format(new_exp, f'0{exp_bits}b')}{new_mantissa}"
    
    return f"{sign}{format(new_exp, f'0{exp_bits}b')}{new_mantissa}"

# Convert tensor to bit sequence
def tensor2bin(tensor, bit_width=16):

    tensor_flattened = tensor.view(-1).numpy()

    bit_list = []
    for number in tensor_flattened:
        bit_list.append(float_to_nbit_bin(number, bit_width))

    return bit_list

# Generalized bin to float
def bin_to_float_nbit(bin_str, bit_width=8):
    if bit_width not in [8, 16, 32, 64]:
        raise ValueError("bit_width must be 8, 16 or 32")
    if len(bin_str) != bit_width:
        raise ValueError(f"Binary string must be {bit_width} bits long")
    
    if bit_width == 8:
        exp_bits = 4
        mantissa_bits = 3
        bias = 7
    elif bit_width == 16:
        exp_bits = 5
        mantissa_bits = 10
        bias = 15
    elif bit_width == 32:
        exp_bits = 8
        mantissa_bits = 23
        bias = 127
    
    sign = int(bin_str[0])
    exp = int(bin_str[1:1 + exp_bits], 2)
    mantissa = int(bin_str[1 + exp_bits:], 2)
    
    if exp == 0 and mantissa == 0:
        value = 0.0
    else:
        significand = 1 + mantissa / (2 ** mantissa_bits)
        value = significand * (2 ** (exp - bias))
    
    if math.isnan(value) or math.isinf(value):       
        value = np.random.randn()
      
    if value > 10:    
      value=np.random.randn()

    if value < -10:     
      value=np.random.randn()

    if value < 1e-2 and value>-1e-2:     
      value = np.random.randn()
    
    return -value if sign == 1 else value

def bin2tensor(input_list, bit_width=16):
    tensor_reconstructed = [bin_to_float_nbit(bin, bit_width) for bin in input_list]
    return torch.FloatTensor(tensor_reconstructed)

# QAM Modulatio
def qam_modulate(bit_list, M=4):
    if M not in [4, 16, 64]:
        raise ValueError("M must be 4, 16, or 64")
    
    bits_per_symbol = int(np.log2(M))
    bit_string = ''.join(bit_list)
    padding_needed = (bits_per_symbol - (len(bit_string) % bits_per_symbol)) % bits_per_symbol
    bit_string += '0' * padding_needed
    bit_chunks = [bit_string[i:i + bits_per_symbol] for i in range(0, len(bit_string), bits_per_symbol)]
    
    # Define constellations
    if M == 4:
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        # Already Gray-coded: 00, 01, 10, 11
        gray_map = [0b00, 0b01, 0b10, 0b11]  # Matches current order
    elif M == 16:
        constellation = np.array([i + 1j*q for i in [-3, -1, 1, 3] for q in [-3, -1, 1, 3]]) / np.sqrt(10)
        # Gray code for 4 levels: 00, 01, 11, 10
        gray_4 = [0b00, 0b01, 0b11, 0b10]
        gray_map = [((i << 2) | q) for i in gray_4 for q in gray_4]  # Combine I and Q bits
    elif M == 64:
        constellation = np.array([i + 1j*q for i in [-7, -5, -3, -1, 1, 3, 5, 7] for q in [-7, -5, -3, -1, 1, 3, 5, 7]]) / np.sqrt(42)
        # Gray code for 8 levels: 000, 001, 011, 010, 110, 111, 101, 100
        gray_8 = [0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100]
        gray_map = [((i << 3) | q) for i in gray_8 for q in gray_8]
    
    # Map bits to symbols 
    symbols = []
    for chunk in bit_chunks:
        chunk_int = int(chunk, 2)  
        idx = gray_map.index(chunk_int)
        symbols.append(constellation[idx])
    
    return np.array(symbols)

# QAM Demodulation
def qam_demodulate(symbols, M=4, bit_width=8):
    if M not in [4, 16, 64]:
        raise ValueError("M must be 4, 16, or 64")
    if bit_width not in [8, 16, 32, 64]:
        raise ValueError("bit_width must be 8, 16, 32, or 64")
    
    bits_per_symbol = int(np.log2(M))
    
    if M == 4:
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        gray_map = [0b00, 0b01, 0b10, 0b11]
    elif M == 16:
        constellation = np.array([i + 1j*q for i in [-3, -1, 1, 3] for q in [-3, -1, 1, 3]]) / np.sqrt(10)
        gray_4 = [0b00, 0b01, 0b11, 0b10]
        gray_map = [((i << 2) | q) for i in gray_4 for q in gray_4]
    elif M == 64:
        constellation = np.array([i + 1j*q for i in [-7, -5, -3, -1, 1, 3, 5, 7] for q in [-7, -5, -3, -1, 1, 3, 5, 7]]) / np.sqrt(42)
        gray_8 = [0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100]
        gray_map = [((i << 3) | q) for i in gray_8 for q in gray_8]
    
    recovered_bits = []
    for symbol in symbols:
        distances = np.abs(constellation - symbol)
        closest_idx = np.argmin(distances)
        bit_string = format(gray_map[closest_idx], f'0{bits_per_symbol}b')
        recovered_bits.append(bit_string)
    
    full_bit_string = ''.join(recovered_bits)
    recovered_bit_list = [full_bit_string[i:i + bit_width] for i in range(0, len(full_bit_string), bit_width)]
    if len(recovered_bit_list[-1]) < bit_width:
        recovered_bit_list[-1] += '0' * (bit_width - len(recovered_bit_list[-1]))
    
    return recovered_bit_list

def awgn_channel(symbols, snr_db):
    
    # Convert symbols to PyTorch tensor for power calculation
    signal = torch.tensor(symbols, dtype=torch.complex64)
    signal_power = torch.mean(torch.abs(signal)**2).item()  # Actual signal power

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10.0)
    
    # Noise power based on actual signal power
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2)  # Std dev per dimension
    
    # Generate complex Gaussian noise
    noise_real = np.random.normal(0, sigma, symbols.shape)
    noise_imag = np.random.normal(0, sigma, symbols.shape)
    noise = noise_real + 1j * noise_imag
    
    return symbols + noise

# Rayleigh fading channel
def rayleigh_fading_channel(symbols, snr_db):

    # Convert symbols to PyTorch tensor for power calculation
    signal = torch.tensor(symbols, dtype=torch.complex64)
    signal_power = torch.mean(torch.abs(signal)**2).item()

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10.0)

    # Noise power based on actual signal power
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2)

    # Random fading coefficient
    h_real = np.random.normal(0, np.sqrt(0.5), 1)
    h_imag = np.random.normal(0, np.sqrt(0.5), 1)
    h = h_real + 1j * h_imag

    # Apply fading
    faded_symbols = h * symbols
    noise_real = np.random.normal(0, sigma, symbols.shape)
    noise_imag = np.random.normal(0, sigma, symbols.shape)
    noise = noise_real + 1j * noise_imag
    received = faded_symbols + noise

    return received

def rician_fading_channel(symbols, snr_db):

    # Convert symbols to PyTorch tensor for power calculation
    signal = torch.tensor(symbols, dtype=torch.complex64)

    # Convert SNR from dB to linear scale
    signal_power = torch.mean(torch.abs(signal)**2).item()
    snr_linear = 10 ** (snr_db / 10.0)

    # Noise power based on actual signal power
    noise_power = signal_power / snr_linear

    sigma = np.sqrt(noise_power / 2)
    K_linear = 2.0
    h_real_rayleigh = np.random.normal(0, np.sqrt(0.5), 1)
    h_imag_rayleigh = np.random.normal(0, np.sqrt(0.5), 1)
    h_rayleigh = h_real_rayleigh + 1j * h_imag_rayleigh
    h = np.sqrt(K_linear / (K_linear + 1)) + np.sqrt(1 / (K_linear + 1)) * h_rayleigh
    faded_symbols = h * symbols
    noise_real = np.random.normal(0, sigma, symbols.shape)
    noise_imag = np.random.normal(0, sigma, symbols.shape)
    noise = noise_real + 1j * noise_imag
    received = faded_symbols + noise

    return received

def ls_channel_estimation(received, pilots, device="cpu"):

    received_tensor = torch.tensor(received, dtype=torch.complex64, device=device)
    pilots_tensor = torch.tensor(pilots, dtype=torch.complex64, device=device)
    numerator = torch.conj(pilots_tensor) @ received_tensor
    denominator = torch.conj(pilots_tensor) @ pilots_tensor
    h_hat = numerator / denominator

    return h_hat

def estimate_snr(received, pilots, h_hat, device="cpu"):

    received_tensor = torch.tensor(received, dtype=torch.complex64, device=device)
    pilots_tensor = torch.tensor(pilots, dtype=torch.complex64, device=device)
    h_hat_tensor = h_hat
    estimated_signal = h_hat_tensor * pilots_tensor
    noise = received_tensor - estimated_signal
    sigma_n_squared = torch.mean(torch.abs(noise)**2).item()
    Ps = torch.mean(torch.abs(pilots_tensor)**2).item()
    estimated_snr_linear = Ps / sigma_n_squared

    return 10 * np.log10(estimated_snr_linear)

def lmmse_equalization(noisy_symbols, h_estimated, snr_estimated, device="cpu"):

    noisy_symbols_tensor = torch.tensor(noisy_symbols, dtype=torch.complex64, device=device)
    h_estimated_tensor = h_estimated.to(device)
    snr_linear = 10 ** (snr_estimated / 10.0)
    signal_power = 1.0
    noise_power = signal_power / snr_linear
    h_conj = torch.conj(h_estimated_tensor)
    h_magnitude_squared = torch.abs(h_estimated_tensor)**2
    w = h_conj / (h_magnitude_squared + noise_power / signal_power)
    equalized_symbols = w * noisy_symbols_tensor

    return equalized_symbols.cpu().numpy()

def normalize(data, min_val, max_val, target_range=(-1, 1)):
    return 2 * (data - min_val) / (max_val - min_val) - 1 if target_range == (-1, 1) else (data - min_val) / (max_val - min_val)

def denormalize(data, min_val, max_val, target_range=(-1, 1)):
    return (data + 1) / 2 * (max_val - min_val) + min_val if target_range == (-1, 1) else data * (max_val - min_val) + min_val


