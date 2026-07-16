import numpy as np
from scipy import signal
from collections import defaultdict
from tqdm import tqdm
#35db
def compute_silence_ratio_numpy(audio_array, sr, silence_db=-35, min_silence_duration_sec=1.0):
    """
    Calcula la proporción de silencio en una señal de audio usando RMS y umbral en dB.
    """
    frame_len = int(0.025 * sr)  # 25 ms
    hop_len = int(0.010 * sr)    # 10 ms

    # RMS móvil
    rms = np.sqrt(signal.convolve(audio_array**2, np.ones(frame_len)/frame_len, mode='valid')[::hop_len])
    rms_db = 20 * np.log10(rms + 1e-10)

    # Mascara de silencio
    silence_mask = rms_db < silence_db

    # Detectar duraciones de silencio consecutivas
    silence_durations = []
    count = 0
    for silent in silence_mask:
        if silent:
            count += 1
        else:
            if count * hop_len >= min_silence_duration_sec * sr:
                silence_durations.append(count * hop_len)
            count = 0

    total_silence = sum(silence_durations)
    return total_silence / len(audio_array)

def estimate_snr(audio, sr, lowpass_freq=4000, order=4):

   # Normalizar audio
   audio = audio / (np.max(np.abs(audio)) + 1e-10)

   # Filtro paso bajo para estimar señal de voz
   b, a = signal.butter(order, lowpass_freq / (sr / 2), btype='low')
   smooth_signal = signal.filtfilt(b, a, audio)
   signal_power = np.mean(smooth_signal**2)
   noise_power = np.mean((audio - smooth_signal)**2)
   snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
   return snr

def detect_noise_snr(audio, sr, snr_threshold=15):
   snr = estimate_snr(audio, sr)
   return snr < snr_threshold


def verify_preprocessing(dataset, audio_column="audio", expected_sr=16000):
    """
    Verifica si los audios cumplen requisitos de preprocesamiento: tasa de muestreo, canales, silencio,
    amplitud y ruido. Optimizado para velocidad.
    """
    stats = {
        'sample_rates': defaultdict(int),
        'channels': defaultdict(int),
        'silence_issues': 0,
        'amplitude_issues': 0,
        'resampling_issues': 0,
        'noise_issues': 0,
        'total_samples': 0,
        'indices': {
            'resampling': [],
            'channels': [],
            'silence': [],
            'amplitude': [],
            'noise': []
        }
    }

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Verificando audios"):
        try:
            audio = sample[audio_column]
            array = audio['array']
            sr = audio['sampling_rate']

            stats['total_samples'] += 1

            # 1. Resampleo
            stats['sample_rates'][sr] += 1
            if sr != expected_sr:
                stats['resampling_issues'] += 1
                stats['indices']['resampling'].append(idx)

            # 2. Canales
            channels = 1 if array.ndim == 1 else array.shape[0]
            stats['channels'][channels] += 1
            if channels != 1:
                stats['indices']['channels'].append(idx)

            # 3. Silencio
            silence_ratio = compute_silence_ratio_numpy(array, sr)
            if silence_ratio > 0.3:
                stats['silence_issues'] += 1
                stats['indices']['silence'].append(idx)

            # 4. Amplitud
            max_amp = np.max(np.abs(array))
            if not 0.85 <= max_amp <= 1.05:
                stats['amplitude_issues'] += 1
                stats['indices']['amplitude'].append(idx)

            # 5. Ruido
            is_noisy = detect_noise_snr(array, sr)
            if is_noisy:
                stats['noise_issues'] += 1
                stats['indices']['noise'].append(idx)

        except Exception as e:
            print(f"Error en muestra {idx}: {e}")
            continue

    # Verificación general
    checks = {
        'resampling': len(stats['sample_rates']) == 1 and expected_sr in stats['sample_rates'],
        'channels': len(stats['channels']) == 1 and 1 in stats['channels'],
        'silence': stats['silence_issues'] == 0,
        'amplitude': stats['amplitude_issues'] == 0,
        'noise': stats['noise_issues'] < 0.1 * stats['total_samples']
    }

    is_consistent = all(checks.values())

    # Informe
    print("\n=== VERIFICACIÓN DE PREPROCESAMIENTO ===")
    for label, ok in checks.items():
        indices = ", ".join(map(str, stats['indices'][label])) if not ok else ""
        print(f"{label.upper():<10}: {'OK' if ok else 'Problema'} {f'(índices: {indices})' if indices else ''}")

    print(f"\nPREPROCESAMIENTO {'CORRECTO' if is_consistent else 'CON PROBLEMAS'}")

    return stats, is_consistent