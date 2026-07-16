import numpy as np
import librosa

def peak_normalize(audio_array, target_peak=0.89, eps=1e-10):
    max_amp = np.max(np.abs(audio_array)) + eps
    if max_amp == 0:
        return audio_array
    return (audio_array / max_amp) * target_peak

def rms_db(audio_array):
    rms = np.sqrt(np.mean(audio_array**2) + 1e-10)
    return 20 * np.log10(rms)

def max_internal_silence_sec(y, sr, silence_db=-25):
    intervals = librosa.effects.split(y, top_db=-silence_db)
    if len(intervals) <= 1:
        return 0.0

    silences = []
    for (s1, e1), (s2, e2) in zip(intervals[:-1], intervals[1:]):
        silences.append((s2 - e1) / sr)

    return max(silences)

# # Ejemplo: integrar al final de tu función preprocess_for_embeddings
def preprocess_for_embeddings(sample, audio_column="audio", target_sr=16000, max_silence_sec=1, target_peak=0.89):
    audio = sample[audio_column]
    array = audio["array"]
    sr = audio["sampling_rate"]
    # 1) mono
    if array.ndim > 1:
        array = librosa.to_mono(array)

    # # 2) resample
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 3) recortar silencios largos (opcional)
    max_sil = max_internal_silence_sec(array, sr)

    if max_sil > max_silence_sec:
        intervals = librosa.effects.split(array, top_db=25)
        if len(intervals) > 1:
            array = np.concatenate([array[s:e] for s,e in intervals])

    # 4) (opcional) reducción ligera de ruido si lo necesitas...

    # 5) normalización final por pico
    array = peak_normalize(array, target_peak=target_peak)

    # Opcional: imprime o guarda métricas para debugging
    # print("Peak after:", np.max(np.abs(array)))
    # print("RMS dB after:", rms_db(array))

    return {
        audio_column: {
            "array": array,
            "sampling_rate": sr
        }
    }
