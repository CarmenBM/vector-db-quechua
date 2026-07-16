from datasets import load_dataset
import numpy as np
from collections import defaultdict

# def analyze_dataset_audio(dataset, audio_column="audio"):
#     """
#     Analiza las características generales de los audios en el dataset.

#     Args:
#         dataset: Dataset de Hugging Face.
#         audio_column: Nombre de la columna con los audios.

#     Returns:
#         dict: Estadísticas generales del dataset.
#     """
#     stats = {
#         'total_samples': 0,
#         'sample_rates': defaultdict(int),
#         'channels': defaultdict(int),
#         'dtypes': defaultdict(int),
#         'formats': defaultdict(int),
#         'durations': [],
#         'max_amplitudes': [],
#         'empty_samples': 0,
#         'invalid_samples': 0
#     }

#     for i, sample in enumerate(dataset):
#         try:
#             audio = sample[audio_column]
#             array = audio['array']
#             sr = audio['sampling_rate']

#             stats['total_samples'] += 1

#             # Tasa de muestreo
#             stats['sample_rates'][sr] += 1

#             # Canales (mono o estéreo)
#             channels = 1 if array.ndim == 1 else 2
#             stats['channels'][channels] += 1

#             # Tipo de datos
#             stats['dtypes'][str(array.dtype)] += 1

#             # formato (si hay path disponible)
#             if "path" in audio and isinstance(audio["path"], str):
#                 ext = audio["path"].rsplit(".", 1)[-1].lower()
#                 stats["formats"][ext] += 1

#             # Duración
#             duration = len(array) / sr if sr > 0 else 0
#             stats['durations'].append(duration)

#             # Máxima amplitud
#             if array.size > 0:
#                 max_amp = np.max(np.abs(array))
#                 stats['max_amplitudes'].append(max_amp)
#             else:
#                 stats['empty_samples'] += 1
#                 print(f"Muestra {i}: Audio vacío (tamaño 0)")

#         except Exception as e:
#             stats['invalid_samples'] += 1
#             print(f"Muestra {i}: Error - {e}")

#     stats['duracion_total'] = np.sum(stats['durations'])

#     # Imprimir resumen
#     print("\n=== ANÁLISIS DE AUDIOS LOCALES ===")
#     print(f"Total de muestras: {stats['total_samples']}")
#     print(f"Muestras vacías: {stats['empty_samples']}")
#     print(f"Muestras inválidas: {stats['invalid_samples']}")
#     print(f"Tasas de muestreo: {dict(stats['sample_rates'])}")
#     print(f"Canales: {dict(stats['channels'])}")
#     print(f"Tipos de datos: {dict(stats['dtypes'])}")
#     print(f"Formatos: {dict(stats['formats'])}")
#     print(f"Duraciones (s): promedio {np.mean(stats['durations']):.2f}, min {np.min(stats['durations']):.2f}, max {np.max(stats['durations']):.2f}")
#     print(f"Duración total del dataset: {stats['duracion_total'] / 60:.2f} min")
#     print(f"Máxima amplitud: promedio {np.mean(stats['max_amplitudes']):.2f}, min {np.min(stats['max_amplitudes']):.2f}, max {np.max(stats['max_amplitudes']):.2f}")

#     return stats

def analyze_dataset_audio(dataset, audio_column="audio"):
    stats = {
        'total_samples': 0,
        'sample_rates': defaultdict(int),
        'channels': defaultdict(int),
        'dtypes': defaultdict(int),
        'formats': defaultdict(int),
        'durations': [],
        'max_amplitudes': [],
        'clipping_count': 0, 
        'empty_samples': 0,
        'invalid_samples': 0
    }

    for i, sample in enumerate(dataset):
        try:
            audio = sample[audio_column]
            array = audio['array']
            sr = audio['sampling_rate']

            stats['total_samples'] += 1

            # Tasa de muestreo
            stats['sample_rates'][sr] += 1

            # Canales: Ajustado para ser más robusto
            if array.ndim == 1:
                num_channels = 1
            else:
                num_channels = array.shape[0] # Asumiendo formato (Canales, Tiempo)
            stats['channels'][num_channels] += 1

            # Tipo de datos
            stats['dtypes'][str(array.dtype)] += 1

            # Formato
            if "path" in audio and isinstance(audio["path"], str):
                ext = audio["path"].rsplit(".", 1)[-1].lower()
                stats["formats"][ext] += 1

            # Usamos shape[-1] para asegurarnos de agarrar las muestras, no los canales
            if sr > 0:
                num_samples = array.shape[-1] 
                duration = num_samples / sr
                stats['durations'].append(duration)
            else:
                stats['durations'].append(0)

            # --- CORRECCIÓN DE AMPLITUD ---
            if array.size > 0:
                max_amp = np.max(np.abs(array))
                stats['max_amplitudes'].append(max_amp)
                
                # Detectar si el audio está saturado (roto)
                if max_amp >= 0.99: # <--- Umbral de seguridad
                    stats['clipping_count'] += 1
            else:
                stats['empty_samples'] += 1
                

        except Exception as e:
            stats['invalid_samples'] += 1
            print(f"Muestra {i}: Error - {e}")

    stats['duracion_total'] = np.sum(stats['durations'])

    print("\n=== ANÁLISIS DE AUDIOS LOCALES ===")
    print(f"Total de muestras: {stats['total_samples']}")
    print(f"Muestras vacías: {stats['empty_samples']}")
    print(f"Muestras inválidas: {stats['invalid_samples']}")
    print(f"Audios con Clipping (Saturados >= 0.99): {stats['clipping_count']}") 
    print(f"Tasas de muestreo: {dict(stats['sample_rates'])}")
    print(f"Canales: {dict(stats['channels'])}")
    print(f"Formatos: {dict(stats['formats'])}")
    
    if stats['durations']:
        print(f"Duraciones (s): promedio {np.mean(stats['durations']):.2f}, min {np.min(stats['durations']):.2f}, max {np.max(stats['durations']):.2f}")
        print(f"Duración total del dataset: {stats['duracion_total'] / 60:.2f} min / {stats['duracion_total'] / 3600:.2f} horas")
    
    if stats['max_amplitudes']:
        print(f"Máxima amplitud (Pico): promedio {np.mean(stats['max_amplitudes']):.2f}, min {np.min(stats['max_amplitudes']):.2f}, max {np.max(stats['max_amplitudes']):.2f}")

    return stats