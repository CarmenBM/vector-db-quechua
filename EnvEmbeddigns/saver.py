import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import re

def guardar_audios_local(dataset, carpeta_salida, nombre_base):
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # Buscar el último número usado en cualquier archivo que siga el patrón _preAUD_###.wav
    #patron = re.compile(r"_preAUD_(\d+)\.wav$")
    patron = re.compile(r"_embedding_(\d+)\.wav$")
    existentes = [
        int(patron.search(f.name).group(1))
        for f in carpeta_salida.glob("*.wav")
        if patron.search(f.name)
    ]

    start_idx = max(existentes) if existentes else 0
    total_final = start_idx + len(dataset)
    ancho_id = len(str(total_final))  # ceros según total acumulado

    for offset, example in enumerate(
        tqdm(dataset, desc="Guardando audios", unit="archivo"), start=1
    ):
        idx = start_idx + offset
        id_str = str(idx).zfill(ancho_id)
        audio_filename = carpeta_salida / f"{nombre_base}_preAUD_{id_str}.wav"

        audio_array = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        sf.write(str(audio_filename), audio_array, sr)

    print(f"\nAñadidos {len(dataset)} audios. Total actual: {total_final}")