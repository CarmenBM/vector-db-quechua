import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

def guardar_dataset_parquet(dataset, carpeta_salida, nombre_base):
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    registros = []

    for idx, example in enumerate(tqdm(dataset, desc="Procesando dataset")):
        # Archivo de audio
        audio_array = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]

        # Guardar WAV en disco (opcional si quieres tener los archivos físicos)
        audio_filename = carpeta_salida / f"{nombre_base}_{idx:05d}.wav"
        sf.write(str(audio_filename), audio_array, sr)

        # Extraer embedding si existe
        emb = example.get("embedding", None)
        if emb is not None:
            emb = np.array(emb, dtype=np.float32).tolist()

        registros.append({
            "id": idx,
            "path": str(audio_filename),
            "sampling_rate": sr,
            #"array": audio_array.tolist(),   # opcional, puede ser muy pesado
            "audio_original": example["audio_path"],
            "sentence": example["sentence"], #para -> nlp:example["target_raw"], quechua:example["sentence"]
            #"source_raw": example["source_raw"],
            #"embedding": emb
        })

    # Crear DataFrame
    df = pd.DataFrame(registros)

    # Guardar en parquet
    parquet_path = carpeta_salida / f"{nombre_base}.parquet"
    df.to_parquet(parquet_path, index=False)

    print(f" Dataset guardado en {parquet_path}")
    return df
