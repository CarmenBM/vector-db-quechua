import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Carga de modelo y procesador
model_name = "ivangtorre/wav2vec2-xlsr-300m-quechua"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.eval()  # desactiva dropout

device = torch.device("cpu")  # Forzamos CPU
model = model.to(device)

def get_embeddings_batch(batch_audio, sr=16000):
    """
    batch_audio: lista de arrays numpy float32 (1D, diferentes duraciones)
    """
    # Verificación de tipos
    for a in batch_audio:
        assert a.dtype == np.float32, "El audio debe ser float32"
    
    # Prepara batch con padding automático
    inputs = processor(
        batch_audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs["input_values"])
        # Mean pooling en el eje de tiempo
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.cpu().numpy()  