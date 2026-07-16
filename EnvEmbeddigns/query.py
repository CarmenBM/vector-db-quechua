from datasets import load_dataset, concatenate_datasets
from analyzer import analyze_dataset_audio
from verifier import verify_preprocessing
from preprocessing import preprocess_for_embeddings
from local_loader import load_local_audio_as_dataset
# Cargar el subset 'quechua' con el split 'train' -> si no hay subset se carga de frente el split
#--dataset = load_dataset("ivangtorre/second_americas_nlp_2022", "quechua")
#--dataset = load_dataset("cportoca/Quechua_dataset")

carpeta = "C:/Users/OCarm/TESIS/BDs/quechua/filter_siminchik_corpus_iwlst_2023/wav"
dataset_completo = load_local_audio_as_dataset(carpeta)

# Unir todos los splits en uno solo
#dataset_completo = concatenate_datasets([ds for ds in dataset.values()])
#print(dataset_completo[0])

#Visualizar acarcteriticas del dataset
#stats = analyze_dataset_audio(dataset_completo)
#stats, ok= verify_preprocessing(dataset_completo)

# Preprocesar el dataset para embeddings
pre_dataset = dataset_completo.map(
    preprocess_for_embeddings,
    batched=False
)
print(pre_dataset[0])