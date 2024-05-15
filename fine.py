import os
import replicate

print("start")

# Utiliser le token lors de la création de l'entraînement
training = replicate.trainings.create(
    model="stability-ai/sdxl",
    version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={
        "input_images": "https://drive.google.com/uc?export=download&id=1qPw5GPYlh0ffqXOVVfpVK-7EEPajiJ_9",
        "token_string": "8b632f7e-34a0-48a3-8e7d-3a0de722f62c",
        "caption_prefix": "a photo of 8b632f7e-34a0-48a3-8e7d-3a0de722f62c",
        "max_train_steps": 1000,
        "use_face_detection_instead": False
    },
    destination="bamboufranck/model_franck"
)

training.reload()
print(training.status)
print("\n".join(training.logs.split("\n")[-10:]))

print("end")
