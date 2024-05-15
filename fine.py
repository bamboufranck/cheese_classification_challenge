import os
import replicate

print("start")

# Utiliser le token lors de la création de l'entraînement
training = replicate.trainings.create(
    model="stability-ai/sdxl",
    version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={
        "input_images": "https://franckbambou.s3.eu-north-1.amazonaws.com/photo.zip?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEIaCmV1LW5vcnRoLTEiRzBFAiEAoee1CSXZWkISZCJydicMilMt6xKOWh543emNejHaqC4CIHYZ4CA7IAyf0OEA7gka3DCZPQ20L0gIA%2FAVBV6KIcdWKu0CCKz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMMzgxNDkyMDU5Nzc4Igxb%2Fe6dV1kx%2F9aQK%2BQqwQLrVj1SWidOJ4l1w4Og90Trf9Wnm8vxdBdEUuzjkvpER5jGLs%2FZrNAQLNOKlHB9gywqQUvVgvWHdqDzi9XSKbfHq2QyeLifsSjddIh1gOAMOXsISux%2FMWlT6v8NSJomjVrIqrw5ga6X2s%2FHtW3bwthnuznFYRxVQ9fEMz%2FSNx3MXmS87gIbmjPFJLYLT6oMES%2Bf6KIGrHU6ThSAJHbSq2djQ%2BA3UaAcvAmNaaNS1HeUu3HrWdQ%2BLV%2FMiig3xSWwUo4IhYeDiWAWrZ0apl9C2jjZMtu1pY6%2BT9MZSuDyFlA3vkJi9QgvV9jJuhYEw3cjmXoOSimjdRoGqcm8rK1c1YdG%2B1afZgBE%2BxiB5tcBo9%2FaWW%2BO%2BtjFqvJ8yuMLiKigzKWNboZAg5dApPfaeGTKHY0qIRYKRphhWuiMYsGu4SnrWMAwz%2B2TsgY6swLssnI6LznOZdlAYiiJZy4BIbcBv4Qu%2Bh7ihDSYrNr1WobBEPi2MVKygIMY5lW2q7MmgozuVt2eitwjDWqiJHY7nJc0kX4aN3ZXO5U2Eh2hwZfNOUBg3TJXRL7S9ChjpkOJ0iq3t9jLEnmYAkiteANi3wGXaynDO6MnWv5MxMPeNF2wsvrSLQYOlubVbTzAdUOUQ0Euu2goIDRwGw2WaTMWmZPJaS2Vc8Y9g%2BKXqvqXikPPO0zrlLfTnYu1DxkCh%2F%2BR6O%2FinsWufNoVqZ%2FxLla%2Bqum7yaUftvGNSKrUp6aqk6JwSND9sjG3SL%2BM64mbVAFqxh3YBjNi6Qt9IyyQFUS3DGA4dlY8L%2F%2BJEWOVDz%2FXWStWE3ChlE0ELHcqNaljt5qzXugSCH6mlehBHSRpObhWO8l8&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240515T202504Z&X-Amz-SignedHeaders=host&X-Amz-Expires=42000&X-Amz-Credential=ASIAVRUVS32BHDXU7FES%2F20240515%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Signature=74300f64e9781e9dc0fe28f610ab623764203e8bac54d6cbca9091abcf2d8314",
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
