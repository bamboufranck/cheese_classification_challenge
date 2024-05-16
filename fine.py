import os
import replicate

print("start")

# Utiliser le token lors de la création de l'entraînement
training = replicate.trainings.create(

    model="bamboufranck/fossi",
    version="e026716504638dbb45364a6ab37ad7283637384acb9ce3758ce79e830be005ca",
    input={
        "input_images": "https://franckbambou.s3.eu-north-1.amazonaws.com/photo.zip?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE8aCmV1LW5vcnRoLTEiRzBFAiEAyNqgavahkAWnOukBvMcDU7%2B%2B4ok5LVXa5kAY4SIY240CIECRNqFQHJRN8a0RiF%2FZozWj%2BookRk8UBu5RZfViCfjAKu0CCLn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMMzgxNDkyMDU5Nzc4Igx4ucfP0GgsEOEjaUIqwQIQOaJzpjMys4yDwVnrXavQLK47oLyGZwfJVylDhiqEtqr3SGbc922sLHZdArslgZ%2FnCkXCAhOjsZUaYsy4IPsbmYQB6K%2FRuDjWjQIDgZwgFH5LZmDplHLJhnuT9dMqwsv2P2IlL87F%2BpLZZuqQDwe%2F6iJyHAK2P8W2AFMBHZBUvePphcR%2B6ghFRrL2s7A1w0xETHpmV3ydjb3pWdo2F1dGmCA8DDcumGTSEhYQP9nFn9z6bA2BJU10PPr98kklD%2BPnEDcKkZWuEUlx7%2FEzHmS1v5fSzVQCn%2BNKcs661lwWCIt5qEl0YYSWGTmeISO5tKtUohdUNJdf2yOE%2BOwOJm%2FSG8vTVzjIhlb3cy404y2DAAtUiJLIwWbpJjQx2dFhvsR6iIkWEgWH%2FGZpWBheAJiIfXlVi6XUL1IdWNKrJEhXVgowt%2BmWsgY6swIbHTQ2dzp%2Fagy9MD7m6AXIfohMthZ4AmDhl2RZw2dKgEnnoG5xX598yHdV5G5dmbcOLCWfUA1dPtiZOyAWYHselSAWBYpDBbrpeDUg8xo%2Bk%2F9zU8QoOGB9vz%2FYJ2nOft2TT4Nqc0sEd0mlZ5cG656ccdxFdAOIjG3Au2CbPwGB6onhCfVDMpZqk12HH8kWq469KON9ZPImaLvwvxBleIW2Ig1D4%2F9A8CvsXFWgBsfnTZYXYUi7jXMhZ7S5KxYgk2wjUSYPiLPOnMztA%2FdntirTR4smRyMOL9fLRwMpimr2Uxay2vXIUAW%2BZNHA7%2FpvmQMfqVOe42u80FB3xC2Chprte0M2dkJKOvVQCgbvVqTB6jCiXLQphs1uUDxDo%2B8oRgYuCK51UlMeLx4dm5xrV%2BUALYAw&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240516T075848Z&X-Amz-SignedHeaders=host&X-Amz-Expires=12000&X-Amz-Credential=ASIAVRUVS32BESWDOL5R%2F20240516%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Signature=9b9585215b50bbf5a8429c055328e4502fc7973204bcd47325e9725f85055eee",
        "token_string": "cedf28f7-d0dd-4091-ade7-765296dd5fe9",
        "caption_prefix": "a photo of cedf28f7-d0dd-4091-ade7-765296dd5fe9 cheese",
        "max_train_steps": 1000,
        "use_face_detection_instead": False
    },
    destination="bamboufranck/fossi"
)


training.reload()
print(training.status)
print("\n".join(training.logs.split("\n")[-10:]))

print("end")
