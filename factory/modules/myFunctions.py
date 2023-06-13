# to convert a number to a roman numeral
def number_to_roman(number):
    roman_mapping = {
        1000: 'M',
        900: 'CM',
        500: 'D',
        400: 'CD',
        100: 'C',
        90: 'XC',
        50: 'L',
        40: 'XL',
        10: 'X',
        9: 'IX',
        5: 'V',
        4: 'IV',
        1: 'I'
    }

    roman_numeral = ''
    for value, symbol in roman_mapping.items():
        while number >= value:
            roman_numeral += symbol
            number -= value

    return roman_numeral


import os
import shutil


# Move all .heic files from a source folder to a destination folder
def move_files(source_folder, destination_folder, file_extension='.heic'):


    # Create the destination folder if it doesn't exist
    # os.makedirs(destination_folder, exist_ok=True)

    # Get all files in the source folder
    files = os.listdir(source_folder)

    # Filter only .heic files
    heic_files = [f for f in files if f.endswith(file_extension)]

    # Move each .heic file to the destination folder
    for file in heic_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)

fromf = '/Users/stephandekker/workspace/pink_lady/storage/images/heic_apples'
tof = '/Users/stephandekker/workspace/pink_lady/storage/images/apple_extended_unedited/Test/Normal_Apple'

move_files(fromf, tof, '.jpg')