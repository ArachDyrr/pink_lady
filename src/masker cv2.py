import cv2
import numpy as np

# Lees de afbeelding in
image = cv2.imread('../../roject3/apple_disease_classification/Test/Blotch_Apple/125.jpg')
"C:\Users\jiyoo\workspace\MakeAIWork3\project3\apple_disease_classification\Train\Rot_Apple\2c45k3y.jpg"

if image is None:
    print("Kan de afbeelding niet laden")
    exit()


# Definieer de ondergrens en bovengrens van de kleur van de appel in HSV-kleurruimte
lower_color = np.array([0, 100, 100])  # Pas de waarden aan op basis van de appelkleur
upper_color = np.array([20, 255, 255])  # Pas de waarden aan op basis van de appelkleur

# Converteer de afbeelding naar HSV-kleurruimte
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Creëer een masker met behulp van de kleurgrenzen
mask = cv2.inRange(hsv_image, lower_color, upper_color)

# Pas morfologische bewerkingen toe om het masker te verbeteren (optioneel)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Maak een maskerbeeld van de oorspronkelijke afbeelding
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Toon de oorspronkelijke afbeelding en het maskerbeeld
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
