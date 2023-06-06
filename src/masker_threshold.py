import cv2
import numpy as np

# Lees de afbeelding in
image = cv2.imread('../project3/apple_disease_classification/Test/Blotch_Apple/124.jpg')

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

# Thresholding op het grijswaardenmasker om de vorm van de appel te verbeteren
_, thresholded_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Pas morfologische bewerkingen toe om het masker te verbeteren (optioneel)
kernel = np.ones((5, 5), np.uint8)
thresholded_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_OPEN, kernel)
thresholded_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_CLOSE, kernel)

# Maak een maskerbeeld van de oorspronkelijke afbeelding
masked_image = cv2.bitwise_and(image, image, mask=thresholded_mask)

# Toon de oorspronkelijke afbeelding, het maskerbeeld en het thresholded masker
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)
cv2.imshow('Thresholded Mask', thresholded_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
