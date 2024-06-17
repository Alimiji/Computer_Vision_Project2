from harris_image import detect_and_draw_corners
from panorama_image import panorama_image
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import plotly.express as px
import skimage

print("Génial !")
def afficher_image(image_path, nom_image):

    print("Affichage de la version originale de l'image " + nom_image)

    image = cv2.imread(image_path)
    if image is None:
        print("Erreur lors de la lecture de l'image.")
        return

    if image.ndim != 3 or image.shape[2] != 3:
        print("L'image n'est pas en format couleur ou a une forme inattendue.")
        return

    # Affichage de l'image originale
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = px.imshow(image_rgb)
    fig.show()

afficher_image("Rainier2.png", "Test affichage d'image")