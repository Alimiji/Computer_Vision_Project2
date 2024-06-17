from harris_image import detect_and_draw_corners, harris_corner_detector
from panorama_image import panorama_image
import numpy as np
import cv2
import imageio.v2 as imageio

import plotly.express as px
import skimage
import matplotlib.pyplot as plt


# Fonction permettant d'afficher une image


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

# Test d'affichage d'une image

afficher_image("Rainier1.png", "Test affichage d'image")

###############################################################################################################
# Détection des coins de Harris
###############################################################################################################

im = imageio.imread("Rainier1.png")
im_cd = detect_and_draw_corners(im, 2, 0.0004, 3)
plt.imshow(im_cd)
plt.savefig('image1_coins_detectes.png')




im2 = imageio.imread("Rainier2.png")
im2_cd = detect_and_draw_corners(im2, 2, 0.0004, 3)
plt.imshow(im2_cd)
plt.savefig('image2_coins_detectes.png')

#b = imageio.imread("Rainier2.png")

#bd = harris_corner_detector(b, 2, 0.0004, 3)
#print("Corners détectés au niveau de l'image b: " + str(bd.tolist()))
###############################################################################################################
# Ajustement de la projection aux données: pairages des caractéristiques
###############################################################################################################


import matplotlib.pyplot as plt
import imageio.v2 as imageio
from panorama_image import find_and_draw_matches

# Charger les images
a = imageio.imread("Rainier1.png")
b = imageio.imread("Rainier2.png")

# Trouver et dessiner les correspondances entre les images
result_image = find_and_draw_matches(a, b, 2, 0.0004, 3)

# Enregistrer l'image résultante dans un fichier
plt.imshow(result_image)
plt.savefig("result_image_pairages_caracteristique.png")

# Afficher un message indiquant que l'image a été sauvegardée
print("Image résultante sauvegardée sous le nom 'result_image_pairages_caracteristique.png'")




###############################################################################################################
# Création du panorama
################################################################################################################


import matplotlib.pyplot as plt
from panorama_image import panorama_image
import imageio.v2 as imageio

# Use imageio.v2.imread instead of imageio.imread
im1 = imageio.imread('Rainier1.png')
im2 = imageio.imread('Rainier2.png')
im3 = imageio.imread('Rainier3.png')
im4 = imageio.imread('Rainier4.png')
im5 = imageio.imread('Rainier5.png')
im6 = imageio.imread('Rainier6.png')

# Panorama obtenu à partir des images 1 et 2

pan = panorama_image(im1, im2)
plt.imshow(pan)
plt.savefig("result_image_panorama12")

# Panorama obtenu à partir des images 2 et 3
pan = panorama_image(im2, im3)
plt.imshow(pan)
plt.savefig("result_image_panorama23")


# Panorama obtenu à partir des images 3 et 4
pan = panorama_image(im3, im4)
plt.imshow(pan)
plt.savefig("result_image_panorama34")

# Panorama obtenu à partir des images 4 et 5
#pan = panorama_image(im4, im5)
#plt.imshow(pan)
#plt.savefig("result_image_panorama45")

# Panorama obtenu à partir des images 1 et 6
pan = panorama_image(im1, im6, 2, 0.00025)
plt.imshow(pan)
plt.savefig("result_image_panorama16")


# Panorama obtenu à partir des images 1 et 6
pan = panorama_image(im5, im6, 2, 0.0005, 3, 0.01, 1000, 10)
plt.imshow(pan)
plt.savefig("result_image_panorama56")



# Panorama obtenu à partir des images 1 et 6
pan = panorama_image(im1, im5, 2, 0.00025)
plt.imshow(pan)
plt.savefig("result_image_panorama15")

# Panorama obtenu à partir des images 1 et 4
pan = panorama_image(im1, im4, 2, 0.00025)
plt.imshow(pan)
plt.savefig("result_image_panorama14")


# Panorama obtenu à partir des images 2 et 4
pan = panorama_image(im2, im4, 2, 0.0005)
plt.imshow(pan)
plt.savefig("result_image_panorama24")