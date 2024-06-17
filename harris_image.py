import numpy as np
from scipy.ndimage import gaussian_filter




def describe_point(im: np.ndarray, pos: list) -> dict:
    """Crée un descripteur de caractéristique pour un point l'image
    Parameters
    ----------
    im: ndarray
        Image source
    pos: (2,) list
        Position (r,c) dans l'image qu'on souhaite décrire
    Returns
    -------
    d: dict
        Descripteur pour cet indice.
    """
    r = 2 # Rayon du voisinage
    
    # Descripteur
    d = dict()
    d["pos"] = pos
    d["n"] = (2*r + 1)**2*im.shape[1] # Nombre de valeurs dans le descripteur
    d["data"] = np.zeros((d["n"],), dtype=float)

    # Valeur du pixel central
    cval = im[pos[0], pos[1]]

    # Limite du voisinage
    r0 = pos[0] - r if pos[0] - r > 0 else 0
    r1 = pos[0] + r + 1 if pos[0] + r + 1 < im.shape[0] else im.shape[0]-1
    c0 = pos[1] - r if pos[1] - r > 0 else 0
    c1 = pos[1] + r + 1 if pos[1] + r + 1 < im.shape[1] else im.shape[1]-1

    # Extraction et normalisation des valeurs
    #values = (im[r0:r1, c0:c1, :].astype(float) - cval).ravel()
    values = (im[r0:r1, c0:c1].astype(float) - cval).ravel()

    # Intégration dans le descripteur
    d['data'][0:len(values)] = values

    return d

def mark_spot(im: np.ndarray, p: list, color: list = [255,0,255]) -> np.ndarray:
    """ Marque la position d'un point dans l'image.
    Parameters
    ----------
    im: ndarray
        Image à marquer
    p: (2,) list
        Position (r,c) du point
    color: (3,) list
        Couleur de la marque
    Returns
    -------
    im: ndarray
        Image marquée.
    """
    r = p[0]
    c = p[1]

    for i in range(-9,10):
        if r+i < 0 or r+i >= im.shape[0] or c+i < 0 or c+i >= im.shape[1]:
            continue # ce pixel est à l'extérieur de l'image
        im[r+i, c, 0] = color[0]
        im[r+i, c, 1] = color[1]
        im[r+i, c, 2] = color[2]
        im[r, c+i, 0] = color[0]
        im[r, c+i, 1] = color[1]
        im[r, c+i, 2] = color[2]

    return im

def mark_corners(im: np.ndarray, d: list, n: int) -> np.ndarray:
    """ Marks corners denoted by an array of descriptors.
    Parameters
    ----------
    im: ndarray
        Image à marquer
    d: list
        Coins dans l'image
    n: int
        Nombre de descripteurs à marquer
    Returns
    -------
    im: ndarray
        Image marquée
    """
    m = np.copy(im)
    for i in range(n):
        m = mark_spot(m, d[i]['pos'])
    return m

def smooth_image(im: np.ndarray, sigma: float) -> np.ndarray:
    """Lissage d'une image avec un filtre gaussien.
    Parameters
    ----------
    im: ndarray
        Image à traiter
    sigma: float
        Écart-type pour la gaussienne.
    Returns
    -------
    s: ndarray
        Image lissée
    """
    s = gaussian_filter(im, sigma)
    return s

def structure_matrix(im: np.ndarray, sigma: float) -> np.ndarray:         #ok#1

    """Calcul du tenseur de structure d'un image.
        Parameters
        ----------
        im: ndarray
            Image à traiter (tons de gris et normalisée entre 0 et 1).
        sigma: float
            Écart-type pour la somme pondérée
        Returns
        -------
        S: ndarray
            Tenseur de structure. 1er canal est Ix^2, 2e canal est Iy^2
            le 3e canal est IxIy
        """
    S = np.zeros((*im.shape, 3))
    # TODO: calcul du tenseur de structure pour im.

    # Calcul des dérivées


    Ix = gaussian_filter(im, (0, 1), order=(0, 1))
    Iy = gaussian_filter(im, (1, 0), order=(1, 0))

    # Étape 2 : Calcul des métriques correspondantes

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Étape 3 : Lissage gaussien à chaque terme

    Ixx_smoothed = smooth_image(Ixx, sigma)
    Iyy_smoothed = smooth_image(Iyy, sigma)
    Ixy_smoothed = smooth_image(Ixy, sigma)

    # Paramètres du filtre gaussien

   # Marge pour la fenêtre
    marge = 1

    # Calcul des pondérations gaussiennes pour chaque terme

    poids_xx = np.exp(-(np.arange(-marge, marge + 1) ** 2) / (2 * sigma ** 2))
    poids_yy = np.exp(-(np.arange(-marge, marge + 1) ** 2) / (2 * sigma ** 2))
    poids_xy = np.exp(-(np.arange(-marge, marge + 1) ** 2) / (2 * sigma ** 2))

    # Normalisation des poids gaussiens pour chaque terme

    poids_xx /= np.sum(poids_xx)
    poids_yy /= np.sum(poids_yy)
    poids_xy /= np.sum(poids_xy)

    # Application de la pondération gaussienne sur chaque terme

    Ixx_pond = Ixx_smoothed
    Iyy_pond = Iyy_smoothed
    Ixy_pond = Ixy_smoothed

    # Initialisation des sommes pondérées
    Sxx_sum = np.zeros_like(Ixx)
    Syy_sum = np.zeros_like(Iyy)
    Sxy_sum = np.zeros_like(Ixy)

   # Calcul des sommes pondérées avec la fenêtre centrée

    for y in range(marge, im.shape[0] - marge):
        for x in range(marge, im.shape[1] - marge):
            Sxx_sum[y, x] = np.sum(Ixx_pond[y - marge:y + 1 + marge, x - marge:x + 1 + marge])
            Syy_sum[y, x] = np.sum(Iyy_pond[y - marge:y + 1 + marge, x - marge:x + 1 + marge])
            Sxy_sum[y, x] = np.sum(Ixy_pond[y - marge:y + 1 + marge, x - marge:x + 1 + marge])


    S[:, :, 0] = Sxx_sum
    S[:, :, 1] = Syy_sum
    S[:, :, 2] = Sxy_sum



    return S








def cornerness_response(S: np.ndarray) -> np.ndarray:                   #???#2
    """Estimation du cornerness de chaque pixel avec le tenseur de structure S.
    Parameters
    ----------
    S: ndarray
        Tenseur de structure de l'image
    Returns
    -------
    R: ndarray
        Une carte de réponse de la cornerness
    """
    R = np.zeros(S.shape[0:2])
    # TODO: Remplir R avec la "cornerness" pour chaque pixel en utilisant le tenseur de structure.
    # On utilise la formulation det(S) - alpha * trace(S)^2, alpha = 0.06
    alpha = 0.06
    #print("Dimension du tenseur de structure: " + str(S.shape))

    # Calcul du déterminant du tenseur de structure
    Ixx = S[:, :, 0]
    Iyy = S[:, :, 1]
    Ixy = S[:, :, 2]
    det_S =  Ixx * Iyy - Ixy ** 2
    # Calcul de la trace du tenseur de structure
    trace_S = Ixx + Iyy

    # Calcul de la cornerness pour chaque pixel en utilisant la formule spécifiée
    R = det_S - alpha * trace_S ** 2

    return R

def nms_image(im: np.ndarray, w: int) -> np.ndarray:             #?????????? #3
    """Effectue la supression des non-maximum sur l'image des feature responses.
    Parameters
    ----------
    im: ndarray
        Image 1 canal des réponses de caractéristiques (feature response)
    w: int
        Distance à inspecter pour une grande réponse
    Returns
    -------
    r: ndarray
        Image contenant seulement les maximums locaux pour un voisinage de w pixels.
    """
    r = np.copy(im)
    # TODO: faire NMS sur la carte de réponse
    # Pour chaque pixel dans l'image: ok
    #     Pour chaque voisin dans w: ok
    #         Si la réponse du voisin est plus grande que la réponse du pixel: ok
    #             Assigner à ce pixel une très petite réponse (ex: -np.inf): ok
    #nb_lignes = im.shape[0] : 
    #nb_colonnes = im.shape[1]
    for l in range(1, im.shape[0]-1):
        for c in range(1, im.shape[1]-1):
            # Extraction de la fenêtre centrée sur le pixel (l, c)
            fen = im[l - w:l + w + 1, c - w:c + w + 1]

            # Vérifier que la fenêtre n'est pas vide
            if fen.size > 0:
                # Obtention de la valeur du pixel central
                pixel_central = r[l, c]

                # Vérifier si la valeur du pixel central est le maximum dans la fenêtre
                if pixel_central < np.max(fen) and not np.any(np.isinf(fen)):
                    # Suppression  du pixel central en lui assignant une valeur négative très basse
                    r[l, c] = -np.inf
            
    return r


def harris_corner_detector(im: np.ndarray, sigma: float, thresh: float, nms: int) -> np.ndarray: # ???#4
    """ Détecteur de coin de Harris, et extraction des caractéristiques autour des coins.
    Parameters
    ----------
    im: ndarray
        Image à traiter (RGB).
    sigma: float
        Écart-type pour Harris.
    thresh: float
        Seuil pour le cornerness
    nms: int
        Distance maximale à considérer pour la supression des non-maximums
    Returns
    -------
    d: list
        Liste des descripteurs pour chaque coin dans l'image
    """
    img = im.mean(axis=2) # Convert to grayscale
    img = (img.astype(float) - img.min()) / (img.max() - img.min())

    print("Dimension de l'image: " + str(im.shape))

    # Calculate structure matrix
    S = structure_matrix(img, sigma)

    # Estimate cornerness
    R = cornerness_response(S)

    # Run NMS on the responses
    Rnms = nms_image(R, nms)

    # TODO: Comptez le nombre de réponses au-dessus d'un seuil thresh: ok 
    count = 0 # changez ceci: ok
    for l in range(Rnms.shape[0]):
        for c in range(Rnms.shape[1]):
            if(Rnms[l,c] > thresh):
                count += 1
            

    n = count # <- fixer n = nombre de coins dans l'image: ok
    d = []
    # Initialiser le tableau d avec des objets de type object
    d = np.zeros(n, dtype=object)

    count = 0
    for l in range(Rnms.shape[0]):
        for c in range(Rnms.shape[1]):
            if Rnms[l, c] > thresh:
                descripteur = describe_point(Rnms, (l, c))
                d[count] = descripteur
                count += 1

    return d

def detect_and_draw_corners(im: np.ndarray, sigma: float, thresh: float, nms: int) -> np.ndarray:
    """ Trouve et dessine les coins d'une image
    Parameters
    ----------
    im: ndarray
        L'image à traiter (RGB).
    sigma: float
        Écart-type pour Harris.
    thresh: float
        Seuil pour le cornerness.
    nms: int
        Distance maximale à considérer pour la supression des non-maximums
    Returns
    m: ndarray
        Image marqué avec les coins détectés
    """
    d = harris_corner_detector(im, sigma, thresh, nms)
    m = mark_corners(im, d, len(d))
    return m