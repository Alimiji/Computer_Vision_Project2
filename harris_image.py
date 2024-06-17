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
    d["n"] = (2*r + 1)**2*im.shape[2] # Nombre de valeurs dans le descripteur
    d["data"] = np.zeros((d["n"],), dtype=float)

    # Valeur du pixel central
    cval = im[pos[0], pos[1], :]

    # Limite du voisinage
    r0 = pos[0] - r if pos[0] - r > 0 else 0
    r1 = pos[0] + r + 1 if pos[0] + r + 1 < im.shape[0] else im.shape[0]-1
    c0 = pos[1] - r if pos[1] - r > 0 else 0
    c1 = pos[1] + r + 1 if pos[1] + r + 1 < im.shape[1] else im.shape[1]-1

    # Extraction et normalisation des valeurs
    values = (im[r0:r1, c0:c1, :].astype(float) - cval).ravel()

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


    # Calcul des dérivées gaussiennes Ix et Iy avec un lissage gaussien
    
    Ix = gaussian_filter(im, (0, 1), order=(0, 1), sigma=sigma)
    Iy = gaussian_filter(im, (1, 0), order=(1, 0), sigma=sigma)

    # Initialisation du tenseur de structure
    S = np.zeros((*im.shape, 3))

    # Somme pondérée des gradients locaux
    for i in range(im.shape[0]):
        
        for j in range(im.shape[1]):
    
            for r in range(-1, 2):                     # Fenêtre de taille 3x3 par défaut
                for s in range(-1, 2):
                    
                    if(r != 0 and s != 0):
                        # Vérification des limites de l'image
                        if 0 <= i - r < im.shape[0] and 0 <= j - s < im.shape[1]:
                        
                            weight = np.exp(-(r**2 + s**2) / (2 * sigma**2)) # pondération gaussienne
                            
                            # Accumulation des termes pondérés dans le tenseur de structure
                            S[i, j, 0] += weight * Ix[i-r, j-s]**2
                            S[i, j, 1] += weight * Iy[i-r, j-s]**2
                            S[i, j, 2] += weight * Ix[i-r, j-s] * Iy[i-r, j-s]

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

    # Calcul du déterminant du tenseur de structure
    det_S = np.linalg.det(S)

    # Calcul de la trace du tenseur de structure
    trace_S = np.trace(S, axis1=0, axis2=1)

    # Calcul de la cornerness pour chaque pixel en utilisant la formule spécifiée
    R = det_S - alpha * (trace_S**2)

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
            
            # Obtention de la valeur du pixel central
            pixel_central = r[l, c]
            
            # Vérifier si la valeur du pixel central est le maximum dans la fenêtre
            if pixel_central < np.max(fen):
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

    # Calculate structure matrix
    S = structure_matrix(img, sigma)

    # Estimate cornerness
    R = cornerness_response(S)

    # Run NMS on the responses
    Rnms = nms_image(R, nms)

    # TODO: Comptez le nombre de réponses au-dessus d'un seuil thresh: ok 
    count = 0 # changez ceci: ok
    for l in Rnms.shape[0]:
        for c in Rnms.shape[1]:
            if(Rnms[l,c] > thresh):
                count += 1
            

    n = count # <- fixer n = nombre de coins dans l'image: ok
    d = []
    d = np.zeros(n) # initialisation du tableau d 
    # TODO: remplir le tableau d avec le descripteur de chaque coin. Utilisez describe_index().: ok
    count = 0
    for l in Rnms.shape[0]:
        for c in Rnms.shape[1]:
            if(Rnms[l,c] > thresh):
                descripteur = describe_point(Rnms, [(l,c)])
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