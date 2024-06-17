from harris_image import harris_corner_detector, mark_corners
import numpy as np
import random


def make_translation_homography(dr: float, dc: float) -> np.ndarray:
    """Create a translation homography
    Parameters
    ----------
    dr: float
        Translation along the row axis
    dc: float
        Translation along the column axis
    Returns
    -------
    H: np.ndarray
        Homography as a 3x3 matrix
    """
    H = np.zeros((3,3))
    H[0,0] = 1
    H[1,1] = 1
    H[2,2] = 1
    H[0,2] = dr # Row translation
    H[1,2] = dc # Col translation

def match_compare(a: float, b: float) -> int:
    """ Comparator for matches
    Parameters
    ----------
    a,b : float
        distance for each match to compare.
    Returns
    -------
    result of comparison, 0 if same, 1 if a > b, -1 if a < b.
    """
    comparison = 0
    if a < b:
        comparison = -1
    elif a > b:
        comparison = 1
    else:
        comparison = 0
    return comparison

def both_images(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Place two images side by side on canvas, for drawing matching pixels.
    Parameters
    ----------
    a,b: ndarray
        Images to place
    Returns
    -------
    c: ndarray
        image with both a and b side-by-side.
    """
    width = a.shape[1] + b.shape[1]
    height = a.shape[0] if a.shape[0] > b.shape[0] else b.shape[0]
    channel = a.shape[2] if a.shape[2] > b.shape[2] else b.shape[2]
    
    both = np.zeros((height,width,channel), dtype=a.dtype)
    both[0:a.shape[0], 0:a.shape[1],0:a.shape[2]] = a
    both[0:b.shape[0], a.shape[1]:a.shape[1]+b.shape[1],0:b.shape[2]] = b
    
    return both

def draw_matches(a: np.ndarray, b: np.ndarray, matches: list, inliers: int) -> np.ndarray:
    """Draws lines between matching pixels in two images.
    Parameters
    ----------
    a, b: ndarray
        two images that have matches.
    matches: list
        array of matches between a and b.
    inliers: int
        number of inliers at beginning of matches, drawn in green.
    Returns
    -------
    c: ndarray
        image with matches drawn between a and b on same canvas.
    """
    both = both_images(a, b)
    n = len(matches)
    for i in range(n):
        r1 = matches[i]['p'][0] # Coordonnée y du point p
        r2 = matches[i]['q'][0] # Coordonnée y du point q
        c1 = matches[i]['p'][1] # Coordonnée x du point p
        c2 = matches[i]['q'][1] # Coordonnée x du point q
        for c in range(c1, c2 + a.shape[1]):
            r = int((c-c1)/(c2 + a.shape[1] - c1)*(r2 - r1) + r1)
            both[r, c, 0] = (0 if i<inliers else 255)
            both[r, c, 1] = (255 if i<inliers else 0)
            both[r, c, 2] = 0
    return both

def draw_inliers(a: np.ndarray, b: np.ndarray, H: np.ndarray, matches: list, thresh: float) -> np.ndarray:
    """ Draw the matches with inliers in green between two images.
    Parameters
    ----------
    a, b: ndarray
        two images to match.
    H: ndarray
        Homography matrix
    matches: list
        array of matches between a and b
    thresh: float
        Threshold to define inliers
    Returns
    -------
    lines: ndarray
        Modified images with inliers
    """
    n_inliers, new_matches = model_inliers(H, matches, thresh)
    lines = draw_matches(a, b, new_matches, n_inliers)
    return lines


def find_and_draw_matches(a: np.ndarray, b: np.ndarray, sigma: float=2, thresh: float=3, nms: int=3) -> np.ndarray:
    """ Find corners, match them, and draw them between two images.
    Parameters
    ----------
    a, b: np.ndarray
         images to match.
    sigma: float
        gaussian for harris corner detector. Typical: 2
    thresh: float
        threshold for corner/no corner. Typical: 1-5
    nms: int
        window to perform nms on. Typical: 3
    Returns
    -------
    lines: np.ndarray
        Images with inliers
    """
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)
    m = match_descriptors(ad, bd)

    a = mark_corners(a, ad, len(ad))
    b = mark_corners(b, bd, len(bd))
    lines = draw_matches(a, b, m, 0)

    return lines

def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates L1 distance between to floating point arrays.
    Parameters
    ----------
    a, b: list or np.ndarray
        arrays to compare.
    Returns
    -------
    l1: float
        l1 distance between arrays (sum of absolute differences).
    """
    l1 = 0
    # TODO: return the correct number.:  ok
    l1 = np.sum(np.abs(a-b))

    return l1


def match_descriptors(a: list, b: list) -> list:
    """Finds best matches between descriptors of two images.
    Parameters
    ----------
    a, b: list
        array of descriptors for pixels in two images.
    Returns
    -------
    matches: list
        best matches found. each descriptor in a should match with at most
        one other descriptor in b.
    """
    an = len(a)
    bn = len(b)
    matches = []
    dmin = 0
    for j in range(an):
        # TODO: for every descriptor in a, find best match in b.: ok
        
        # record ai as the index in a and bi as the index in b.: ok
        bind = 0 # <- find the best match: ok
        
       # p =  (1/a[j]['pos'][-1]) * a[j]['pos']   # Normalisation du point de l'image a' 
        
        for k in range(bn):
            
           # q =  (1/b[k]['pos'][-1]) * b[k]['pos']  # Normalisation du point de l'image b'
            
            if(dmin < l1_distance(a[j]['pos'], b[k]['pos'])):
                dmin = l1_distance(a[j]['pos'], b[k]['pos'])
                bind = k
                
        matches[j]['ai'] = j
        matches[j]['bi'] = bind # <- should be index in b.
        matches[j]['p'] = a[j]['pos']
        matches[j]['q'] = b[bind]['pos']
        matches[j]['distance'] = dmin # <- should be the smallest L1 distance!
            
        
        """
                matches[j]['ai'] = j
                matches[j]['bi'] = bind # <- should be index in b.
                matches[j]['p'] = a[j]['pos']
                matches[j]['q'] = b[bind]['pos']
                matches[j]['distance'] = 0 # <- should be the smallest L1 distance!
                """
    seen = [] 
    
    # TODO: we want matches to be injective (one-to-one).: ok
    # Then throw out matches to the same element in b. Use seen to keep track.: ok
    # Construction de l'injection: elimination des descripteurs dont le point de l'image b a 
    # a déjà un match dans l'image a
    for descript in matches:
        
        if descript['q'] not in seen:
            
            seen.append(descript['q'])
        else:
            matches.remove(descript)
        
    filtered_matches = []
    
    # Sort matches based on distance using match_compare and sort: ok
    # Each point should only be a part of one match.: ok
    # Some points will not be in a match.: ok
    # In practice just bring good matches to front of list.: ok
    i_min = 0
    # descripteur_interm = matches[0]
    # Triage des descripteurs selon leur distance
    for i in range(an):
        i_min = i
        
        for j in range(i, an):
            if(matches[i_min]["distance"] > matches[j]["distance"]):
                i_min = j
        filtered_matches.append(matches[i_min])
        
        """
        if(i_min != i):
            descripteur_interm = matches[i]
            matches[i] = matches[i_min]
            matches[i_min] = descripteur_interm
            """

    matches = filtered_matches

    return matches

def project_point(H, p):
    """ Apply a projective transformation to a point.
    Parameters
    ----------
    H: np.ndarray
        homography to project point, of shape 3x3
    p: list
        point to project.
    Returns
    -------
    q: list
        point projected using the homography.
    """

    c = np.zeros((3, 1))
    # TODO: project point p with homography H.: ok
    # Remember that homogeneous coordinates are equivalent up to scalar.: ok
    # Have to divide by.... something...: ok
    # Transformer la liste en une matrice colonne : ok
    c = np.array(p).reshape(-1, 1)
    
    produit_matr = np.dot(H, c)         # Projection du point
    produit_matr = (1/produit_matr[-1, 0])*produit_matr # Normalisation du point

    q = [0, 0]
    q = [produit_matr[0, 0], produit_matr[1, 0]]

    return q

def point_distance(p, q):
    """ Calculate L2 distance between two points.
    Parameters
    ----------
    p, q: list
        points.
    Returns
    -------
    l2: float
        L2 distance between them.
    """
    l2 = 0
    # TODO: should be a quick one.: ok
    liste_carre_diff = [(a - b)**2 for a, b in zip(p, q)]
    l2 = np.sqrt(np.sum(liste_carre_diff))
    return l2

def model_inliers(H: np.ndarray, matches: list, thresh: float) -> tuple:
    """Count number of inliers in a set of matches. Should also bring inliers to the front of the array.
    Parameters
    ----------
    H: np.ndarray
        homography between coordinate systems.
    matches: list
        matches to compute inlier/outlier.
    thresh: float
        threshold to be an inlier.
    Returns
    -------
    count: int
        number of inliers whose projected point falls within thresh of their match in the other image.
    matches: list
        Should also rearrange matches so that the inliers are first in the array. For drawing.
    """
    """
     matches.append({})
        matches[j]['ai'] = j
        matches[j]['bi'] = bind # <- should be index in b.
        matches[j]['p'] = a[j]['pos']
        matches[j]['q'] = b[bind]['pos']
        matches[j]['distance'] = 0 # <- should be the smallest L1 distance!
    
    """
    count = 0
    new_matches = [] # To reorder the matches : ok
    # TODO: count number of matches that are inliers: ok
    # i.e. distance(H*p, q) < thresh : ok
    # Also, sort the matches m so the inliers are the first 'count' elements.: ok
    
    # Ajout des descripteurs inliers en premieres positions dans la liste: new_matches
    for j in range(len(matches)):
        p = project_point(H, matches[j]["p"])
        q = matches[j]["q"]
        if(l1_distance(p, q) < thresh):
            count += 1
            new_matches.append(matches[j])
    # Ajout des descripteurs outliers en dernieres positions
    new_matches += [descripteur for descripteur in matches if descripteur not in new_matches]

    return (count, new_matches)


def randomize_matches(matches: list) -> list:
    """ Randomly shuffle matches for RANSAC.
    Parameters
    ----------
    matches: list
        matches to shuffle in place
    Returns
    -------
    shuffled_matches: list
        Shuffled matches
    """
    """
    Algo de Fisher-Yates:
        Pour mélanger un tableau a de n éléments (indicés de 0 à n-1),
        l'algorithme est le suivant.

     Pour i allant de n − 1 à 1 faire :
       j ← entier aléatoire entre 0 et i
       échanger a[j] et a[i]
       
     source: https://fr.wikipedia.org/wiki/M%C3%A9lange_de_Fisher-Yates
    """
    # TODO: implement Fisher-Yates to shuffle the array.: ok
    descripteur_interm = dict()
    
    for i in range(len(matches)):
        
        j = int(i * random.random())
        
        descripteur_interm = matches[i]
        matches[i] = matches[j]
        matches[j] = descripteur_interm
    
    return matches


def compute_homography(matches: list, n: int) -> np.ndarray:  # ok
    """Computes homography between two images given matching pixels.
    Parameters
    ----------
    matches: list
        matching points between images.
    n: int
        number of matches to use in calculating homography.
    Returns
    -------
    H: np.ndarray
        matrix representing homography H that maps image a to image b.
    """
    assert n >= 4, "Underdetermined, use n>=4"

    M = np.zeros((n*2,8))
    b = np.zeros((n*2,1))

    for i in range(n):
        r  = float(matches[i]['p'][0]) # mx = r
        rp = float(matches[i]['q'][0]) # nx = rp
        c  = float(matches[i]['p'][1]) # my = c
        cp = float(matches[i]['q'][1]) # ny = cp
        # TODO: fill in the matrices M and b.: ok
        # Remplissage de la matrice M
        
        M[i*2, :] = [r, c, 1, 0, 0, 0, -r * rp, -c * rp]
        M[i*2 + 1, :] = [0, 0, 0, r, c, 1, -r * cp, -c * cp]
        
        # Remplissage de la matrice b
        
        b[i*2, 1] = rp   # nx
        b[i*2 + 1, 1] = cp  # ny
        

    # Solve the linear system
    if M.shape[0] == M.shape[1]:
        a = np.linalg.solve(M, b)
    else: # Over-determined, using least-squared
        a = np.linalg.lstsq(M,b,rcond=None)
        a = a[0]
    # If a solution can't be found, return empty matrix;
    if a is None:
        return None

    H = np.zeros((3,3))
    # TODO: fill in the homography H based on the result in a.
    H[0, :] = [a[0], a[1], a[2]]
    H[1, :] = [a[3], a[4], a[5]]
    H[2, :] = [a[6], a[7], 1]

    return H

def RANSAC(matches: list, thresh: float, k: int, cutoff: int):
    """Perform RANdom SAmple Consensus to calculate homography for noisy matches.
    Parameters
    ----------
    matches: list
        set of matches.
    thresh: float
        inlier/outlier distance threshold.
    k: int
        number of iterations to run.
    cutoff: int
        inlier cutoff to exit early.
    Returns
    -------
    Hb: np.ndarray
        matrix representing most common homography between matches.
    """
    best = 0
    Hb = make_translation_homography(0, 256) # Initial condition
    # TODO: fill in RANSAC algorithm.: ok
    # for k iterations: ok
    #     shuffle the matches :
    #     compute a homography with a few matches (how many??)
   
    
    #         remember it and how good it is
    
    """
    for j in range(len(matches)):
        p = project_point(H, matches[j]["p"])
        q = matches[j]["q"]
        if(l1_distance(p, q) < thresh):
            count += 1
            new_matches.append(matches[j])
    
    """
    
   
    for i in range(k):
        # Shuffle the matches
        matches = randomize_matches(matches)
        
        # Select a subset of matches to calculate homography
        subset_matches = matches[:4]
        
        # Compute homography with the subse  =  model_inliers(current_H, subset_matches, thresh)[0]t of matches
        current_H = compute_homography(subset_matches)
        
        # Identification of inliers based on the computed homography and threshold
        
        (nb_inliers , inliers_matches) = model_inliers(current_H, subset_matches, thresh)#[0:]
        inliers = inliers_matches[0:nb_inliers]
         
        # if new homography is better than old (how can you tell?): :ok
      
        if len(inliers) > best:
            # Compute updated homography using all inliers: ok
            Hb = compute_homography(inliers)
            best = len(inliers)
            
            #         if it's better than the cutoff: ok
            #             return it immediately: ok
            
            if best > cutoff:
                return Hb
        
        # if we get to the end return the best homography: ok

    return Hb

def combine_images(a, b, H):
    """ Stitches two images together using a projective transformation.
    Parameters
    ----------
    a, b: ndarray
        Images to stitch.
    H: ndarray
        Homography from image a coordinates to image b coordinates.
    Returns
    -------
    c: ndarray
        combined image stitched together.
    """
    Hinv = np.linalg.inv(H)

    # Project the corners of image b into image a coordinates.
    # Coins de l'image b
    c1 = project_point(Hinv, [0, 0])
    c2 = project_point(Hinv, [b.shape[0], 0])
    c3 = project_point(Hinv, [0, b.shape[1]])
    c4 = project_point(Hinv, [b.shape[0], b.shape[1]])

    # Find top left and bottom right corners of image b warped into image a.
    # Détermination des coordonnées topleft et botright délimitant la région 
    #d'image nécessaire pour contenir les deux images assemblées.
    
    topleft = [0,0]
    botright = [0,0]
    botright[0] = int(max([c1[0], c2[0], c3[0], c4[0]]))
    botright[1] = int(max([c1[1], c2[1], c3[1], c4[1]]))
    topleft[0]  = int(min([c1[0], c2[0], c3[0], c4[0]]))
    topleft[1]  = int(min([c1[1], c2[1], c3[1], c4[1]]))

    # Find how big our new image should be and the offsets from image a.
    # Détermination des dimensions du panorama: image résultante des deux images
    # a et b
    dr = int(min(0, topleft[0]))
    dc = int(min(0, topleft[1]))
    h = int(max(a.shape[0], botright[0]) - dr)
    w = int(max(a.shape[1], botright[1]) - dc)

    # Can disable this if you are making very big panoramas.
    # Usually this means there was an error in calculating H.
    # Vérification de la dimension de l'image
    if w > 7000 or h > 7000:
        print("output too big, stopping.")
        return np.copy(a)
    # Initialisation de la matrice du panorama (c)
    
    c = np.zeros((h,w,a.shape[2]), dtype=a.dtype)
    
    # Paste image a into the new image offset by dr and dc.
    for k in range(a.shape[2]):
        for j in range(a.shape[1]):
            for i in range(a.shape[0]):
                # TODO: remplir l'image: ok
                # Vérification de l'emplacement des coordonnées, s'ils sont à l'intérieur 
                # des limites de l'image c
                if 0 <= i - dr < h and 0 <= j - dc < w:
               # Copie la valeur du pixel de l'image a dans la nouvelle image c
                   c[i - dr, j - dc, k] = a[i, j, k]

               # pass
    # TODO: Paste in image b as well.
    # You should loop over some points in the new image (which? all?)
    # and see if their projection from a coordinates to b coordinates falls
    # inside of the bounds of image b. If so, use interpolation
    # PPV (nearest neighbours)
    # estimate the value of b at that projection, then fill in image c.
    # Boucle sur tous les canaux de couleur, colonnes et lignes de la nouvelle image c
    for k in range(c.shape[2]):
        for j in range(c.shape[1]):
            for i in range(c.shape[0]):
            # TODO: Ajoutez votre code ici pour traiter chaque pixel de la nouvelle image c
                # Projection des points de C (coordonnées de a) vers les coordonnées de b
                # Coordonnées dans le système de coordonnées de la nouvelle image c
                x_c = j
                y_c = i
                
                p = [x_c + dr, y_c + dc, k]
                x_b, y_b, z_b = project_point(H, p)
                # Verification si les coordonnées obtenus par projection se trouve à l'intérieur de
                # la limite de l'image b
                if 0 <= x_b < b.shape[1] and 0 <= y_b < b.shape[0]:
                    # interpolation PPV (nearest neighbours) des projections se trouvant dans la limite de 
                    # l'image b
                    x_b_proche = int(round(x_b))
                    y_b_proche = int(round(y_b))
                    #z_b_proche = int(round(z_b))
                    
                pixel_b = b[y_b_proche, x_b_proche, k]

                # Assignation de la valeur estimée à la position correspondante dans la nouvelle image c
                c[x_c, y_c, k] = pixel_b
              
                
    
    return c

def panorama_image(a, b, sigma=2, thresh=0.0003, nms=3, inlier_thresh=10, iters=1000, cutoff=15):
    """ Create a panoramam between two images.
    Parameters
    ----------
    a, b: ndarray
        images to stitch together.
    sigma: float
        gaussian for harris corner detector. Typical: 2
    thresh: float
        threshold for corner/no corner. Typical: 0.0001-0.0005
    nms: int
        window to perform nms on. Typical: 3
    inlier_thresh: float
        threshold for RANSAC inliers. Typical: 2-5
    iters: int
        number of RANSAC iterations. Typical: 1,000-50,000
    cutoff: int
        RANSAC inlier cutoff. Typical: 10-100
    """
    # Calculate corners and descriptors
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)

    # Find matches
    m = match_descriptors(ad, bd)
    
    # Run RANSAC to find the homography
    H = RANSAC(m, inlier_thresh, iters, cutoff)

    # Stitch the images together with the homography
    comb = combine_images(a, b, H)
    return comb