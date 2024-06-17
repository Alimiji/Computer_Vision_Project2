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
    a = np.array(a)
    b = np.array(b)
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
        best matches found. Each descriptor in a should match with at most
        one other descriptor in b.
    """
    an = len(a)
    bn = len(b)
    matches = []

    for j in range(an):
        # Find the best match in b for each descriptor in a.
        bind = min(range(bn), key=lambda k: l1_distance(a[j]['pos'], b[k]['pos']))

        match = {
            'ai': j,
            'bi': bind,  # Index in b.
            'p': a[j]['pos'],
            'q': b[bind]['pos'],
            'distance': l1_distance(a[j]['pos'], b[bind]['pos'])  # L1 distance.
        }
        matches.append(match)  # Append the match dictionary to the list.

    seen = set()
    filtered_matches = []

    # Ensure matches are injective (one-to-one).
    for match in matches:
        if match['q'] not in seen:
            seen.add(match['q'])
            filtered_matches.append(match)

    # Sort matches based on distance.
    filtered_matches.sort(key=lambda x: x['distance'])

    return filtered_matches


def find_and_draw_matches(a: np.ndarray, b: np.ndarray, sigma: float = 2, thresh: float = 3,
                          nms: int = 3) -> np.ndarray:
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

    #print("Image a: "  + str(a))
    #print("Image b: " + str(b))
    # Detect corners in image 'a' using Harris corner detector
    ad = harris_corner_detector(a, sigma, thresh, nms) # array

    print("Liste des corners de l'image a: " + str(ad))

    # Detect corners in image 'b' using Harris corner detector
    bd = harris_corner_detector(b, sigma, thresh, nms) # array

    print("Liste des corners de l'image b: " + str(bd.tolist()))

    # Match descriptors between the two sets of corners
    m = match_descriptors(ad.tolist(), bd.tolist()) # recoit deux listes et renvoit une liste
    print("Liste de match des descripteurs: " + str(m))

    # Mark corners in image 'a' using the detected corners and their count
    a = mark_corners(a, ad.tolist(), len(ad))

    # Mark corners in image 'b' using the detected corners and their count
    b = mark_corners(b, bd.tolist(), len(bd))

    # Draw lines connecting the matched corners between images 'a' and 'b'
    lines = draw_matches(a, b, m, 10)

    # Return the resulting image with inliers (lines connecting matched corners)
    return lines

def project_point(H, point):
    """
    Project a 2D point using a homography matrix H.

    Parameters:
    - H: np.ndarray
        Homography matrix (3x3).
    - point: np.ndarray
        Point to be projected, represented as a column vector [x, y].

    Returns:
    - np.ndarray
        Projected point.
    """
    # Ensure the point is a column vector with a homogeneous coordinate
    point = np.array([point[0], point[1], 1.0])

    # Perform the matrix multiplication
    projected_point = np.dot(H, point)

    # Normalize the homogeneous coordinates
    projected_point /= projected_point[2]

    # Return the 2D coordinates
    return projected_point[:2]


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


import numpy as np


def compute_homography(matches: list, n: int) -> np.ndarray:
    """Computes homography between two images given matching pixels.

    Parameters
    ----------compute_
    matches: list
        Matching points between images.
    n: int
        Number of matches to use in calculating homography.

    Returns
    -------
    H: np.ndarray
        Matrix representing homography H that maps image a to image b.
    """
    assert n >= 4, "Underdetermined, use n >= 4"

    M = np.zeros((n * 2, 8))
    b = np.zeros((n * 2, 1))

    for i in range(n):
        r, c = matches[i]['p']  # mx = r, my = c
        rp, cp = matches[i]['q']  # nx = rp, ny = cp

        # Fill in the matrices M and b
        M[i * 2, :] = [r, c, 1, 0, 0, 0, -r * rp, -c * rp]
        M[i * 2 + 1, :] = [0, 0, 0, r, c, 1, -r * cp, -c * cp]

        # Fill in the matrix b
        b[i * 2, 0] = rp  # nx
        b[i * 2 + 1, 0] = cp  # ny

    # Solve the linear system
    if M.shape[0] == M.shape[1]:
        a = np.linalg.solve(M, b)
    else:  # Over-determined, using least-squares
        a = np.linalg.lstsq(M, b, rcond=None)[0]

    # If a solution can't be found, return None
    if a is None:
        return None

    # Fill in the homography H based on the result in a
    H = np.array([[float(a[0]), float(a[1]), float(a[2])],
                  [float(a[3]), float(a[4]), float(a[5])],
                  [float(a[6]), float(a[7]), 1.0]])

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
        n = len(subset_matches)
        current_H = compute_homography(subset_matches, n)
        
        # Identification of inliers based on the computed homography and threshold

        (nb_inliers , inliers_matches) = model_inliers(current_H, subset_matches, thresh)#[0:]
        inliers = inliers_matches[0:nb_inliers]
         
        # if new homography is better than old (how can you tell?): :ok
      
        if len(inliers) > best:
            # Compute updated homography using all inliers: ok
            Hb = compute_homography(inliers, len(inliers))
            best = len(inliers)
            
            #         if it's better than the cutoff: ok
            #             return it immediately: ok
            
            if best > cutoff:
                return Hb
        
        # if we get to the end return the best homography: ok

    return Hb


def combine_images(a, b, H):
    """Stitches two images together using a projective transformation.
    Parameters
    ----------
    a, b: ndarray
        Images to stitch.
    H: ndarray
        Homography from image a coordinates to image b coordinates.
    Returns
    -------
    c: ndarray
        Combined image stitched together.
    """
    Hinv = np.linalg.inv(H)

    # Project the corners of image b into image a coordinates.
    c1 = project_point(Hinv, [0, 0])
    c2 = project_point(Hinv, [b.shape[0], 0])
    c3 = project_point(Hinv, [0, b.shape[1]])
    c4 = project_point(Hinv, [b.shape[0], b.shape[1]])

    topleft = [0, 0]
    botright = [0, 0]
    botright[0] = int(max([c1[0], c2[0], c3[0], c4[0]]))
    botright[1] = int(max([c1[1], c2[1], c3[1], c4[1]]))
    topleft[0] = int(min([c1[0], c2[0], c3[0], c4[0]]))
    topleft[1] = int(min([c1[1], c2[1], c3[1], c4[1]]))

    dr = int(min(0, topleft[0]))
    dc = int(min(0, topleft[1]))
    h = int(max(a.shape[0], botright[0]) - dr)
    w = int(max(a.shape[1], botright[1]) - dc)

    if w > 7000 or h > 7000:
        print("Output too big, stopping.")
        return np.copy(a)

    c = np.zeros((h, w, a.shape[2]), dtype=a.dtype)

    for k in range(a.shape[2]):
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if 0 <= i - dr < h and 0 <= j - dc < w:
                    c[i - dr, j - dc, k] = a[i, j, k]

    for k in range(b.shape[2]):
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                x_c = j
                y_c = i
                p = [x_c + dr, y_c + dc, k]
                projected_point = project_point(H, p)
                x_b, y_b = projected_point[:2]
                x_b_proche = int(round(x_b))
                y_b_proche = int(round(y_b))

                if 0 <= x_b_proche < b.shape[1] and 0 <= y_b_proche < b.shape[0]:
                    c[i, j, k] = b[y_b_proche, x_b_proche, k]

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