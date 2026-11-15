"""Where is the pitch? A colour mask that removes people standing off the field.

SoccerNet only annotates people on the field: a COCO person detector also fires on
coaches, substitutes, ball boys and spectators, which are false positives here.
The visible part of a planar convex pitch, seen by a pinhole camera that has the
whole pitch in front of it, is convex: we take the convex hull of the largest
green region, which also fills the holes left by players and white lines.
"""
from __future__ import annotations

import cv2
import numpy as np

from .boxes import foot_points


# --------------------------------------------------------------------------- pitch mask
def pitch_mask(frame_bgr: np.ndarray, hsv_low=(35, 40, 40), hsv_high=(85, 255, 255),
               close_px: int = 25, scale: float = 0.5) -> np.ndarray:
    """Binary pitch mask (H, W) of dtype uint8 with values in {0, 1}.

    Steps, computed on a copy downscaled by `scale` for speed: (1) HSV threshold
    (OpenCV hue is in [0, 180)); (2) morphological closing with an elliptic kernel of
    about close_px pixels at full resolution; (3) keep the largest connected component;
    (4) replace it by its convex hull; (5) resize back with nearest-neighbour
    interpolation. A frame without any green pixel gives an all-zero mask.
    """
    # Récupérer la hauteur et la largeur de l'image originale
    h, w = frame_bgr.shape[:2]

    # 1. Redimensionner l'image pour accélérer les traitements
    # INTER_AREA est adapté à la réduction de résolution
    small = (
        cv2.resize(
            frame_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
        if scale != 1.0 else frame_bgr
    )

    # 2. Convertir l'image de BGR vers HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Sélectionner les pixels dont la couleur appartient à l'intervalle vert
    # green est un masque binaire de valeurs 0 ou 255
    green = cv2.inRange(
        hsv,
        np.array(hsv_low, np.uint8),
        np.array(hsv_high, np.uint8)
    )

    # 3. Construire un noyau elliptique pour la fermeture morphologique
    # Adapter sa taille à la résolution réduite et imposer une taille impaire
    k = max(1, int(round(close_px * scale)) | 1)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k)
    )

    # Appliquer une fermeture morphologique pour combler les petits trous
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 4. Identifier les composantes connexes du masque binaire
    # Transformer le masque en valeurs 0 et 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (green > 0).astype(np.uint8),
        connectivity=8
    )

    # Si aucun pixel vert n'est détecté, retourner un masque entièrement nul
    if n <= 1:
        return np.zeros((h, w), np.uint8)

    # Ignorer la composante 0 (arrière-plan)
    # Sélectionner la composante de plus grande aire
    largest = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )

    # Extraire les pixels de la plus grande composante
    # et calculer leur enveloppe convexe
    hull = cv2.convexHull(
        cv2.findNonZero(
            (labels == largest).astype(np.uint8)
        )
    )

    # Créer un masque vide à la résolution réduite
    mask = np.zeros(small.shape[:2], np.uint8)

    # Remplir l'enveloppe convexe avec la valeur 1
    cv2.fillConvexPoly(mask, hull, 1)

    # 5. Redimensionner le masque à la résolution originale
    # INTER_NEAREST préserve les valeurs binaires 0 et 1
    return cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )
    


def distance_to_mask(points, mask: np.ndarray) -> np.ndarray:
    """Euclidean distance in pixels from each point (u, v) to the mask (0 inside it).

    Points outside the image are clipped to the nearest pixel first.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0)
    if not mask.any():
        return np.full(len(pts), np.inf)
    # distanceTransform: distance of every non-zero pixel to the nearest zero pixel
    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = mask.shape
    cols = np.clip(np.floor(pts[:, 0]).astype(int), 0, w - 1)
    rows = np.clip(np.floor(pts[:, 1]).astype(int), 0, h - 1)
    return dist[rows, cols].astype(np.float64)


def foot_distance_to_pitch(boxes, mask: np.ndarray) -> np.ndarray:
    """Distance in pixels from each box's foot point to the pitch mask (0 if on the pitch).

    Stored with every cached detection, so that any tolerance radius can be tested
    offline (keep a detection if distance <= radius).
    """
    # the foot point sits ON the bottom edge: step half a pixel up into the box
    return distance_to_mask(foot_points(boxes) - np.array([0.0, 0.5]), mask)
