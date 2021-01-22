# Zweck: Hilfsfunktionen für Datenverarbeitung und Training eines Models
# Author: David Kostka
# Datum: 05.11.2020

def unnormalize_bbox(bbox, size):
    '''
    Skaliert normalisiere BBOX Koordinaten in Pixel Koordinaten um
    Input: (xmin, ymin, xmin, ymax), (img_height, img_width)
    Werte von 0.0 bis 1.0

    Output: (xmin, ymin, xmin, ymax)
    Skalierte Werte von 0 bis size
    '''
    bbox[..., 0] *= size[1]
    bbox[..., 2] *= size[1]
    bbox[..., 1] *= size[0]
    bbox[..., 3] *= size[0]
    return bbox.copy()

def normalize_bbox(bbox, size):
    '''
    Liefert Normalisiere BBOX Koordinaten
    Input: (xmin, ymin, xmax, ymax), (img_height, img_width)
    Werte von 0 bis size

    Output: (xmin, ymin, xmin, ymax)
    Werte von 0.0 bis 1.0
    '''
    bbox[..., 0] /= size[1]
    bbox[..., 2] /= size[1]
    bbox[..., 1] /= size[0]
    bbox[..., 3] /= size[0]
    return bbox.copy()