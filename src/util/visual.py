# Zweck: Hilfsfunktionen zur Visualisierung von Bildern + Labels
# Author: David Kostka
# Datum: 05.11.2020

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import util
import numpy

def plot_bbox(img, bbox, conf_threshold=0.5):
    plt.imshow(img, cmap='gray')

    x, y, xm, ym = bbox[:4].astype('int')
    is_obj = bbox[4]
    box_clr = [is_obj, 0, 0, is_obj]
    txt_clr = [1, 1, 1, is_obj]
    
    if is_obj > conf_threshold:
        bbox = Rectangle((x, y), xm - x, ym - y, linewidth=2, edgecolor=box_clr, facecolor='none')
        plt.gca().add_patch(bbox)
        plt.text(x, y, str([x, y, xm, ym, is_obj]), color=txt_clr, bbox=dict(facecolor=box_clr, alpha=0.5))
    else:
        plt.text(5, 5, 'No Object found, conf: ' + str(is_obj), color='white', bbox=dict(facecolor=box_clr, alpha=0.5))

    plt.show()

def plot_roc(roc_auc, fpr, tpr, optimal_threshold=None, optimal_idx=None):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    if optimal_threshold and optimal_idx:
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro')
        plt.text(fpr[optimal_idx] + 0.02, tpr[optimal_idx] - 0.05, 'trsh: ' + '%.3f' % optimal_threshold)
        plt.text(fpr[optimal_idx] + 0.02, tpr[optimal_idx] - 0.1, 'tpr: ' + '%.3f' % tpr[optimal_idx])
        plt.text(fpr[optimal_idx] + 0.02, tpr[optimal_idx] - 0.15, 'fpr: ' + '%.3f' % fpr[optimal_idx])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
