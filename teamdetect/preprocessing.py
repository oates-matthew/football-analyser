import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def flatten(img):
    return img.flatten()

 
def standardise(img):
    return img - np.mean(img) / np.std(img)


def resize(img, target_size=(128, 64)):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def crop_image(frame, tlwh):
    l, t, w, h = tlwh
    t, l, w, h = int(t), int(l), int(w), int(h)
    b, r = t + h, l + w

    img = frame[t:b, l:r]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img


def preprocess(img):
    img_copy = resize(img)
    img_copy = standardise(img_copy)
    img_copy = flatten(img_copy)
    return img_copy


def save_img(images, teams, frame_id, detection_nos, bins=16):

    for i, img in enumerate(images):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
        ax1.imshow(img)
        ax1.set_title('Player detected team {}'.format(teams[i]))
        ax1.axis('off')

        colors = ('y', 'b', 'r')
        for j, color in enumerate(colors):
            ax2.hist(img[:, :, j].ravel(), bins=bins, color=color, range=(0, 255), alpha=0.5)
        ax2.set_title('Color Histogram')
        ax2.set_xlabel('Intensity Value')
        ax2.set_ylabel('Count')

        plt.tight_layout()

        filename = "plots/team{}/frame{}/detection{}.png".format(teams[i], frame_id, detection_nos[i])
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename)
        plt.close(fig)
