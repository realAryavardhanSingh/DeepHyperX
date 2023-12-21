from utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "DFC2018_HSI": {
        "img": "2018_IEEE_GRSS_DFC_HSI_TR.HDR",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
        "download": False,
        "loader": lambda folder: dfc2018_loader(folder),
    },

    "Randy01": {
        "fname": "Randy01.mat",
        "download": False,
        "loader": lambda folder: Randy_loader(folder),
    },

    "Randy02": {
        "fname": "Test_5-6_7-9.mat",
        "download": False,
        "loader": lambda folder: Randy_loader(folder),
    },

    "Randy03": {
        "fname": "Train_1-3_2-4.mat",
        "download": False,
        "loader": lambda folder: Randy_loader(folder),
    },

    "Randy04": {
        "fname": "Test_5-6_7-9.mat",
        "download": False,
        "loader": lambda folder: Randy_loader(folder),
    }
}


def Randy_loader(folder):
    version = folder[-3:-1]
    folder = folder[:-3] + folder[-1]
    Randy_file = open_file(folder + CUSTOM_DATASETS_CONFIG["Randy"+version]["fname"])
    img = Randy_file["im"]  #e.g. 600pixels x 500 pixels x 462 bands
    gt = Randy_file["gt"]   #e.g. 600pixels x 500 pixels x 1category... category will be 0 to 7, where 0 means unlabelled, 1-7 are our classes.
    gt = gt.astype("uint8")
    rgb_bands = (199-1, 124-1, 63-1)  #Resonon sensor
    ignored_labels = [0, 4, 5]
    label_values = list(map(str, np.arange(8)))  #0 is unlabelled 1-7 are the 7 classes 
    palette = None

    #preprocess
    #img = img / img[:, :, [150]]
    #img[np.isnan(img)] = 0
    
    return img, gt, rgb_bands, ignored_labels, label_values, palette


def dfc2018_loader(folder):
    img = open_file(folder + "2018_IEEE_GRSS_DFC_HSI_TR.HDR")[:, :, :-2]
    gt = open_file(folder + "2018_IEEE_GRSS_DFC_GT_TR.tif")
    gt = gt.astype("uint8")

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unclassified",
        "Healthy grass",
        "Stressed grass",
        "Artificial turf",
        "Evergreen trees",
        "Deciduous trees",
        "Bare earth",
        "Water",
        "Residential buildings",
        "Non-residential buildings",
        "Roads",
        "Sidewalks",
        "Crosswalks",
        "Major thoroughfares",
        "Highways",
        "Railways",
        "Paved parking lots",
        "Unpaved parking lots",
        "Cars",
        "Trains",
        "Stadium seats",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette
