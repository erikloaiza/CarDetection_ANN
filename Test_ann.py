import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt

def get_images(path):
    images = []
    for img in glob.glob(path + "*.jpg"):
        im = cv2.imread(img,0)
        im = cv2.equalizeHist(im)
        images.append(im) 
    return images

def getHog():
    cell_size = (16, 16)  # h x w in pixels
    block_size = (1, 1)  # h x w in cells
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    # cell_size is the size of the cells of the img patch over which to calculate the histograms
    # block_size is the number of cells which fit in the patch
    return cv2.HOGDescriptor(_winSize=(64 // cell_size[1] * cell_size[1],
                                      64 // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

def main():
    
    path_class1_test = "./Test/"
    imgs_test = get_images(path_class1_test) ##Funcion que devuelve las imagenes para entrenamiento
    ann = cv2.ml.ANN_MLP_load("hog_ann_mlp.yml")

    hog = getHog()
    data_test = np.array([])
    for im in imgs_test:
        h = hog.compute(im)
        if len(data_test)==0:
            data_test=h.reshape(1,len(h))
        else:
            data_test=np.vstack((data_test, h.reshape(1,len(h))))
    
    print('testing...')
    _ret,resp=ann.predict(data_test)
    print(resp)

if __name__ == '__main__':
    main()