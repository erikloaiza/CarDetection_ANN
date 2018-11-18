import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import imutils  


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            #yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
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
#Written by Ross Girshick
def detect_Objects(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if(img.shape==(64,64)):
        h = getHog().compute(img)
        data_test=h.reshape(1,len(h))
        _ret,resp=ann.predict(data_test)
        #Get just the objects if the probabilistic criteria match
        if resp[0][0]>1.025 and abs(resp[0][0]-resp[0][1])>0.9: #resp[0][1]<0.01: 
            print (resp)
            return True
        else:
            return False
    else:
        return False

def pyramid(image, scale=1.2, minSize=(64, 64)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image

#From https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def main():
    img = cv2.imread("./Test/test_original/010.jpg") ##Funcion que devuelve las imagenes para entrenamiento
    wsize = 64
    hmap = img.copy()
    hmap = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hmap[:][:] = 0
    for im in pyramid(img):
            overlay = hmap.copy()
            for (x,y,window) in sliding_window(im,(int)(wsize*0.25),(wsize,wsize)):
                if detect_Objects(window):
                    t=hmap.copy()
                    t[:][:] = 0
                    x1= (int)(translate(x,0,im.shape[0],0,img.shape[0]))
                    y1 = (int)(translate(y,0,im.shape[1],0,img.shape[1]))
                    scale_w = (int)(translate(wsize,0,im.shape[1],0,img.shape[1]))
                    (x2,y2) = (x1+(int)(scale_w),y1+(int)(scale_w))
                    cv2.rectangle(t,(x1,y1),(x2,y2),255,-1)
                    cv2.addWeighted(t, 0.1, overlay,0.9,0,overlay)
            cv2.addWeighted(overlay, 0.5, hmap,0.5,0,hmap)
    hmap2 = hmap.copy()
    np.place(hmap2,hmap<hmap.max()*0.5,0)
    _,contours,_ = cv2.findContours(hmap2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    hmap =cv2.applyColorMap(hmap,cv2.COLORMAP_HOT)
    cv2.imwrite('obj.png',img)
    cv2.imshow("result",img)
    cv2.imshow("Heatmap",hmap)
    cv2.waitKey()#Comentar


if __name__ == '__main__':
    #Variables Globales
    ann = cv2.ml.ANN_MLP_load("hog_ann_mlp.yml")
    main()
