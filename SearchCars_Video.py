import cv2
import numpy as np
import os
import glob
import imutils  
import threading
import time

def pyramid(image, scale=1.5, minSize=(64, 64)):
    yield image
    while image.shape[0] > minSize[1] and image.shape[1] > minSize[0]:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        yield image



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
    img = cv2.equalizeHist(img)
    if(img.shape==(64,64)):
        h = getHog().compute(img)
        data_test=h.reshape(1,len(h))
        _ret,resp=ann.predict(data_test)
        #Get just the objects if the probabilistic criteria match
        if resp[0][0]>1 and abs(resp[0][0]-resp[0][1])>1:
            return True
        else:
            return False
    else:
        return False

#From https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
    
def createHeatmap(img,im,hmap):
    wsize = 64
    overlay = hmap.copy()
    for (x,y,window) in sliding_window(im,(int)(wsize*0.25),(wsize,wsize)):
        if detect_Objects(window):
            t=hmap.copy()
            t[:][:] = 0
            x1= (int)(translate(x,0,im.shape[0],0,img.shape[0]))
            y1 = (int)(translate(y,0,im.shape[1],0,img.shape[1]))
            scale_w = (int)(translate(wsize,0,im.shape[1],0,img.shape[1]))
            (x2,y2) = (x1+(int)(scale_w),y1+(int)(scale_w))
            cv2.rectangle(t,(x1,y1),(x2,y2),(255),-1)
            cv2.addWeighted(t, 0.1, overlay,0.9,0,overlay)
            #cv2.rectangle(overlay,(x1,y1),(x2,y2),(255),-1)
    cv2.addWeighted(overlay, 0.5, hmap,0.5,0,hmap)
    #cv2.addWeighted(overlay, 0.1, hmap,0.9,0,hmap)

def main():
    ret, img = vid.read()
    if(img is None):
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, img = vid.read()
    img = cv2.resize(img,None,fx=vscale, fy=vscale, interpolation = cv2.INTER_LINEAR)
    hmap = img.copy()
    hmap = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hmap[:][:] = 0
    jobs = []
    for im in pyramid(img):
        createHeatmap(img,im,hmap)
#Multithreading doesn't improve speed
##        thread = threading.Thread(target=createHeatmap(img,im,hmap))
##        jobs.append(thread)
##    # Start the threads (i.e. calculate the random number lists)
##    for j in jobs:
##        j.start()
##	# Ensure all of the threads have finished
##    for j in jobs:
##        j.join()
    
    hmap2 = hmap.copy()
    np.place(hmap2,hmap<hmap.max()*0.65,0)
    #if(hmap2.max() == np.where(hmap2!=0).min()):
    #    hmap[:][:] = 0
    _,contours,_ = cv2.findContours(hmap2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if(w/h>0.6 and h/w>0.4 and w>50 and h>50):
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    hmap =cv2.applyColorMap(hmap,cv2.COLORMAP_HOT)

    (px,py) = ((int)(img.shape[0]*0.05),(int)(img.shape[1]*0.05))

    cv2.rectangle(img,(0,py-(int)(30*vscale)),(px+(int)(1050*vscale),py+(int)(40*vscale)),0,-1)
    cv2.putText(img,"Select a scale: 1=100%, 2=75%, 3=50%",(px,py), cv2.FONT_HERSHEY_SIMPLEX,1*vscale,(255,255,255))
    cv2.putText(img,"Actions: 'q'=quit, 'r'=restart, 'c'=change video '.'=move forward",(px,(int)(py+(30*vscale))), cv2.FONT_HERSHEY_SIMPLEX,1*vscale,(255,255,255))

    cv2.imshow("Heatmap",hmap)
    cv2.imshow("Filtered Heatmap",hmap2)
    cv2.imshow("result",img)
    #frame = (int)(vid.get(cv2.CAP_PROP_POS_FRAMES))
    #cv2.imwrite('./Temp/'+str(frame)+'.png',img)
    #cv2.waitKey()#Comentar

def noProcess():
    while(cv2.waitKey(0) & 0xFF == ord('.')):
        ret, img = vid.read()
        if(img is None):
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, img = vid.read()
        img = cv2.resize(img,None,fx=vscale, fy=vscale, interpolation = cv2.INTER_LINEAR)
        (px,py) = ((int)(img.shape[0]*0.05),(int)(img.shape[1]*0.05))
        cv2.rectangle(img,(0,py-(int)(30*vscale)),(px+(int)(800*vscale),py+(int)(40*vscale)),0,-1)
        cv2.putText(img,"keep '.' button pressed to move forward",(px,py), cv2.FONT_HERSHEY_SIMPLEX,1*vscale,(255,255,255))
        cv2.putText(img,"Press any other button to resume detections",(px,(int)(py+(30*vscale))), cv2.FONT_HERSHEY_SIMPLEX,1*vscale,(255,255,255))
        cv2.imshow("result",img)

def get_videos(path):
    videos = []
    for vid in glob.glob(path + "*.mp4"):
        vi = cv2.VideoCapture(vid)
        videos.append(vi)
    return videos

if __name__ == '__main__':
    #Variables Globales
    ann = cv2.ml.ANN_MLP_load("hog_ann_mlp.yml")
    videos = get_videos('./Videos/')
    i=0
    vid = videos[i]
    vscale=0.75
    while True:
        k= cv2.waitKey(1) & 0xFF
        if k== ord('q'):
            break
        elif k== ord('1'):
            vscale=1
        elif k== ord('2'):
            vscale=0.75
        elif k== ord('3'):
            vscale=0.5
        elif k== ord('r'):
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif k== ord('.'):
            noProcess()
        elif k== ord('c'):
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            i+=1
            if i>len(videos)-1:
                i=0
            vid = videos[i]
        main()
    os._exit(0)
