import cv2
import glob
import numpy as np



      
def get_images(path):
    images = []
    for img in glob.glob(path + "*.png"):
        im = cv2.imread(img)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2YCR_CB)
        images.append(im)

    return images

def getTrainData(imgClass1,imgClass2):

    cell_size = (16, 16)  # h x w in pixels
    block_size = (1, 1)  # h x w in cells
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    # cell_size is the size of the cells of the img patch over which to calculate the histograms
    # block_size is the number of cells which fit in the patch
    hog = cv2.HOGDescriptor(_winSize=(64 // cell_size[1] * cell_size[1],
                                      64 // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)


    data_train = np.array([])
    label_train=np.array([])
    for img in imgClass1:
        h = hog.compute(img)
        if len(data_train)==0:
            data_train=h.reshape(1,len(h))
            label_train=np.float32(1.0)
            label_train=np.vstack((label_train, np.float32(0.0)))
        else:
            data_train=np.vstack((data_train, h.reshape(1,len(h))))
            label_train=np.vstack((label_train, np.float32(1.0)))
            label_train=np.vstack((label_train, np.float32(0.0)))
    for img in imgClass2:
        h = hog.compute(img)
        data_train=np.vstack((data_train, h.reshape(1,len(h))))
        label_train=np.vstack((label_train, np.float32(0.0)))
        label_train=np.vstack((label_train, np.float32(1.0)))
    label_train = label_train.reshape(len(imgClass1)+len(imgClass2),2)
    return (data_train,label_train)

def main():
    path_class1_0 = "./Cars/vehicles/Far/"   
    path_class1_1 = "./Cars/vehicles/Left/" 
    path_class1_2 = "./Cars/vehicles/Right/" 
    path_class1_3 = "./Cars/vehicles/MiddleClose/"
    path_class1_4 = "./Cars/vehicles/KITTI_extracted/"

    path_class2_0 = "./Cars/non-vehicles/Far/" 
    path_class2_1 = "./Cars/non-vehicles/Left/" 
    path_class2_2 = "./Cars/non-vehicles/Right/" 
    path_class2_3 = "./Cars/non-vehicles/MiddleClose/"
    path_class2_4 = "./Cars/non-vehicles/Extras/"

    vehicle_imgs = get_images(path_class1_0) + get_images(path_class1_1) + get_images(path_class1_2)  + get_images(path_class1_3)+ get_images(path_class1_4)##Funcion que devuelve las imagenes para entrenamiento
    nonvehicle_imgs = get_images(path_class2_0) + get_images(path_class2_1) + get_images(path_class2_2) + get_images(path_class2_3) + get_images(path_class2_4) ##Funcion que devuelve las imagenes para entrenamiento
    (data_train,label_train) = getTrainData(vehicle_imgs,nonvehicle_imgs)
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([data_train.shape[1],32,2], dtype= np.uint16))
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    ann.setBackpropMomentumScale(0.0)
    ann.setBackpropWeightScale(0.001) #(0.001) (0.1) %El valor por defecto sugerido por opencv es 0.01
    ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
    ann.train(data_train,cv2.ml.ROW_SAMPLE,label_train)
    ann.save("hog_ann_mlp.yml")

if __name__ == '__main__':
    main()
