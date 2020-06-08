import numpy as np
from skimage import feature as skif
import cv2
def lbp_histogram(image,P=8,R=1,method = 'nri_uniform'):
    '''
    image: shape is N*M 
    '''
    lbp = skif.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
    # cv2.imwrite("lbp.png",lbp)
    max_bins = int(lbp.max() + 1) # max_bins is related P
    hist,_= np.histogram(lbp,  normed=True, bins=max_bins, range=(0, max_bins))
    return hist
# file_list is a txt file, like this:
# image_path label
def save_features(file_list,file_name):
    feature_label = []
    for line in open(file_list):
        image_path = line.strip().split(' ')[0]
        label = int(line.strip().split(' ')[1])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_h = lbp_histogram(image[:,:,0]) # y channel
        cb_h = lbp_histogram(image[:,:,1]) # cb channel
        cr_h = lbp_histogram(image[:,:,2]) # cr channel
        feature = np.concatenate((y_h,cb_h,cr_h))
        feature_label.append(np.append(feature,np.array(label)))
    np.save(file_name,np.array(feature_label))
if __name__ == "__main__":
    save_features("/data/test.txt","test_feature.npy")