import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve, auc
import matplotlib.pyplot as plt 
def metric(predict_proba,labels):
    predict =  np.greater(predict_proba[:,1],0.5)
    tn, fp, fn, tp = confusion_matrix(labels,predict).flatten()
    acc = (tp+tn)/(tp+tn+fp+fn)
    far = fp / (fp + tn) # apcer
    frr = fn / (tp + fn) # bpcer
    hter=(far+frr) / 2 # acer

    fpr, tpr, threshold = roc_curve(labels, predict_proba[:,1])
    auc_v = auc(fpr, tpr) # area under curve
    dist = abs((1-fpr) - tpr)
    eer = fpr[np.argmin(dist)]
    plt.plot(fpr, tpr, label='area under curve(auc): %0.2f' % auc_v)
    plt.plot([0, 1], [1, 0])
    plt.plot([eer, eer], [0,tpr[np.argmin(dist)]],label = '@EER',linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend()
    plt.savefig("roc.png")
    return acc,eer,hter
