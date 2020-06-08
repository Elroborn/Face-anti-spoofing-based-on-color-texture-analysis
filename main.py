import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from metric import metric
def load_feature_label(file_name):
    feature_label = np.load(file_name)
    return feature_label[:,:-1],feature_label[:,-1].astype(np.uint8)
def train():
    train_feature,train_label = load_feature_label("train_feature.npy")
    model = SVC(kernel='rbf', C=1e3, gamma=0.5, class_weight='balanced', probability=True)
    model.fit(train_feature, train_label)
    joblib.dump(model, "./model.m")
    predict_proba = model.predict_proba(train_feature)
    predict = model.predict(train_feature)
    acc,eer,hter = metric(predict_proba,train_label)
    print("train acc is:%f eer is:%f hter is:%f"%(acc,eer,hter))
def test():
    test_feature,test_label = load_feature_label("test_feature.npy")
    model = joblib.load("./model.m")
    predict_proba = model.predict_proba(test_feature)
    predict = model.predict(test_feature)
    acc,eer,hter = metric(predict_proba,test_label)
    print("test acc is:%f eer is:%f hter is:%f"%(acc,eer,hter))
if __name__ == "__main__":
    # train()
    test()