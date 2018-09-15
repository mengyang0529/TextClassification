import smallDataSet as sds

if __name__=="__main__":
    trianPath = "data/txt/training-data-small.txt"
    testPath = "data/txt/test-data-small.txt"
    sds.preview(trianPath)
    Y_train,X_train,X_test = sds.padding(trianPath,testPath)
    clf = sds.train(Y_train,X_train)
    sds.predict(clf,X_test)
