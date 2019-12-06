# CNN
CNN code from scratch on CIFAR10 dataset

#ex to run code:
dataobj = Data() <br/>
(trainData, trainLabel), (testData, testLabel) = dataobj.load_data()
model = CNN()
model.addconv(10, (3, 3, 3), "ReLU", (32, 32, 3))
model.addpool((2, 2), 2)
model.addconv(8, (3, 3, 10), "ReLU")
model.addpool((2, 2), 2)
model.addDense(80, "ReLU")
model.addDense(10, "Softmax")
model.summary()
start = time.time()
trainedModel = model.fit(trainData, trainLabel, epoch=30, savelayer=True)       # epoch =2
end = time.time()
print("Time taken: ", (end-start), "seconds")
model.history()

trainedModel = open('model', 'rb')
layers = pickle.load(trainedModel)
print(layers)
model.predict(layers, testData, testLabel)
