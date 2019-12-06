# CNN
CNN code from scratch on CIFAR10 dataset

# ex to run code: 
dataobj = Data() <br/>
(trainData, trainLabel), (testData, testLabel) = dataobj.load_data() <br/>
model = CNN() <br/>
model.addconv(10, (3, 3, 3), "ReLU", (32, 32, 3))<br/>
model.addpool((2, 2), 2)<br/>
model.addconv(8, (3, 3, 10), "ReLU")<br/>
model.addpool((2, 2), 2)<br/>
model.addDense(80, "ReLU")<br/>
model.addDense(10, "Softmax")<br/>
model.summary()<br/>
start = time.time()<br/>
trainedModel = model.fit(trainData, trainLabel, epoch=30, savelayer=True)<br/>
end = time.time()<br/>
print("Time taken: ", (end-start), "seconds")<br/>
model.history()<br/>

trainedModel = open('model', 'rb')<br/>
layers = pickle.load(trainedModel)<br/>
print(layers)<br/>
model.predict(layers, testData, testLabel)<br/>
