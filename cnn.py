import numpy as np
import os
import pickle
import time
import seaborn as sns


class Data:

    __slots__ = "datasetPath", "classes", "cifardata", "trainData", "trainLabel", "testData", "testLabel"

    def __init__(self):
        self.datasetPath = os.getcwd() + "\\" + "cifar-10-batches-py"
        self.classes = 0
        self.cifardata = []
        self.trainData = []
        self.trainLabel = []
        self.testData = []
        self.testLabel = []

    def load_data(self):
        for file in os.listdir(self.datasetPath):
            if file != 'batches.meta' and file != 'readme.html':
                data = self.unpickle(self.datasetPath + "\\" + file)
                self.cifardata.append(data)
            if file == 'batches.meta':
                self.classes = self.unpickle(self.datasetPath + "\\" + file)
        train, test = self.splitData(self.cifardata)
        train = np.array(train)
        test = np.array(test)
        return (train[0], train[1]), (test[0], test[1])

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def splitData(self, cifardata):
        for index in cifardata:
            if b'training' in index[b'batch_label']:
                self.trainLabel.append(index[b'labels'])
                self.trainData.append(index[b'data'].tolist())
            else:
                self.testLabel.append(index[b'labels'])
                self.testData.append(index[b'data'].tolist())
        return (self.trainData, self.trainLabel), (self.testData, self.testLabel)


class ConvLayer:

    __slots__ = "LayerNumber", "numberOfFilters", "kerneldimension", "activation", "inputImageShape", "layername", "kernel", "classes"

    def __init__(self, numFilt=0, kerDim=(), act="", inImsize=(), layer=0):
        self.LayerNumber = layer
        self.numberOfFilters = numFilt
        self.kerneldimension = kerDim
        self.activation = act
        self.inputImageShape = inImsize
        self.layername = "ConvLayer"
        self.kernel = self.filterInit(self.numberOfFilters, self.kerneldimension)
        self.classes = Data.classes

    def __str__(self):
        return str(self.layername) + " [" + str(self.numberOfFilters) + ", " + str(self.kerneldimension) + ", " + \
               str(self.activation) + ", " + str(self.inputImageShape) + "]    "

    def filterInit(self, filters, dim):
        return np.random.normal(0, 0.35, (filters, dim[0], dim[1], dim[2]))


class Maxpool():

    __slots__ = "Layernumber", "rowfactor", "columnfactor", "stride", "layername", "dimensions"

    def __init__(self, factor, stride=2, layer=0, dim=()):
        self.Layernumber = layer
        self.rowfactor = factor[0]
        self.columnfactor = factor[1]
        self.stride = stride
        self.layername = "MaxPool"
        self.dimensions = dim

    def __str__(self):
        return str(self.layername) + " [" + "(" + str(self.rowfactor) + ", " + str(self.columnfactor) + \
               "), " + str(self.stride) + "]  "


class Dense:

    __slots__ = "Layernumber", "hiddenNodes", "activation", "layername", "weightMatrix", "BiasMatrix"

    # ishidden = False

    def __init__(self, layer, nodes, act, dimm):
        self.Layernumber = layer
        self.hiddenNodes = nodes
        self.activation = act
        self.layername = "Dense"
        self.weightMatrix, self.BiasMatrix = self.weightInit(nodes, dimm)

    def __str__(self):
        return str(self.layername) + " [" + str(self.hiddenNodes) + ", " + str(self.activation) + "]  "

    def weightInit(self, nodes, dim):
        inputsize = dim[0] * dim[1] * dim[2]
        return np.random.normal(0, 0.35, (inputsize, nodes)), np.random.normal(0, 0.35, (nodes))


class CNN:

    __slots__ = "numberOfFilters", "kerneldimension", "activation", "inputImageShape", "layernumber", "layers", \
                "allLayers", "Cmatrix", "numpar"

    dimensions = (0, 0, 0)
    convback = False
    num = 0

    def __init__(self):
        self.numberOfFilters = 0
        self.kerneldimension = (0, 0, 0)
        self.activation = ""
        self.inputImageShape = (0, 0, 0)
        self.layernumber = 0
        self.layers = {}
        self.allLayers = {}
        self.Cmatrix = np.zeros((10, 10))
        self.numpar = 0

    def addconv(self, filternumber, kernelSize, activation, inputShape=()):
        try:
            if self.layernumber == 0:
                layer = ConvLayer(filternumber, kernelSize, activation, inputShape, self.layernumber)
                self.layers[self.layernumber] = layer
                self.layernumber += 1
                CNN.dimensions = (inputShape[0], inputShape[1], filternumber)
            else:
                if inputShape is not ():
                    print("No need for input dimensions for next layer")
                else:
                    layer = ConvLayer(filternumber, kernelSize, activation, CNN.dimensions)
                    self.layers[self.layernumber] = layer
                    self.layernumber += 1
                    CNN.dimensions = (CNN.dimensions[0], CNN.dimensions[1], filternumber)
        except:
            print("Form:\n object.addconv(numberoffilers as int, size of tensor filter as tuple, "
                  "activationfunctionname as string, tensorimagesize as tuple)")

    def addpool(self, factor, stride):
        try:
            poolLayer = Maxpool(factor, stride, self.layernumber, CNN.dimensions)
            self.layers[self.layernumber] = poolLayer
            self.layernumber += 1
            CNN.dimensions = (CNN.dimensions[0]//factor[0], CNN.dimensions[1]//factor[1], CNN.dimensions[2])
        except:
            print("Form:\n object.addpool(factor as a tuple, stride as int)")

    def addDense(self, hiddenNodes, activation):
        try:
            denseLayer = Dense(self.layernumber, hiddenNodes, activation, CNN.dimensions)
            self.layers[self.layernumber] = denseLayer
            self.layernumber += 1
            CNN.dimensions = (hiddenNodes, 1, 1)
        except:
            print("Form:\n object.addDense(hiddenNodes as int, activation as string)")

    def summary(self):
        print(self)

    def __str__(self):
        dictionary = ""
        for key in self.layers:
            dictionary += "\n" + str(key) + ": " + str(self.layers[key])
            if self.layers[key].layername == "ConvLayer":
                numparam = np.prod(self.layers[key].kernel.shape)
                self.numpar += numparam
                dictionary += "---> Parameters to train: " + str(numparam)
            elif self.layers[key].layername == "MaxPool":
                numparam = 0
                dictionary += "\t\t\t\t\t\t\t---> Parameters to train: " + str(numparam)
            elif self.layers[key].layername == "Dense":
                if not self.layers[key].activation == "Softmax":
                    numparam = np.prod(self.layers[key].weightMatrix.shape) + np.prod(self.layers[key].BiasMatrix.shape)
                    self.numpar += numparam
                    dictionary += "\t\t\t\t\t\t\t\t---> Parameters to train: " + str(numparam)
                else:
                    numparam = 0
                    dictionary += "\t\t\t\t\t\t\t---> Parameters to train: " + str(numparam)
        CNN.num = int(self.numpar)
        return "~~~~~~~~~~~~~~~~~~~~~CNN Model~~~~~~~~~~~~~~~~~~~~" + "\nLayers: {   " + dictionary + "\n}\n" + \
               "Total number of trainable parameters: " + str(self.numpar)

    def convolve(self, image, instr):
        ans = 0
        features = []
        shape = instr.inputImageShape
        image = np.array(image).reshape((shape[2], shape[0], shape[1]))
        for i in range(instr.kernel.shape[0]):
            for j in range(instr.kernel.shape[1]):
                kernel2d = instr.kernel[i, j]
                ans = np.add(ans, np.convolve(image[j].flatten(), kernel2d.flatten(), 'same'))
            features.append(ans)
        return np.array(features)

    def softmaxinit(self, label):
        z = [0 for _ in range(10)]
        z[label-1] = 1
        return z

    def getWeights(self, layers):
        w = 0
        for l in layers:
            instr = layers[l]
            if instr.layername == 'ConvLayer':
                w += np.sum(np.square(instr.kernel))
            elif instr.layername == 'Dense':
                w += np.sum(np.square(instr.weightMatrix))
                w += np.sum(np.square(instr.BiasMatrix))
        return w

    def crossEntropy(self, prediction, target, reg, layers):
        wts = self.getWeights(layers)
        lambdaValue = ((reg / (2 * CNN.num)) * wts)
        return (- np.sum(np.multiply(target, np.log(prediction)))) + lambdaValue

    def fit(self, trainData, trainLabel, epoch=1, savelayer=False):
        previous = 0
        try:
            multiple = (CNN.num // 1000)
            inputImage = 0
            nanflag = False
            counter = 1
            # loss = []
            imloss = 100
            for e in range(epoch):
                learningRate = 3 / (10 ** (multiple + 1))                   #0.000001
                regularizer = 3 / (10 ** (multiple - 3))                    #0.03
                print(learningRate, regularizer)
                print("Epoch:", e+1)
                for i in range(trainData.shape[0]):
                    print(str(counter) + "0k")
                    counter += 1
                    previous = self.layers
                    for j in range(trainData.shape[1]):
                        CNN.convback = False
                        previous = self.layers
                        inputImage = (np.array(trainData[i, j])/255).tolist()
                        label = trainLabel[i, j]
                        softlabel = self.softmaxinit(label)
                        features = 0
                        for layer in self.layers:
                            instr = self.layers[layer]
                            if instr.layername == 'ConvLayer':
                                features = self.convolve(inputImage, instr)
                                inputImage = self.activate(features, instr.activation)
                                self.allLayers[layer] = inputImage
                            elif instr.layername == 'MaxPool':
                                shape = instr.dimensions
                                inputImage = self.pool(inputImage, instr.rowfactor, instr.columnfactor, instr.stride, shape)
                                self.allLayers[layer] = inputImage
                            elif instr.layername == 'Dense':
                                inputImage = self.dense(np.array(inputImage).flatten(), instr)
                                self.allLayers[layer] = inputImage
                        a = self.crossEntropy(inputImage, softlabel, regularizer, self.layers)
                        imloss = a
                        # print(counter)
                        # counter += 1
                        if np.isnan(a):
                            nanflag = True
                            self.layers = previous
                            print("Issue at loss!!!")
                            print(counter)
                            print("Nan encountered... Rolling back ....")
                            break
                        w, b = [], []
                        for layer in sorted(self.layers, reverse=True):
                            instr = self.layers[layer]
                            if instr.layername == 'Dense':
                                if instr.activation == "Softmax":
                                    error = np.subtract(np.array(inputImage), np.array(softlabel))
                                    w, b = self.gradientDescent(np.array(inputImage), error, instr, learningRate, regularizer)
                                    if True in np.isnan(instr.weightMatrix) or True in np.isnan(instr.BiasMatrix):
                                        nanflag = True
                                        self.layers = previous
                                        print("Issue at Dense Layer Softmax!!!!")
                                        print("Nan encountered... Rolling back ....")
                                        break
                                elif instr.activation == "ReLU":
                                    hiddenError = np.dot(np.subtract(np.array(inputImage), np.array(softlabel)), w.T)
                                    inputImage = self.allLayers[layer]
                                    w, b = self.gradientDescent(np.array(inputImage), hiddenError, instr, learningRate, regularizer)
                                    softlabel = hiddenError
                                    if True in np.isnan(instr.weightMatrix) or True in np.isnan(instr.BiasMatrix):
                                        nanflag = True
                                        self.layers = previous
                                        print("Issue at Dense Layer ReLU!!!!")
                                        print("Nan encountered... Rolling back ....")
                                        break
                            elif instr.layername == 'MaxPool':
                                if not CNN.convback:
                                    a = np.subtract(np.array(inputImage), np.array(softlabel))
                                    hiddenError = np.dot(a, w.T)
                                    softlabel = self.unpool(hiddenError, instr, layer)
                                else:
                                    hiddenError = softlabel
                                    softlabel = self.unpool(hiddenError, instr, layer)
                            elif instr.layername == 'ConvLayer':
                                if layer > 0:
                                    inputImage = self.allLayers[layer-1]
                                    softlabelnew = self.dxUpdate(softlabel, instr)
                                    reg = regularizer * instr.kernel
                                    instr.kernel = np.subtract(instr.kernel, np.multiply(learningRate, np.add(self.backpropconv(softlabel, instr, np.array(inputImage)), reg)))
                                    softlabel = softlabelnew
                                    w = instr.kernel
                                    CNN.convback = True
                                    if True in np.isnan(instr.kernel):
                                        nanflag = True
                                        self.layers = previous
                                        print("Issue at Convolution Layer!!!!")
                                        print("Nan encountered... Rolling back ....")
                                        break
                                else:
                                    instr.kernel = np.subtract(instr.kernel, np.multiply(learningRate, self.backpropconv(softlabel, instr, np.array(inputImage))))
                    learningRate /= 100
                    if nanflag:
                        break
                print(learningRate, regularizer)
                # epochloss = sum(imloss)/len(imloss)
                # loss.append(epochloss)
                print("Loss per epoch:", imloss)
            print("Loss:", imloss)
            if savelayer:
                model = open('model', 'wb')
                pickle.dump(self.layers, model)

            # validation
            counter = 1
            print("Validating.....")
            for i in range(trainData.shape[0]):
                print(str(counter) + "0k")
                counter += 1
                for j in range(trainData.shape[1]):
                    # print(counter)
                    # counter += 1
                    CNN.convback = False
                    inputImage = (np.array(trainData[i, j])/255).tolist()
                    label = trainLabel[i, j]
                    softlabel = self.softmaxinit(label)
                    features = 0
                    for layer in self.layers:
                        instr = self.layers[layer]
                        if instr.layername == 'ConvLayer':
                            features = self.convolve(inputImage, instr)
                            inputImage = self.activate(features, instr.activation)
                        elif instr.layername == 'MaxPool':
                            shape = instr.dimensions
                            inputImage = self.pool(inputImage, instr.rowfactor, instr.columnfactor, instr.stride, shape)
                        elif instr.layername == 'Dense':
                            inputImage = self.dense(np.array(inputImage).flatten(), instr)
                    pred = inputImage.index(max(inputImage))
                    targ = softlabel.index(max(softlabel))
                    self.Cmatrix[pred, targ] += 1
            return self.layers
        except KeyboardInterrupt:
            model = open('model', 'wb')
            pickle.dump(previous, model)

    def dxUpdate(self, dh, instr):
        t, r, c = instr.inputImageShape
        dx = np.zeros((c, t, r))
        w = instr.kernel
        temp = np.zeros((w.shape[0], w.shape[3], dh.shape[1], dh.shape[2]))
        for i in range(w.shape[0]):
            for j in range(dx.shape[0]):
                temp[i, j, :, :] = np.convolve(w[i, :, :, j].flatten(), dh[i, :, :].flatten(), 'same').reshape((dx.shape[1], dx.shape[2]))
        dx = np.sum(temp, axis=0)
        return dx

    def backpropconv(self, dh, instr, X):
        dw = np.zeros((dh.shape[0], instr.kerneldimension[0], instr.kerneldimension[1], instr.kerneldimension[2]))
        stride = dw.shape[1]
        for i in range(dw.shape[0]):
            for j in range(dw.shape[3]):
                temp = 0
                for k in range(dw.shape[1]):
                    for l in range(dw.shape[2]):
                        temp = np.add(temp, np.convolve(X[j, k:k+stride, l:l+stride].flatten(), dh[i, k:k+stride, l:l+stride].flatten(), 'same'))
                dw[i, :, :, j] = temp.reshape((dw.shape[1], dw.shape[2]))
        return dw

    def unpool(self, out, instr, layer):
        t, x, y = np.array(self.allLayers[layer]).shape
        out = out.reshape((t, x, y))
        newarr = np.zeros((instr.dimensions[2], instr.dimensions[0], instr.dimensions[1]))
        a = 0
        for i in range(newarr.shape[0]):
            b = 0
            for j in range(0, newarr.shape[1], instr.rowfactor):
                c = 0
                for k in range(0, newarr.shape[2], instr.columnfactor):
                    newarr[i, j, k] = out[a, b, c]
                    newarr[i, j + 1, k] = out[a, b, c]
                    newarr[i, j, k + 1] = out[a, b, c]
                    newarr[i, j + 1, k + 1] = out[a, b, c]
                    c += 1
                b += 1
            a += 1
        return newarr

    def gradientDescent(self, prediction, error, instr, alpha, lambdaValue):
        instr.weightMatrix = np.subtract(instr.weightMatrix, np.multiply(alpha, np.add(np.dot(prediction.T, error), instr.weightMatrix * lambdaValue)))
        instr.BiasMatrix = np.subtract(instr.BiasMatrix, np.multiply(alpha, np.sum(error)))
        return instr.weightMatrix, instr.BiasMatrix

    def dense(self, input, instr):
        return self.activate(np.add(np.dot(input, instr.weightMatrix), instr.BiasMatrix), instr.activation)

    def activate(self, features, activation):
        if self.ReLU.__name__ == activation:
            return self.ReLU(features.flatten()).tolist()
        elif self.Softmax.__name__ == activation:
            return self.Softmax(features).tolist()

    def ReLU(self, val):
        return np.maximum(0, val)

    def Softmax(self, val):
        return np.exp(val) / np.sum(np.exp(val))

    def pool(self, image, rfactor, cfactor, stride, shape):
        image = np.array(image).reshape((shape[2], shape[0], shape[1]))
        newimage = np.random.random((shape[0]//rfactor, shape[1]//cfactor))
        features = []
        for i in range(shape[2]):
            x = 0
            for j in range(0, shape[0], stride):
                y = 0
                for k in range(0, shape[1], stride):
                    z = image[i, j:j+rfactor, k:k+cfactor]
                    newimage[x, y] = np.max(z)
                    y += 1
                x += 1
            features.append(newimage)
        return features

    def history(self):
        print((np.trace(self.Cmatrix)/np.sum(self.Cmatrix))*100, "% accuracy")
        sns.heatmap(self.Cmatrix, annot=True, cbar=False)

    def predict(self, layers, testData, testLabel):
        cmatrix = np.zeros((10, 10))
        for i in range(testData.shape[0]):
            for j in range(testData.shape[1]):
                CNN.convback = False
                inputImage = (np.array(testData[i, j])/255).tolist()
                label = testLabel[i, j]
                softlabel = self.softmaxinit(label)
                features = 0
                for layer in layers:
                    instr = layers[layer]
                    if instr.layername == 'ConvLayer':
                        features = self.convolve(inputImage, instr)
                        inputImage = self.activate(features, instr.activation)
                    elif instr.layername == 'MaxPool':
                        shape = instr.dimensions
                        inputImage = self.pool(inputImage, instr.rowfactor, instr.columnfactor, instr.stride, shape)
                    elif instr.layername == 'Dense':
                        inputImage = self.dense(np.array(inputImage).flatten(), instr)
                pred = inputImage.index(max(inputImage))
                targ = softlabel.index(max(softlabel))
                cmatrix[pred, targ] += 1
        print((np.trace(self.Cmatrix) / np.sum(self.Cmatrix)) * 100, "% accuracy")
        sns.heatmap(self.Cmatrix, annot=True, cbar=False)
