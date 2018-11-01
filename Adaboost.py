#
import numpy as np 

def loadSimpleData():
	datMat = np.matrix([[1.,2.1], [2.,1.1], [1.3,1.], [1.,1.], [2.,1.]])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0]

	return datMat,classLabels

#decision stump-generating function
def stumpClassify(dataMatrix,dimen,threshVal,thresIneq):
	retArray = np.ones((np.shape(dataMatrix)[0],1)) #initialized with matrix 1 -> class 1
	if thresIneq == 'lt':#less than
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #class-1
	else: #greater than
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0 #class-1
	return retArray
#weak learner
def buildStump(dataArr,classLabels,D):
	dataMatrix = np.mat(dataArr)
	labelMat = np.mat(classLabels).T 
	m,n = np.shape(dataMatrix)
	numSteps = 10.0
	bestStump = {}  #empty dictionary to store the classifier information corresponding to 
					#the best choice of a decision stump given this weight vector D
	bestClasEst = np.mat(np.zeros((m,1)))
	minError = np.inf #initialized to positive infinity, to iterate over the possible values of the features
	for i in range(n): #go over all the features in the dataset
		rangeMin = dataMatrix[:,i].min()
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax - rangeMin) / numSteps#due to numeric values, use (max-min) to see how large the step size is

		for j in range(-1,int(numSteps) + 1):
			for inequal in ['lt','gt']: #< or >
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) #return the class pridiction
				errArr = np.mat(np.ones((m,1)))#initialization with matrix 1
				errArr[predictedVals == labelMat] = 0 #if the predictedVals != actual class in labelMat, the value=1 
				weightedError = D.T * errArr #calculate the weighted error
				#print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %(i, threshVal,inequal,weightedError))
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()#save the error below minError
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump, minError, bestClasEst

#Adaboost training with decision stumps
def adaBoostTrainDS(dataArr,classLabels,numIt = 40):#numIt: number of iterations
	weakClassArr = []#to store the output array of decision stump
	m = np.shape(dataArr)[0] #the number of data points in dataset
	print("m:",m)
	D = np.mat(np.ones((m,1)) / m)#weight vector,initialized with matrix 1/m
	print("D:",D)
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArr,classLabels,D)#build decision stump
		print("D(T):",D.T)
		alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))#1e-16,to avoid divisor is 0 when no error
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		print("classEst(T):",classEst.T)
		#calculate D for next iteration		
		expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
		D = np.multiply(D,np.exp(expon))
		D = D / D.sum()
		aggClassEst += alpha * classEst
		print("aggClassEst(T):",aggClassEst.T)
		#calculate aggregate error
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
		errorRate = aggErrors.sum() / m
		print("total error: %f"%errorRate,"\n")
		if errorRate == 0.0:
			break
	return weakClassArr,aggClassEst

#Adaboost classification function
#datToClass is input, classifierArr is an array of weak classifiers
def adaClassify(datToClass, classifierArr):
	dataMatrix = np.mat(datToClass)#convert datToClass to Numpy matrix
	m = np.shape(dataMatrix)[0]#number of instances in datToClass
	aggClassEst = np.mat(np.zeros((m,1)))#column vector of all 0s
	for i in range(len(classifierArr)):#look over all of the weak classifiers in classifierArr
		#get a class estimate from stumpClassify() for each of weak classifiers in classifierArr
		classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst
		print(aggClassEst)
	return np.sign(aggClassEst)

#Adaptive load data function
def loadDataSet(filename):
	numFeat = len(open(filename).readline().split('\t'))
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat - 1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat

#ROC plotting and AUC calculating function
#predStrengths: the strength of the classifier's predictions
def plotROC(predStrengths, classLabels):
	import matplotlib.pyplot as plt
	cur = (1.0,1.0) #cursor position
	ySum = 0.0 #for calculating AUC
	numPosClas = sum(np.array(classLabels) == 1.0)#the number of positive instances
	yStep = 1 / float(numPosClas)
	xStep = 1 / float(len(classLabels) - numPosClas)
	sortedIndices = predStrengths.argsort()#sorted index in ascending order
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sortedIndices.tolist()[0]:#array to list
		if classLabels[index] == 1.0:#true positive rate
			delX = 0
			delY = yStep
		else:#false positive rate
			delX = xStep
			delY = 0
			ySum += cur[1]
		ax.plot([cur[0], cur[0] - delX],[cur[1], cur[1] - delY], c = 'b')
		cur = (cur[0] - delX, cur[1] - delY)
	ax.plot([0,1], [0,1], 'b--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for Adaboost Horse Colic Detection System')
	ax.axis([0,1,0,1])
	plt.show()
	print("the Area Under the Curve is ",ySum * xStep)#area