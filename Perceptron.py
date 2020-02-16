import numpy as np
import random
from tqdm import tqdm
import os
import csv
import sys
class Perceptron:
    def __init__(self, percentData):
        self.numberList = []
        self.labelList = []
        self.numberValList = []
        self.valLabelList = []
        self.testNumberList = []
        self.testLabelList = []
        self.trainingSet = []
        self.weightList = [random.uniform(0, 1) for i in range(561)]
        self.learningRate = 1
        self.allWeightVectors = []
        self.extractedFeatures = []
        self.percentData = percentData
        self.allWeights = []
        self.randomIndicies = []

    def resetWeights(self):
        self.weightList = [random.uniform(0, 1) for i in range(21)]
        return

    def grabValNumbers(self):
        number = []
        filepath = 'validationimages.txt'
        with open(filepath) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                if (line.rstrip() == ''):
                    #print("empty thing works")
                    if(len(number) > 5):
                        self.numberValList.append(number)
                        #print("Line {}: {}".format(cnt, lastLine))
                    number = []
                    line = fp.readline()
                    continue
                else:
                    numLine = line.rstrip()
                    while(len(numLine) < 28):
                        numLine += ' '
                    number.append(numLine)
                    #print("Line {}: {}".format(cnt, line.rstrip()))
                    #lastLine = numLine
                    line = fp.readline()
                    cnt += 1
            if(len(number) > 0):
                self.numberValList.append(number)
            number = []

    def grabTestNumbers(self):
        number = []
        filepath = 'testimages.txt'
        with open(filepath) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                if (line.rstrip() == ''):
                    #print("empty thing works")
                    if(len(number) > 5):
                        self.testNumberList.append(number)
                        #print("Line {}: {}".format(cnt, lastLine))
                    number = []
                    line = fp.readline()
                    continue
                else:
                    numLine = line.rstrip()
                    while(len(numLine) < 28):
                        numLine += ' '
                    number.append(numLine)
                    #print("Line {}: {}".format(cnt, line.rstrip()))
                    #lastLine = numLine
                    line = fp.readline()
                    cnt += 1
            if(len(number) > 0):
                self.testNumberList.append(number)
            number = []

    def grabTestLabels(self):
        filepath = 'testlabels.txt'
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                numLabel = line.rstrip()
                self.testLabelList.append(numLabel)
                line = fp.readline()


    def grabNumbers(self):
        number = []
        filepath = 'trainingimages.txt'
        with open(filepath) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                if (line.rstrip() == ''):
                    #print("empty thing works")
                    if(len(number) > 5):
                        self.numberList.append(number)
                        #print("Line {}: {}".format(cnt, lastLine))
                    number = []
                    line = fp.readline()
                    continue
                else:
                    numLine = line.rstrip()
                    while(len(numLine) != 28):
                        numLine += ' '
                    number.append(numLine)
                    #print("Line {}: {}".format(cnt, line.rstrip()))
                    #lastLine = numLine
                    line = fp.readline()
                    cnt += 1
            if(len(number) > 0):
                self.numberList.append(number)
            number = []



    def grabLabels(self):
        filepath = 'traininglabels.txt'
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                numLabel = line.rstrip()
                self.labelList.append(numLabel)
                line = fp.readline()

    def grabValLabels(self):
        filepath = 'validationlabels.txt'
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                numLabel = line.rstrip()
                self.valLabelList.append(numLabel)
                line = fp.readline()

    def countLeft(self, line):
        count = 0
        for c in line:
            if(c == '+' or c == '#'):
                break
            else:
                count += 1
        return count

    def countRight(self, line):
        i = 0
        count = 0
        lastCharIndex = 0
        while(i < len(line)):
            if(line[i] == '+' or line[i] == '#'):
                #print("char is " + str(line[i]))
                #print(str(lastCharIndex))
                lastCharIndex = i
            i += 1
        j = lastCharIndex + 1
        while(j <= len(line)):
            #print("length of line " + str(len(line)))
            count += 1
            j += 1
        if(count > 0):
            count -= 1
        return count

    #def findWeirdNumbers(self):

    def getFeatureVector(self, number):
        featureVector = []
        bias = 1
        featureVector.append(bias)
        #prevFeature = 0
        for j in range(len(number)):
            #if(len(number) != 20):
                #print(len(number))
            numberLine = number[j]
            while (len(numberLine) < 28):
                numberLine += ' '
            featureLine = []
            spaceNum = 0
            plusNum = 0
            poundNum = 0
            for c in numberLine:
                #featureLine = []
                if(c == ' '):
                    spaceNum += 1
                if(c == '+'):
                    plusNum += 1
                if(c == '#'):
                    poundNum += 1
            featureLine = [spaceNum, plusNum, poundNum]
            leftSpace = self.countLeft(numberLine)
            rightSpace = self.countRight(numberLine)
            holeSpace = spaceNum - leftSpace - rightSpace
            #print("holeSpace is: " + str(holeSpace))
            if(spaceNum >= 0):
                feature = ((1.2*(plusNum) + 1.2*(poundNum) + holeSpace )/(spaceNum + plusNum + poundNum))
                #feature =
                #feature = (((plusNum) + (poundNum) + holeSpace))#/(spaceNum + plusNum + poundNum))
                #prevFeature = feature
                #feature = (spaceNum - (plusNum + poundNum))
            else:
                #print('zero spaces')
                feature = 0
                #prevFeature = feature
            featureVector.append(feature)
        while(len(featureVector) < 21):
            featureVector.append(0)
        #print(featureVector)
        return featureVector

    def getFeatureVectorTwo(self, number):
        featureVector = []
        bias = 1
        featureVector.append(bias)

        for i in range(len(number)):
            for j in range(len(number[i])):
                c = number[i][j]
                if(c == '+' or c == '#'):
                    feature = 1
                else:
                    feature = 0
                featureVector.append(feature)
        while(len(featureVector) < 561):
            featureVector.append(0)

        return featureVector

    def getWeightVector(self, featureVector):
        weightVector = [0 for i in len(featureVector)]
        self.weightList.append(weightVector)

    def makeTrainingSet(self):
        self.grabNumbers()
        self.grabLabels()
        self.grabValNumbers()
        self.grabValLabels()
        self.grabTestNumbers()
        self.grabTestLabels()
        lenNumberList = len(self.numberList)
        totalData = lenNumberList*self.percentData
        dataPoints = int(totalData)
        self.randomIndicies = random.sample(range(lenNumberList), dataPoints)



    def weightChange(self, yReal, yPred, featureVec):
        for i in range(len(self.weightList)):
            changeWeight = self.learningRate*(yReal - yPred)*featureVec[i]
            newWeight = self.weightList[i] + changeWeight
            self.weightList[i] = newWeight
            #print(self.weightList)
        return


    def computeFunction(self, featureVector, weightVector):
        featureVec = np.array(featureVector)
        weightVec = np.array(weightVector)
        funcValue = np.dot(featureVec, weightVec)
        return funcValue

    def trainData(self):
        winStreak = 0
        count = 0
        n = 0
        lenNumberList = len(self.numberList)
        for j in tqdm(range(10)):
            #count = 0
            #print("j in for loop: " + str(j))
            #while ((count < 30)):
            for count in tqdm(range(1)):
                #print("count in loop: " + str(count))
            #print("in loop")
                for i in tqdm(self.randomIndicies):
                    #print("i in loop: " + str(i))
                    #print("label is " + str(self.labelList[i]))
                    #print("j is " + str(j))
                    featureVec = self.getFeatureVectorTwo(self.numberList[i])
                    funcValue = self.computeFunction(featureVec, self.weightList)
                    if(funcValue >= 0):
                        yPred = True
                    else:
                        yPred = False
                    if(self.labelList[i] == str(j)):
                        yReal = True
                        #print("label is " + str(self.labelList[i]))
                        #print("j is " + str(j))
                    else:
                        yReal = False
                    if((funcValue >= 0 and yReal == True) or (funcValue < 0 and yReal == False)):
                        winStreak += 1
                    #count += 1
                        n += 1
                    #print("it skipped " + str(n))
                        continue
                    else:
                        self.weightChange(yReal, yPred, featureVec)
                        winStreak = 0
                    #count += 1
                #self.learningRate = self.learningRate - self.learningRate*.1
                #print("updating weight")
                #count += 1
            self.allWeights.append(self.weightList)
            self.weightList = [random.uniform(0, 1) for i in range(561)]
        #print(winStreak)
        #print(count)
        return

    def makeGuess(self, number):
        boolGuesses = []
        featureVec = self.getFeatureVectorTwo(number)
        #for i in range(len(featureVec)):
            #print(featureVec[i])
        max = float('-inf')
        numGuess = None
        for i in range(10):#for each weightList
            funcValue = self.computeFunction(featureVec, self.allWeights[i])
            if(funcValue > max):
                max = funcValue
                numGuess = i
            if(funcValue >= 0):
                yPred = True
            else:
                yPred = False
            print("------ " + str(i) + " ------")
            print("Perceptron classifies this as " + str(yPred))
            print(funcValue)
            boolGuesses.append(yPred)
        boolGuesses.append(numGuess)
        return boolGuesses

    def collectTestAccuracy(self):
        self.makeTrainingSet()
        self.trainData()
        lenNumberList = len(self.testNumberList)
        correct = 0
        for i in range(lenNumberList):
            guessVec = self.makeGuess(self.testNumberList[i])
            if(str(guessVec[-1]) == str(self.testLabelList[i])):
                correct += 1
        acc = correct/(lenNumberList)*100
        return acc


def main():

    percent = input("What percentage of the data would you like to use (in decimal e.g. 0.55):  ")
    percentData = float(percent)

    p = Perceptron(percentData)
    p.makeTrainingSet()
    numberList = p.testNumberList
    labelList = p.testLabelList
    #i = len(numberList) - 1
    lenNumberList = len(numberList)
    #print("There are " + str(len(labelList)) + " in labelList")
    #print()
    p.trainData()
    correct = 0
    #truePositive = 0
    #trueNegative = 0
    #falsePositive = 0
    #falseNegative = 0
    for i in range(lenNumberList):
        #if(p.makeGuess(numberList[i]) == True or p.makeGuess(numberList[i]) == False):
        guessVec = p.makeGuess(numberList[i])
        #print("Perceptron classifies this as: " + str(guessVec[-1]))
        #for k in range((len(guessVec) - 1)):
            #if(guessVec[k] == True and str(k) == labelList[i]):
            #    truePositive += 1
            #if(guessVec[k] == True and str(k) != labelList[i]):
            #    falsePositive += 1
            #if(guessVec[k] == False and str(k) != labelList[i]):
            #    trueNegative += 1
            #if(guessVec[k] == False and str(k) == labelList[i]):
            #    falseNegative += 1
        if(str(guessVec[-1]) == str(labelList[i])):
            correct += 1
            print("Correct")
        else:
            print("Incorrect")
        for j in range(len(numberList[i])): #For printing the number
            #print(len(numberList[i][j]))
            print(numberList[i][j])
        print("Label is " + str(labelList[i]))
        print("There are " + str(j) + " lines in the number")
    print("Total correct: " + str(correct))
    print("Total numbers: " + str(lenNumberList))
    acc = (correct/lenNumberList)*100
    print("Correct/Total Numbers: "+ str(acc) + '%')
    #realAcc = ((truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative))*100
    #print("Real Accuracy: " + str(realAcc) + '%')
    """
    if not os.path.exists('Perceptron/Digits'):
            os.makedirs('Perceptron/Digits')
            with open('Perceptron/Digits/digitAccuracy.csv', 'w') as f:
                f.write('Data Percent, Accuracy\n')

    with open ('Perceptron/Digits/digitAccuracy.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([p.percentData, acc])

    #str = " +    #       "
    #print(str)
    #print(p.countLeft(str))
    #print(p.countRight(str))
    meanList = []
    sdList = []
    for percent in tqdm(range(1, 11)):
        allAcc = []
        totalSum = 0
        mean = 0
        sD = 0
        for trial in tqdm(range(1)):
        #print("Percentage Used: " +  str(percentData))
            percentData = 0.1 * percent
            p = Perceptron(percentData)
            acc = p.collectTestAccuracy()
            #print(acc)
            acc = round(acc, 1)
            allAcc.append(acc)
        totalSum = sum(allAcc)
        mean = totalSum/len(allAcc)
        mean = round(mean, 2)
        accArray = np.array(allAcc)
        sD = np.std(accArray)
        meanList.append(mean)
        sdList.append(sD)

    print()
    print()
    print()
    print()
    percent = 10
    for i in range(len(meanList)):
        print("Mean for " + str(percent) + '%: ' + str(meanList[i]) )
        print("Standard Deviation for " + str(percent) + '%: ' + str(sdList[i]) )
        print()
        percent += 10
    """

if __name__ == "__main__":
    main()
