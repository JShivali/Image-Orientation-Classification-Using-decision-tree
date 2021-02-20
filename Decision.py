import sys
import pandas as pd
import numpy as np
from collections import Counter
import math
import pickle

# Configurable global variables
maxDepth=9
decision={}
tree={}
threshEntro=0.4
labels=[0,90,180,270]

# function for calculating entropy
def calcEntropy(indexes,label):
    entropy = 0
    counts={0:0,90:0,180:0,270:0}
    label = label[indexes]
    label_length = len(label)
    if (label_length == 0):
        return np.inf

    for each in label:
        counts[each]+=1
    for i in counts.keys():
        ratio = counts[i] / label_length
        if (ratio != 0):
            entropy += (-ratio) * np.log2(ratio)
    return entropy


def calcInfo(train,f,label):
    indexL=[]
    indexR=[]
    for i in range(np.shape(train)[0]):
        if(train[i][f]<128):
             indexL.append(i)
        else:
            indexR.append(i)
    entroL=calcEntropy(indexL,label)
    entroR=calcEntropy(indexR,label)

    avgEntro=(((len(indexL)/len(train))*entroL)+((len(indexR)/len(train))*entroR))
    return avgEntro,entroL,entroR,indexL,indexR

# Function for decision tree
def treesplit(train, label, node, dep):
    indexesL = []
    indexesR = []
    min = math.inf
    featToUse = -1
    entropyR = 0
    entropyL = 0


    if (dep== maxDepth):
        decision[np.floor(node / 2)]=Counter(label).most_common(1)[0][0]
        return

    for f in range(np.shape(train)[1]):
        avgEntropy,entroL,entroR,indexL,indexR=calcInfo(train,f,label)
        if(avgEntropy<min):
            indexesL=indexL
            indexesR=indexR
            entropyL=entroL
            entropyR=entroR
            featToUse=f
            min=avgEntropy

    if(featToUse==-1):
        return

    if(len(indexesL)!=0):
         if(entropyL<threshEntro):
            # take decision
            decision[np.floor(node / 2)]=label[indexesL[0]]
         else:
             treesplit(train[indexesL], label[indexesL], node * 2, dep + 1)

    if (len(indexesR) != 0):
        if (entropyR < threshEntro):
            # take decision
            decision[np.floor(node / 2)] = label[indexesR[0]]
        else:
            treesplit(train[indexesR], label[indexesR], node * 2 + 1, dep + 1)
    tree[node] = featToUse


#Classification function for test data
def classifyTestData(testData):
    prediction=[]
    file = open('tree_model.txt', 'rb')
    decision = pickle.load(file)
    tree = pickle.load(file)
    for i in range(len(testData)):
        feature = 1
        for p in range(maxDepth):       # for iterating through the tree till max depth
            if (feature in decision):        # Check decisions made in training if a decision for this feature has been made.
                 prediction.append(decision[feature])
                 break
            featureNo= tree[feature]
            if (testData[i][featureNo] < 128):
                feature = feature * 2
            else:
                feature = feature * 2 + 1
    return prediction

#Function to check accuracy and output to file
def accOutput(output,images,test_data_label):
    output_file = open("output_tree.txt", 'w')
    count=0
    for i in range(len(output)):
        output_file.write(images[i]+" "+str(output[i])+"\n")
        if(output[i] == test_data_label[i]):
            count += 1
    acc=count/len(output)
    return acc


if sys.argv[1] == "train":
        train_data = pd.read_table(sys.argv[2], delim_whitespace=True, header=None)
        train_data_label = train_data[1].values
        train_data.drop([0, 1], axis=1, inplace=True)
        train_data = train_data.values
        depth=0
        colToSplit=1
        treesplit(train_data,train_data_label,colToSplit,depth)

        # Pickle the data into files
        file = open('tree_model.txt', 'wb')
        pickle.dump(decision, file)
        pickle.dump(tree, file)
        file.close()


if sys.argv[1] == "test":
        test_data = pd.read_table(sys.argv[2], delim_whitespace=True, header=None)
        test_data_images=test_data[0].values
        test_data_label = test_data[1].values
        test_data.drop([0, 1], axis=1, inplace=True)
        test_data = test_data.values

        pre=classifyTestData(test_data)
        accuracy = accOutput(pre,test_data_images,test_data_label) *100
        print(accuracy)
