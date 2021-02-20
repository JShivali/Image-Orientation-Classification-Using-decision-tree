import numpy as np
import pandas as pd
import sys

if __name__ == "__main__":
    if sys.argv[1] == "train":
        
        if sys.argv[4] == "nearest":
            import knn
        elif sys.argv[4] == "tree":
            import Decision
        elif sys.argv[4] == "nnet":
            import NeuralNetwork
            nnet = NeuralNetwork.Neural_Network()
            nnet.train(sys.argv[2], sys.argv[3])

    if sys.argv[1] == "test":

        if sys.argv[4] == "nearest":
            import knn
        elif sys.argv[4] == "tree":
            import Decision
        elif sys.argv[4] == "nnet":
            import NeuralNetwork
            nnet = NeuralNetwork.Neural_Network()
            nnet.test(sys.argv[2], sys.argv[3])
            
