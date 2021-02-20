# Image-Orientation-Classification-Using-decision-tree

### Implementation:
* Training phase:
For training phase, the entire train data is used to train the model and the decisions
taken by the model are stored in DecisionClassifier.txt.
For each feature in the training set, we calculate the information gain. We use the mid
value of 255 i.e. 128 to classify if the data will go to left or right branch. Once split, we
calculate the entropies of each branch and the weighted average entropy. We then
decide the feature to split data on by selecting the feature with the least entropy (max
homogeneous classifications). We then evaluate the entropies of the left and right
branches. If the entropy of a branch is less than a threshold, then we take the decision
to classify. Classification decisions made are stored in a dictionary with tree node
number as the key and the feature index as the value. If the entropy is greater than
threshold, then we call the same function recursively till max depth is reached and then
make a decision.
The decisions are stored in a dictionary and later stored in DecisionClassifier.txt file once
training phase is complete.
* Testing phase:
1. The test data and the decisions dictionary from the training phase are loaded.
2. For each feature in training set, check if some decision was made in training phase
with respect to this feature. If yes then make the same decision as made in training
phase, else compare value of data with 128 and decide which feature to consider
next.
3. For every image, go on appending the predicted classification to a dictionary.

### Results:
Our algorithm was optimal at depth 9 with accuracy of 61.93%

Please refer the report attached to view the results and experimental evaluation




