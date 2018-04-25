@Author: Arthur Shing
@Date:   2018-04-16T21:06:11-07:00
@Filename: README.txt
@Last modified by:   Arthur Shing
@Last modified time: 2018-04-18T18:12:15-07:00

There are two source codes, one per problem:
prob1.py
prob2.py

To run the code, you need to have matplotlib and numpy installed:
pip install --user matplotlib
pip install --user numpy

To compile and run the code:
python prob1.py
python prob2.py

prob1.py will output the weight vector, and the training/testing ASEs with/without the dummy variable.
It will also output the weight and training/testing ASEs for when random features are added.

prob2.py will output the iteration (epoch), the lambda (1 for default), the learn rate, the SSE, the accuracy when ran on the training dataset, and the accuracy when ran on the testing dataset, each iteration.

Graphs will be saved as:
acctrain.png (For 2.1, batch gradient descent)
acctrain2.png (For 2.3, L2 Regularization)
randfeatures.png (For 1.4, adding random features)
