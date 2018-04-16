1c1
< # @Author: Arthur Shing & Monica Sek
---
> # @Author: Arthur Shing
5c5
< # @Last modified time: 2018-04-16T03:53:09-07:00
---
> # @Last modified time: 2018-04-16T03:22:59-07:00
23a24,42
> def predict(x, w):
>     # for row in x:
>     #     for i in row:
>     #         continue
>     return
>     # w is just zeros, so the first iteration should return 0.5
> 
> # def predict1(row, w):
> #     yhat = 0
> #     for i in range(len(row)-1):
> #         yhat += w[i] * row[i]
> #     return 1.0 / (1.0 + np.exp(-yhat))
> #
> def checkAccuracy(x):
>     if x > 0.5 or x < -0.5:
>         return 0
>     else:
>         return 1
> 
28a48,71
> # Online gradient descent
> def train(data, learnRate, epoch, zeros, label):
>     w = zeros
>     acc = np.zeros((epoch,), dtype=float)
>     for e in range(epoch):
>         sumError = 0
>         for row in range(len(data)):
>             y = sigmoidFunct(w, data[row].reshape(256,1))
>             error = label[row] - y
>             sumError += error**2
>             odds = y * (1-y)
>             learnError = error * learnRate * odds
> 
>             w = np.add(w, np.multiply(learnError, data[row].reshape(256,1))) # this took me 5ever
>             # why does the above work so much better than the one below?
>             # for pixel in range(len(data[row])):
>             #     w[pixel] = w[pixel] + (learnRate*error*odds*data[row][pixel])
> 
>         acc[e] = test(data, label, w)
>         # print acc[e]
>         print('epoch: %d, learn rate: %.6f, SSE: %.6f, accuracy: %.6f' % (e, learnRate, sumError, acc[e]))
>     return (w, acc)
> 
> 
30,38c73
< # data = training dataset
< # epoch = number of iterations
< # zeros should be an array of zeros. Unnecessary and I should fix it later.
< # label = array of 'is training data 4 or 9?' (0 is 4, 1 is 9)
< # xTest and lTest is the data and label for the test file, assuming
< ''' 
< this is a test
< '''
< def trainBatch(data, learnRate, epoch, zeros, label, xTest, lTest):
---
> def trainBatch(data, learnRate, epoch, zeros, label):
41d75
<     accTest = np.zeros((epoch,), dtype=float)
55d88
<         accTest[e] = test(xTest, lTest, w)
57,58c90,94
<         print('epoch: %d, learn rate: %.6f, SSE: %.6f, accuracy: %.6f, accuracyTest: %.6f' % (e, learnRate, sumError, acc[e], accTest[e]))
<     return (w, acc, accTest)
---
>         # print acc[e]
>         print('epoch: %d, learn rate: %.6f, SSE: %.6f, accuracy: %.6f' % (e, learnRate, sumError, acc[e]))
> 
>     return (w, acc)
> 
60,61d95
< # Tests multiple images (x is the test file data) over their actual numbers (l)
< # Returns percent correct in decimal form (0 to 1)
74a109,111
> 
> 
> 
77d113
<     problem1()
79d114
< def problem1():
82,91c117,132
< 
<     # Learning Rate
<     learn = 0.000001
<     # Iterations
<     epoch = 65
< 
<     # Accuracy = array of % correctly predicted in each iteration
<     (coefB, accuracyB, accuracyTestB) = trainBatch(x, learn, epoch, w, l, xTest, lTest)
< 
<     # TODO:: Add legend and labels
---
>     # for row in data[780:790]:
>     #     yhat = predict1(row, w[0:])
>     #     what = sigmoidFunct(w[0:], row[:-1].reshape(256,1))
>     #     print("Expected=%.3f, Predicted=%.3f [%d] || %.3f [%d]" % (row[-1], yhat, round(yhat), what, round(what)))
>     learn = 0.00005
>     # learn = 0.000001
>     epoch = 80
>     (coef, accuracy) = train(x, learn, epoch, w, l)
>     (coefB, accuracyB) = trainBatch(x, 0.000001, epoch, w, l)
>     print test(xTest, lTest, coef)
>     print test(xTest, lTest, coefB)
> 
>     # TODO
>     # print coef
>     # print accuracy.shape
>     plot(range(epoch), accuracy, "acctrain.png")
93,94d133
<     plot(range(epoch), accuracyTestB, "acctrain.png")
<     return
96c135,141
< # Plots into a png file
---
>     # plot(test, "test.png")
>     # Equation: prediction g(z)
>         # g(z) = 1 / (1 + e^-(z)) where z is linear model a + bx + cx etc
>         # as linear model approaches -infinity, prediction = 0
>         # as linear model approaches 0, prediction = 0.5
>         # as linear model approaches infinity, prediction = 1
> 
