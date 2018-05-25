@Author: Arthur Shing
@Date:   2018-04-16T21:06:11-07:00
@Filename: README.txt
@Last modified by:   Arthur Shing
@Last modified time: 2018-05-24T22:39:08-07:00

There are two source codes, one per problem:
prob1.py
prob2.py

To run the code, you need to have matplotlib and numpy installed:
pip install --user matplotlib
pip install --user numpy

To compile and run the code:
python prob1.py
python prob2.py

prob1.py will output the shape of the random centers as (k, 784) for each k, as well as the SSE for each iteration.
Part 1 graph is saved as sse_part1.png,
Part 2 graph is saved as sse_part2.png.

prob2.py will output:
the shape of the data,
the top 10 eigenvalues,
image of the mean, as eig15.png,
images for the top 10 eigenvectors, as eig0.png through eig9.png,
images with the largest value in the 10 dimensions by the eigenvectors, as eig100.png through eig109.png
