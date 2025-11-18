r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

The loss function for each $(x,y)$ on each class $j\ne y$ is:
$$
\max⁡\{0,\Delta+w_j^T x - w_y^T x\}
$$
And it is above 0 if and only if:
$$
w_j^T x>w_y^T x-\Delta
$$

Meaning that we get panelty if a "bad" score is bigger than the "right" score (score of the correct class).
If $\Delta<0$, we get loss penalty in a less strict way, we might even not get penalty at all for a wrong class that have a bigger score than the correct one, because we only panelize a very "big" gap since $-\Delta>0$.



"""

part2_q2 = r"""
**Your answer:**

Similarly to the lecture, good weights are expected to represent in some way, the "average" class member's shape. Therefore, as we would expect, the weights as images appear to be similar to the handwritten numbers. Therefore, we can deduce that what the model learned is the weights in the positions of parts of the handwritten number. Thus, we can explain the classification errors to be mostly rare cases where some number occupies positions of another number due to different positions on paper or unique style of handwriting.

"""

part2_q3 = r"""
**Your answer:**

1.	Good. If the rate would be too big, we could see divergence in the graph, or at last much bigger volatility (Zigzags). If it was too small, we would see the opposite with very little change in the graph and possibly not getting good accuracy and it would seem as if we didn't have enough epochs – or like a stretched version of the beginning of our graph.
2.	Slightly overfitted. Since we got more accuracy on the training set, we conclude that the model learned how to be correct on the training data set more accurately than on the random unseen data we had in the test set.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
