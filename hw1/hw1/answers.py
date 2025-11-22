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

The ideal case is that all residuals are as close as possible to 0, maybe even all sits on the 0 line, if possible (not probable, since the data is assumed to be normally distributed and therefore there should be gaussian noise in real cases), meaning that the predictions are exact. Based on the residuals plot, it looks like the fitting is not so good. For houses worth around 10-30k $, we have a lot of deviations of about 5k $, which is not so small. However, there is a significant amount of data points that have very small residuals, and in any case, it looks like the expected residual is 0, which means that the model is correct in expectation. The residuals plot got improved (in the sense of smaller variance of the residuals) by implementing non linearities, and then even more by finding the bets hyperparameters using CV.

"""

part3_q2 = r"""
**Your answer:**
1. Yes it is still a linear regression model. The non linearities are being added in a pre processing step, and after that it can be considered as a linear model again.
2. In principle, in a compact domain (for instance if y is bounded), the universal approximation theorem applies, and therefore yes, any function can be approximated with ReLU's foor example, and non-linearities. In any case, usually we can treat the y values as bounded because we know that for example a 700k $ house is not possible for most cases.
3. Like we saw in lecture for using ReLU to "bend" the linear hyper-plane, adding non linearities can "curve" and "bend" and "twist" the hyper-plane and change the shape of it, thus allowing better data separation. Afte that it would not be a hyper-plane, since linearity for elements in it won’t exist anymore.
Write your answer using **markdown** and $\LaTeX$:
```python
"""

part3_q3 = r"""
**Your answer:**

1.

The expectation is:
\[
\mathbb{E}[|y - x|], \qquad x, y \sim \mathrm{Uniform}(0,1).
\]

The joint density of \(x\) and \(y\) on \([0,1]^2\) is 1, so we have:
\[
\mathbb{E}[|y - x|]
= \int_0^1 \int_0^1 |y - x| \, dy \, dx.
\]

The regime is symmetrical, so we can compute only the region where \(y \ge x\) and multiply by 2:
\[
\mathbb{E}[|y - x|]
= 2 \int_0^1 \int_x^1 (y - x)\, dy \, dx.
\]

The inner integral:
\[
\int_x^1 (y - x)\, dy
= \left[ \tfrac{1}{2}(y - x)^2 \right]_{y=x}^{1}
= \tfrac{1}{2}(1 - x)^2.
\]

\[
\mathbb{E}[|y - x|]
= 2 \int_0^1 \tfrac{1}{2} (1 - x)^2 \, dx
= \int_0^1 (1 - x)^2 \, dx.
\]

Then:
\[
\int_0^1 (1 - x)^2 dx
= \left[ 1 - x + \tfrac{x^2}{3} \right]_{0}^{1}
= \frac{1}{3}.
\]

\[
\boxed{\mathbb{E}[|y - x|] = \frac{1}{3}}
\]


2.
The expectation is:
\[
\mathbb{E}_x\!\left[\,|\hat{x} - x|\,\right], \qquad x \sim \mathrm{Uniform}(0,1),
\]
where \(\hat{x}\) is fixed in \([0,1]\).

Since \(x\) is uniform on \([0,1]\), the expectation is:
\[
\mathbb{E}_x[|\hat{x} - x|]
= \int_0^1 |\hat{x} - x|\,dx.
\]

We split at \(x = \hat{x}\):
\[
\int_0^1 |\hat{x} - x|\,dx
= \int_0^{\hat{x}} (\hat{x} - x)\,dx 
  + \int_{\hat{x}}^{1} (x - \hat{x})\,dx.
\]

The first part:
\[
\int_0^{\hat{x}} (\hat{x} - x)\,dx
= \left[\hat{x}x - \frac{x^2}{2}\right]_{0}^{\hat{x}}
= \frac{\hat{x}^2}{2}.
\]

The second part:
\[
\int_{\hat{x}}^{1} (x - \hat{x})\,dx
= \left[\frac{x^2}{2} - \hat{x}x\right]_{\hat{x}}^{1}
= \left(\frac{1}{2} - \hat{x}\right) 
  - \left(\frac{\hat{x}^2}{2} - \hat{x}^2\right)
= \frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}.
\]

Summing both gives:
\[
\mathbb{E}_x[|\hat{x} - x|]
= \frac{\hat{x}^2}{2}
  + \left(\frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}\right)
= \hat{x}^2 - \hat{x} + \frac{1}{2}.
\]

\[
\boxed{\mathbb{E}_x[|\hat{x} - x|] = \hat{x}^2 - \hat{x} + \frac{1}{2}}
\]

"""

# ==============

# ==============
