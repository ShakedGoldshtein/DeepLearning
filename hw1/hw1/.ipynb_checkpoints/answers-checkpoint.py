r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1.	False. We would want to split the data as uniformly as possible to represent the real data distribution as closely as possible. There might be marginal distribution that will make the classifier be correct perfectly on the training set, while being very wrong on the test set. Plus We would want a somewhat large training and testing sets, since small datasets cannot usually represent enough of the data's distribution for training and testing.

2.	False. The test set is to be used after training the model, to test its accuracy of prediction. In cross-validation we want to optimize hyperparameters so that the training will be as efficient and accurate as possible. Using the test set in the cross-validation step is like cheating, as we let the model fits itself best to the test set, thus making the generalization error estimation on the test set not credible.

3. True. Cross-validation works by splitting the dataset into $k$ folds. Then, for $k$ times we train the model on $k-1$ folds and evaluate the model's performance on the other fold (validation).
Each time, the model is tested on data it has never seen during training, so the validation accuracy of each fold provides an estimate of the model's performance on unseen data. Moreover,  in cross-validation, it produces $k$ different validation errors, one from each fold. By averaging these $k$ validation errors, we obtain a more stable and reliable approximation of the model's generalization error. Since every fold uses a different subset of the data for validation, the evaluation does not depend on a single train-test split, which could be unrepresentative or susceptible to overfitting.
This reduces the risk of overfitting during model selection and hyperparameter tuning. Instead of optimizing the model based on performance on one specific validation set (which may bias the model toward that particular subset and make the accuracy of this choosing unrepresentitive), cross-validation makes sure that hyperparameters are chosen based on performance across multiple distinct validation sets. Thus, the selection process becomes less local to a particular dataset partition and more global, and actually reflecting the model's behavior across the entire dataset.

4.	True. Adding noise can assist in making sure the model is not too sensitive to little changes in the data. And also it helps to get the idea of how good the model deals with realistic scenarios, where real data can be skewed a lot of times.


"""

part1_q2 = r"""
**Your answer:**

The approach is not justified. By using the test set as an information to the training step, he is cheating and making the model "overfit" on the test set, which is the worst possible thing you can do, as it will make it look as if the, model is great at generalizing, but actually, it used previous knowledge on the test set and that might be the only reason why it looks good, and real generalization is not promised by that approach at all.

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
If $\Delta<0$, we get loss penalty in a less strict way, we might even not get penalty at all for a wrong class that have a bigger score than the correct one, because we only panelize a very "big" gap since $-\Delta>0$. What that would mean specifically in practice, is that the loss no longer enforces that the correct class must score higher than all incorrect classes. Instead, when $\Delta < 0$, we only penalize cases where an incorrect class has a bigger score than the correct one by more than $|\Delta|$. Thus, the model is no longer maximizing the margin between the true class and the other classes, defeating the purpose of using the hinge loss function.

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

The ideal case is that all residuals are as close as possible to 0, maybe even all sits on the 0 line, if possible (not probable, since the data is assumed to be normally distributed and therefore there should be gaussian noise in real cases), meaning that the predictions are exact. Based on the residuals plot, it looks like the fitting is not perfect as for houses worth around 10-30k dollars, we have a deviations of about 5k dollars, which is not so small compared to 10-30k. However, there is a significant amount of data points that have very small residuals, and in any case, it looks like the expected residual is 0, which means that the model is correct in expectation and in anyhow, the MSE and r^2 show the models is pretty good at explaining the variance. The residuals plot got improved (in the sense of smaller variance of the residuals and also smaller MSE and bigger r^2) by implementing non linearities, and then even more by finding the best hyperparameters using CV.
"""

part3_q2 = r"""
**Your answer:**
1. Yes it is still a linear regression model. The non linearities are being added in a pre processing step, and after that it can be considered as a linear model again.
2. In principle, in a compact domain (for instance if y is bounded), the universal approximation theorem applies, and therefore yes, any function can be approximated with ReLU's foor example, and non-linearities. In any case, usually we can treat the y values as bounded because we know that for example a 700k dollars house is not possible for most cases.
3. Like we saw in lecture for using ReLU to "bend" the linear hyper-plane, adding non linearities can "curve" and "bend" and "twist" the hyper-plane and change the shape of it, thus allowing better data separation. Afte that it would not be a hyper-plane, since linearity for elements in it won’t exist anymore.
"""

part3_q3 = r"""
**Your answer:**

1.

The expectation is:
$$
\mathbb{E}[|y - x|], \qquad x, y \sim \mathrm{Uniform}(0,1).
$$

The joint density of $x$ and $y$ on $[0,1]^2$ is 1, so we have:
$$
\mathbb{E}[|y - x|]
= \int_0^1 \int_0^1 |y - x| \, dy \, dx.
$$

The regime is symmetrical, so we can compute only the region where $y \ge x$ and multiply by 2:
$$
\mathbb{E}[|y - x|]
= 2 \int_0^1 \int_x^1 (y - x)\, dy \, dx.
$$

The inner integral:
$$
\int_x^1 (y - x)\, dy
= \left[ \tfrac{1}{2}(y - x)^2 \right]_{y=x}^{1}
= \tfrac{1}{2}(1 - x)^2.
$$

$$
\mathbb{E}[|y - x|]
= 2 \int_0^1 \tfrac{1}{2} (1 - x)^2 \, dx
= \int_0^1 (1 - x)^2 \, dx.
$$

Then:
$$
\int_0^1 (1 - x)^2 dx
= \left[ 1 - x + \tfrac{x^2}{3} \right]_{0}^{1}
= \frac{1}{3}.
$$

$$
\boxed{\mathbb{E}[|y - x|] = \frac{1}{3}}
$$


2.
The expectation is:
$$
\mathbb{E}_x\!\left[\,|\hat{x} - x|\,\right], \qquad x \sim \mathrm{Uniform}(0,1),
$$
where \(\hat{x}\) is fixed in $[0,1]$.

Since $x$ is uniform on $[0,1]$, the expectation is:
$$
\mathbb{E}_x[|\hat{x} - x|]
= \int_0^1 |\hat{x} - x|\,dx.
$$

We split at $x = \hat{x}$:
$$
\int_0^1 |\hat{x} - x|\,dx
= \int_0^{\hat{x}} (\hat{x} - x)\,dx 
  + \int_{\hat{x}}^{1} (x - \hat{x})\,dx.
$$

The first part:
$$
\int_0^{\hat{x}} (\hat{x} - x)\,dx
= \left[\hat{x}x - \frac{x^2}{2}\right]_{0}^{\hat{x}}
= \frac{\hat{x}^2}{2}.
$$

The second part:
$$
\int_{\hat{x}}^{1} (x - \hat{x})\,dx
= \left[\frac{x^2}{2} - \hat{x}x\right]_{\hat{x}}^{1}
= \left(\frac{1}{2} - \hat{x}\right) 
  - \left(\frac{\hat{x}^2}{2} - \hat{x}^2\right)
= \frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}.
$$

Summing both gives:
$$
\mathbb{E}_x[|\hat{x} - x|]
= \frac{\hat{x}^2}{2}
  + \left(\frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}\right)
= \hat{x}^2 - \hat{x} + \frac{1}{2}.
$$

$$
\mathbb{E}_x[|\hat{x} - x|] = \hat{x}^2 - \hat{x} + \frac{1}{2}
$$

3.	Because the $\frac{1}{2}$ is a constant which does not effect the minimization, just as we saw in the derivation of the $L_2$ loss function in lecture, all constants do not matter for the minimization objective. In addition, one can say that in computing the gradient of the loss function, the constant's derivative is 0, therefore it does not effect the gradient steps.  
"""

# ==============

# ==============
