r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


1.
The Jacobian $\frac{\partial Y}{\partial X}$ has derivatives of all
$64 \cdot 512$ output scalars with respect to all
$64 \cdot 1024$ input scalars. Therefore,
$$
\frac{\partial Y}{\partial X} \in \mathbb{R}^{(64\cdot512)\times(64\cdot1024)} .
$$

2.
Viewing the Jacobian as a 2D block matrix, it has $64 \times 64$ blocks,
each of size $512 \times 1024$.
The block $(i,j)$ corresponds to
$\frac{\partial Y_i}{\partial X_j}$.
Since each output row $Y_i$ depends only on the input row $X_i$,
all off-diagonal blocks are zero.
So the Jacobian is a block-diagonal, and each diagonal block is equal to $W$.

3.
Because the Jacobian is block-diagonal with identical blocks, it does not need
to be materialized explicitly.
It can be represented implicitly using the weight matrix $W$ and the batch
structure, with effective size $64 \times 512 \times 1024$.

4.
Given the downstream gradient
$\delta Y = \frac{\partial L}{\partial Y} \in \mathbb{R}^{64 \times 512}$,
we get the gradient with respect to the input using the
vector Jacobian product:
$$
\delta X = \frac{\partial L}{\partial X} = \delta Y \, W \in \mathbb{R}^{64 \times 1024},
$$
and this computation does not require explicit construction of the Jacobian.

5.
The Jacobian with respect to the weights has the shape:
$$
\frac{\partial Y}{\partial W} \in \mathbb{R}^{(64\cdot512)\times(512\cdot1024)} .
$$
This Jacobian is not block diagonal, since each output element depends on all
weights in the corresponding row of $W$.
As a result, the gradient with respect to the weights is
$$
\frac{\partial L}{\partial W} = \delta Y^{\top} X \in \mathbb{R}^{512 \times 1024}.
$$
"""


part1_q2 = r"""
**Your answer:**

Yes, second order derivatives can be useful in optimization. They are usually
not used in standard gradient descent since it is very expensive to compute the Hessian.
Second order derivatives (the Hessian) give information about the curvature of
the loss function/space. This can help distinguish between a minimum point or a saddle points, and
flat regions where the gradient is small but the point is not a minimum point.
Methods such as the Newton method use the second order information to change/control the step
direction and size, which can lead to faster convergence.
Since it is usually very expensive, approximations of the Hessian are often used instead.


"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.01
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr_vanilla = 0.02
    lr_momentum = 0.002
    lr_rmsprop = 0.00015
    reg = 0.002
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.15
    lr = 0.0008 
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1.
Yes, the results match what we expect.
Without dropout, the training error is lower and the training accuracy is higher,
which indicates overfitting. On the test set, the no-dropout model performs worse.
With dropout, the training performance decreases, but the test performance improves,
showing better generalization.

2.
With higher dropout, the training accuracy is lower and the error is higher, which
is expected since more neurons are dropped during training, which kind of "takes away" the chance of getting perfect overfitting.
On the test set, low dropout performs better than high dropout, since too much
dropout causes underfitting. Low-mid dropout gives a better balance between preventing
overfitting and keeping enough model capacity.


"""

part2_q2 = r"""

Yes, this is possible.

Accuracy depends only on whether predictions are correct or not, while the cross entropy loss
also depends on the confidence/probability of the predictions. It is possible that the model
becomes more confident on many correctly classified examples, which reduces the
overall loss, while at the same time a small number of samples cross the decision
threshold and become misclassified, causing the accuracy to decrease.

For example, a single sample may change from being barely correct to barely
incorrect, reducing accuracy in percentage, while the confidence on many other samples increases
enough to decrease the total loss value.


"""

part2_q3 = r"""


1.
Gradient Descent (GD) computes the gradient using the entire dataset at every step,
which makes the updates stable and with low variance. However, it is expensive and may
get stuck in a local minimum or flat regions.
Stochastic Gradient Descent (SGD) uses a single sample or a small batch, which
introduces noise into the gradient. This noise increases variance but can help
escape local minimum and often leads to faster progress in practice.

2.
Yes, momentum should also be used with GD.
Momentum helps smooth oscillations/flactuations and it accelerates convergence by accumulating
past gradients and using those to dictate the next steps. These advantages apply to GD as well, although SGD usually benefits
from momentum even more due to its higher variance.

3.
(a)
Yes, this approach produces the same gradient as GD.
If the total loss is
$L = \sum_{i=1}^N L_i$, then
$\nabla L = \sum_{i=1}^N \nabla L_i$.
Computing losses on disjoint batches and summing them before backpropagation is
mathematically equivalent to computing the gradient on the full dataset (sum of derivative is the derivative of the sum).

(b)
The out-of-memory error occurred because the computation graph of each forward
pass was kept in memory until the backward pass. As more batches were processed,
all intermediate activations were stored, eventually exhausting memory (מה הועילו חכמים בתקנתם).

(c)
To solve this, one should perform a backward pass per batch and accumulate the
gradients, or explicitly detach the graph after each batch. This prevents storing
all intermediate activations at once and keeps memory usage small.


"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4
    hidden_dims = 8
    activation = "relu"
    out_activation = "relu"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.01
    weight_decay = 0.01
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""


1.
(a) Optimization error refers to difficulties in minimizing the training loss due
to the training procedure itself, such as poor choice of learning rate, optimizer,
or getting stuck in local minimum or oscillations.

(b) Generalization error refers to the gap between performance on the training set
and the validation/test set. It usually appears when the model fits the training
data well but performs significantly worse on unseen data (overfitting).

(c) Approximation error refers to limitations of the model's capacity. If the model
is too simple to represent the true decision boundary, it will perform poorly even
on the training set.

2.
The approximation error is low, since the model achieves very high accuracy on the
training set and also good accuracy on the validation set.  
There is some generalization error: after a certain point, training continues to
improve while validation performance stops improving and slightly degrades, which
indicates mild overfitting.  
There does not appear to be a high optimization error, as the loss decreases
smoothly and the accuracy increases as expected on both training and validation
sets.


"""

part3_q2 = r"""

1.
We would prefer to minimize the false positive rate (FPR) when false positives are
very costly. For example, in spam email filtering, classifying an important email
as spam (false positive) may cause us to miss critical information. On the other
hand, allowing some spam emails into the inbox (false negatives) is usually less
harmful.

2.
We would prefer to minimize the false negative rate (FNR) when missing a positive
case is dangerous. For example, in medical diagnosis, classifying a sick patient
as healthy (false negative) can prevent them from receiving treatment, which is
very serious. In contrast, classifying a healthy patient as sick (false positive)
may lead to additional tests or treatment, which is typically less harmful (assuming that redundant treatment does not do any harm).

"""

part3_q3 = r"""


1.
When the depth is fixed and the width increases, the decision boundary becomes more
expressive. With depth=1 the boundary is almost linear and cannot separate the data
well. Increasing the width improves the separation, but after a certain point
(e.g. width 8 or 32) the improvement is small, indicating diminishing returns.

2.
When the width is fixed and the depth increases, the decision boundary becomes more
complex and highly non-linear. Deeper models can fit more complicated shapes, but
this also increases the risk of overfitting. In some cases the boundary appears
overly complex relative to the data, which can hurt generalization.

3.
Although both configurations have the same number of parameters, depth=4, width=8
uses more non-linear transformations than depth=1, width=32. This gives it higher
representational power, but also a higher risk of overfitting. In practice, the
shallower model achieved slightly higher accuracy, while the deeper model produced
a more detailed decision boundary and a lower optimal threshold, which may reduce
the FNR.

4.
Threshold selection on the validation set improves test performance because it
allows choosing a better tradeoff between FPR and FNR. By tuning the threshold, the
classifier can be adapted to the task's priorities instead of relying on a default
value.


"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.03
    weight_decay = 0.01
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""

We compare two residual blocks operating on a 256-channel input:

(A) Regular block:
two 3x3 convolutions, 256→256.

(B) Bottleneck block:
1x1 (256→64), then 3x3 (64→64), then 1x1 (64→256).

1) Number of parameters

A convolution with kernel KxK, C_in input channels and C_out output channels has
K^2 · C_in · C_out parameters.

(A) Regular block:
Each 3x3 conv:
3^2 · 256 · 256 = 589,824  
Two such layers:
$$
2 \cdot 589{,}824 = 1{,}179{,}648
$$

(B) Bottleneck block:
First 1x1:
1^2 · 256 · 64 = 16,384  
Middle 3x3:
3^2 · 64 · 64 = 36,864  
Last 1x1:
1^2 · 64 · 256 = 16,384  
Total:
$$
16{,}384 + 36{,}864 + 16{,}384 = 69{,}632
$$

Thus, the bottleneck uses about 17 times fewer parameters.

2) FLOPs (qualitative)

The main cost comes from 3x3 convolutions.
The regular block has two expensive 3x3 convs on 256 channels.
The bottleneck has only one 3x3 conv, and it operates on 64 channels, while the
1x1 convs are much cheaper.
Therefore, the bottleneck requires significantly fewer FLOPs.

3) Ability to combine the input

(a) Spatially:
The regular block has two 3x3 convolutions, leading to stronger spatial mixing
and a larger effective receptive field.
The bottleneck has only one 3x3 convolution, so spatial mixing is more limited.

(b) Across feature maps:
The bottleneck uses 1x1 convolutions to explicitly mix channels, first reducing
256 channels to 64 and then expanding back to 256.
This provides strong and flexible channel mixing, while keeping spatial
computation efficient.

"""


part4_q2 = r"""


1)
We have $y_1 = M x_1$. Let $g_1 := \frac{\partial L}{\partial y_1}$.
Using the chain rule for a linear map:
$$
\frac{\partial L}{\partial x_1}
= \left(\frac{\partial y_1}{\partial x_1}\right)^{\top} \frac{\partial L}{\partial y_1}
= M^{\top} g_1 .
$$

2)
We have $y_2 = x_2 + M x_2 = (I + M)x_2$. Let $g_2 := \frac{\partial L}{\partial y_2}$.
Again by the chain rule:
$$
\frac{\partial L}{\partial x_2}
= \left(\frac{\partial y_2}{\partial x_2}\right)^{\top} \frac{\partial L}{\partial y_2}
= (I+M)^{\top} g_2
= (I + M^{\top}) g_2 .
$$

3)
In a deep stack of such layers without residuals, each layer contributes a factor
of $M^{\top}$ to the backpropagated gradient, so after $k$ layers the gradient is
multiplied by $(M^{\top})^k$. Since $M$ has small entries (and typically spectral
norm < 1), repeated multiplication shrinks the gradient toward zero, i.e. vanishing
gradients.

With residual layers, each layer contributes $(I+M^{\top})$ instead. This keeps an
identity path for the gradient:
$$
(I+M^{\top}) g = g + M^{\top} g .
$$
So even if $M^{\top} g$ is small, the term $g$ is passed through unchanged.
Across many layers, the gradient is multiplied by products of $(I+M^{\top})$ rather
than $(M^{\top})$, which prevents it from collapsing to zero and makes training of
very deep networks much more stable.


"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
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
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""