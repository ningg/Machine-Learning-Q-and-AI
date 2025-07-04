







# Chapter 12: Fully Connected and Convolutional Layers [](#chapter-12-fully-connected-and-convolutional-layers)



**Under which circumstances can we replace fully connected layers with
convolutional layers to perform the same computation?**

Replacing fully connected layers with convolutional layers can offer
advantages in terms of hardware optimization, such as by utilizing
specialized hardware accelerators for convolution operations. This can
be particularly relevant for edge devices.

There are exactly two scenarios in which fully connected layers and
convolutional layers are equivalent: when the size of the convolutional
filter is equal to the size of the receptive field and when the size of
the convolutional filter is 1. As an illustration of these two
scenarios, consider a fully connected layer with two input and four
output units, as shown in
Figure [1.1](#fig-ch12-fig01){reference="fig-ch12-fig01"
reference-type="ref"}.

![Four inputs and\
two outputs connected via\
eight weight parameters](../images/ch12-fig01.png){#fig-ch12-fig01}

The fully connected layer in this figure consists of eight weights and
two bias units. We can compute the output nodes via the following dot
products:

Node 1

\\\[w\_{1, 1} \\times x_1 + w\_{1, 2} \\times x_2 + w\_{1, 3} \\times
x_3 + w\_{1, 4} \\times x_4 + b_1\\\]

Node 2

\\\[w\_{2, 1} \\times x_1 + w\_{2, 2} \\times x_2 + w\_{2, 3} \\times
x_3 + w\_{2, 4} \\times x_4 + b_2\\\]

The following two sections illustrate scenarios in which convolutional
layers can be defined to produce exactly the same computation as the
fully connected layer described.

## When the Kernel and Input Sizes Are Equal [](#when-the-kernel-and-input-sizes-are-equal)

Let's start with the first scenario, where the size of the
convolutional filter is equal to the size of the receptive field. Recall
from Chapter [\[ch11\]](../ch11){reference="ch11" reference-type="ref"}
how we compute a number of parameters in a convolutional kernel with one
input channel and multiple output channels. We have a kernel size of
2\\(\\times\\)2, one input channel, and two output channels. The input
size is also 2\\(\\times\\)2, a reshaped version of the four inputs
depicted in Figure [1.2](#fig-ch12-fig02){reference="fig-ch12-fig02"
reference-type="ref"}.

![A convolutional layer with a 2Ã---2 kernel\
that equals the input size and two output
channels](../images/ch12-fig02.png){#fig-ch12-fig02}

If the convolutional kernel dimensions equal the input size, as depicted
in Figure [1.2](#fig-ch12-fig02){reference="fig-ch12-fig02"
reference-type="ref"}, there is no sliding window mechanism in the
convolutional layer. For the first output channel, we have the following
set of weights:

\\\[{W}\_1 = \\begin{bmatrix} w\_{1, 1} & w\_{1, 2}\\\\ w\_{1, 3} &
w\_{1, 4} \\end{bmatrix}\\\]

For the second output channel, we have the following set of weights:

\\\[{W}\_2 = \\begin{bmatrix} w\_{2, 1} & w\_{2, 2}\\\\ w\_{2, 3} &
w\_{2, 4} \\end{bmatrix}\\\]

If the inputs are organized as

\\\[{x} = \\begin{bmatrix} x\_{1} & x\_{2}\\\\ x\_{3} & x\_{4}
\\end{bmatrix}\\\]

we calculate the first output channel as *o*~1~ = \\(\\sum_i\\)(*W*~1~
\* **x**)*~i~* + *b*~1~, where the convolutional operator \* is equal to
an element-wise multiplication. In other words, we perform an
element-wise multiplication between two matrices, *W*~1~ and **x**, and
then compute the output as the sum over these elements; this equals the
dot product in the fully connected layer. Lastly, we add the bias unit.
The computation for the second output channel works analogously: *o*~2~
= \\(\\sum_i\\)(*W*~2~ \* **x**)*~i~* + *b*~2~.

As a bonus, the supplementary materials for this book include PyTorch
code to show this equivalence with a hands-on example in the
`supplementary/q12-fc-cnn-equivalence`{.language-plaintext
.highlighter-rouge} subfolder at
<https://github.com/rasbt/MachineLearning-QandAI-book>.

## When the Kernel Size Is 1 [](#when-the-kernel-size-is-1)

The second scenario assumes that we reshape the input into an input
"image"? with \\(1\\times1\\) dimensions where the number of "color
channels"? equals the number of input features, as depicted in
Figure [1.3](#fig-ch12-fig03){reference="fig-ch12-fig03"
reference-type="ref"}.

![The number of output nodes equals the number\
of channels if the kernel size is equal to the input
size.](../images/ch12-fig03.png){#fig-ch12-fig03}

Each kernel consists of a stack of weights equal to the number of input
channels. For instance, for the first output layer, the weights are

\\\[{W}\_1 = \[ w\^{(1)}\_{1} w\^{(2)}\_{1} w\^{(3)}\_{1}
w\^{(4)}\_{1}\]\\\]

while the weights for the second channel are:

\\\[{W}\_2 = \[ w\^{(1)}\_{2} w\^{(2)}\_{2} w\^{(3)}\_{2}
w\^{(4)}\_{2}\]\\\]

To get a better intuitive understanding of this computation, check out
the illustrations in Chapter [\[ch11\]](../ch11){reference="ch11"
reference-type="ref"}, which describe how to compute the parameters in a
convolutional layer.

## Recommendations [](#recommendations)

The fact that fully connected layers can be implemented as equivalent
convolutional layers does not have immediate performance or other
advantages on standard computers. However, replacing fully connected
layers with convolutional layers can offer advantages in combination
with developing specialized hardware accelerators for convolution
operations.

Moreover, understanding the scenarios where fully connected layers are
equivalent to convolutional layers aids in understanding the mechanics
of these layers. It also lets us implement convolutional neural networks
without any use of fully connected layers, if desired, to simplify code
implementations.

### Exercises [](#exercises)

12-1. How would increasing the stride affect the equivalence discussed
in this chapter?

12-2. Does padding affect the equivalence between fully connected layers
and convolutional layers?

\

------------------------------------------------------------------------

