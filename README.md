# Contents:
1. Describtion
2. Problem
3. Method
    * 3.1. Dataset
    * 3.2. Model
    * 3.3. Cost function
    * 3.4. Optimization procedure
    * 3.5. Initializing data
4. Code
5. Results

# 1. Describtion
A hello world project to practice making a simple neural network with PyTorch.
The whole code and idea is credited to Josh Starmer (https://youtu.be/FHdlXe1bSe4?si=kzO6fpvrfrzfXpM6)

# 2. Problem
Abstract problem of defining the effective medicine dosage.

# 3. Method
Algorithm structure is comprised of 5 parts: Dataset, Model, Cost function, Optimization procedure and Initializing data. Describtion of every part is provided below.
## 3.1. Dataset
Abstract synthetic dataset.

$$
𝕏=\lbrace{0, 0.5, 1 \rbrace},~
f^*=\begin{bmatrix}
    0 \\
    1 \\
    0 \\
    \end{bmatrix}
$$

## 3.2. Model
### > FNN architecture:
<p align="center">
  <img src="https://github.com/AKAD0/pytorch_hellow/blob/master/Fig1.png">
</p>

$$
\text{Fig.1: Topology of the architecture}
$$
### > Output layer:
$$
y=f^{(2)}(h; w,b_final) = \sum_ {}(w^Th)+b_final
$$

$$
\\
\begin{aligned}
\text{where}~
&w-\text{weights vector of}~f^{(2)} \\
&b_final-\text{biases vector of}~f^{(2)} \\
&h-\text{output vector of}~f^{(1)} \\
\end{aligned}
$$


### > Hidden layer:
$$
h = f^{(1)} = g( z( x; W,b))
$$

$$
\\
\begin{aligned}
\text{where}~
&x-\text{input vector} \\
&W-\text{weights vector of}~f^{(1)} \\
&b-\text{biases vector of}~f^{(1)} \\
&z(x; W,b)=x^TW+b-\text{input function} \\
&g(z_i)=max\lbrace0, z_{:,i}\rbrace-\text{activation function ReLU}
\end{aligned}
$$


### > Input layer:
$$
X \in ℕ^{11×1}
$$

Every point represents a sample of one number to pass to FNN.

### > Composition
$$
f(x; W,b,w,b_final) = f^{(2)}( f^{(1)}( x)) = w^Tmax\lbrace0, W^Tx+b\rbrace+b_final
$$

### > Cost Function
$$
J(θ) = ( f^{*}(x) - f(x;θ))^2
$$

$$
\\
\begin{aligned}
\text{where}~
&θ-\text{optimizing parameters W,b,w,bfinal} \\
&x-\text{input data} \\
&f^{*}-\text{true function} \\
&f-\text{approximating function}
\end{aligned}
$$
### > Optimization procedure
For the problem the SGD was chosen to optimize the problem.

Only the b_final was setted to optimize.

### > Initializing data
$$
W = \begin{bmatrix}
    1.70 \\
    12.6 \\
    \end{bmatrix}
,~
b = \begin{bmatrix}
    -0.85 \\
    0 \\
    \end{bmatrix}
,~
w = \begin{bmatrix}
    -40.8 \\
    2.7 \\
    \end{bmatrix}
,~
b_final=0
$$

# 4. Code

Working script is provided in the '/script.ipynb' file.

# 5. Results
Simple human check of outputted results show that the algorithm solves the problem.

<p align="center">
  <img src="https://github.com/AKAD0/pytorch_hellow/blob/master/Fig2.png">
</p>

$$
\text{Fig.2: Resulting outputs}
$$