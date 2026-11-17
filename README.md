<h1 align="center">Scalar Autograd from Scratch</h1><p align="center"><strong>A small Python implementation of automatic differentiation, checked against PyTorch.</strong></p><p align="center">
  <img src="https://img.shields.io/badge/Python-1D4ED8?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-B93822?style=for-the-badge&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Jupyter-9A3412?style=for-the-badge&amp;logo=jupyter&amp;logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Automatic_Differentiation-6D28D9?style=for-the-badge" alt="Automatic differentiation">
</p>I built a scalar autograd engine to see how the chain rule works in code. The notebook also includes a discrete sampler that turns uniform random draws into category indices using cumulative probabilities.
Scalar autograd
Each MyScalar stores its value, the local derivative of the operation that produced it, and a reference to its parent. To compute gradients, get_gradient starts at the output with a derivative of 1 and walks backward, multiplying by each local derivative along the way.
Saved check	Gradient with respect to a at a = 2
Custom engine for exp(a²)	218.3926
PyTorch autograd for the same expression	218.3926


The notebook checks this example with math.isclose at a relative tolerance of 1e-5. The engine supports chains of scalar operations with one variable input per step, including exp, ln, sin, cos, and arithmetic with constants.
Discrete distribution sampler
my_sampler checks that the probabilities are nonnegative and sum to approximately one. It draws uniform values with torch.rand and uses cumulative probability thresholds to assign each draw to a category. The optional requires_grad flag makes the returned floating-point tensor a leaf that can receive gradients in later calculations. Sampling itself runs without gradient tracking.

The histogram is the saved result of 10,000 draws with category probabilities [0.1, 0.2, 0.7] for indices 0, 1, and 2.
Run the notebook
Open the .ipynb file in this repository to view the saved results. To run it locally, install the dependencies and start JupyterLab from the repository folder.
python -m pip install torch matplotlib jupyterlab
jupyter lab
Run the cells in order to generate a new sample histogram and reproduce the gradient comparison.
