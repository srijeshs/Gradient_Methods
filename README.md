# Gradient Methods
Demonstration of gradient descent methods

## Description of files
- The script Gradient_Descent.m solves the linear system <img src="https://render.githubusercontent.com/render/math?math=y%20\=%20Hx"> using the steepest gradient descent method, where:  
     <img src="https://render.githubusercontent.com/render/math?math=H%20\in%20\mathbb{R}^{m%20\times%20n}">,  <img src="https://render.githubusercontent.com/render/math?math=x%20\in%20\mathbb{R}^{n%20\times%201}"> and <img src="https://render.githubusercontent.com/render/math?math=y%20\in%20\mathbb{R}^{m%20\times%201}">   
     
## Visualization
- The script Gradient_Visualize.m solves the same system in 2 dimensions and plots the trajectory of the solution vector ( in red ) in the solution search space at each iteration.
- Eventually, the solution vector converges to the expected analytical solution ( in green )

![gradient_visualization](https://github.com/srijeshs/Gradient_Methods/blob/master/GDescent_FF-min.gif "Visualization of Gradient Descent")
