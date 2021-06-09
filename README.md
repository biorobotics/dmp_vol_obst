# README #
This repo is code largely taken from the below work by M. Ginesi et al.; please see their implementation at https://github.com/mginesi/dmp_vol_obst. The code has been modified to include interacting with ROS to recieve start/goal positions, rudimentary singularity avoidance for the d3 robot, and to include planning for a path w.r.t orientation. 


The rest of the README is the original README from the repo this was based on.

This repository contains the implementation of Dynamic Movement Primitives (DMPs), together with different approaches to obstacle avoidance, in Python 3.5.

In particular, this repository contains all the synthetic tests done for the works:

_M. Ginesi, D. Meli, A. Calanca, D. Dall'Alba, N. Sansonetto and P. Fiorini_, **Dynamic Movement Primitives: Volumetric Obstacle Avoidance,** 2019 19th International Conference on Advanced Robotics (ICAR), Belo Horizonte, Brazil, 2019, pp. 234-239.

doi: 10.1109/ICAR46387.2019.8981552

URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8981552&isnumber=8981543

### bibtex ###

`
@INPROCEEDINGS{8981552,
author={Ginesi, Michele and Meli, Daniele and Calanca, Andrea and Dall'Alba, Diego and Sansonetto, Nicola and Fiorini, PAolo},
booktitle={2019 19th International Conference on Advanced Robotics (ICAR)},
title={Dynamic Movement Primitives: Volumetric Obstacle Avoidance},
year={2019},
volume={},
number={},
pages={234-239},
keywords={},
doi={10.1109/ICAR46387.2019.8981552},
ISSN={null},
month={Dec}}
`

and the work, currently under revision,

_M. Ginesi, D. Meli, A. Roberti, N. Sansonetto and P. Fiorini_, **Dynamic Movement Primitives: Volumetric ObstacleAvoidance Using Dynamic Potential Functions**

whose preprint can be found at

https://arxiv.org/abs/2007.00518

## Install ##

The package can be installed by running

```
pip install -e .
```

or

```
pip3 install -e .
```

## Repository Contents ##

The repository contains two folders: _codes_ and _demos_.

### The _codes_ folder ###

The _codes_ folder contains all the scripts performing the basis functions needed to implement DMPs (together with obstacle avoidance).
In particular:
* _cs.py_ implements the so-called "Canonical System";
* _dmp_cartesian.py_ is the class that generates and handle the DMP, able to handle both the learning of the weights given a trajectory and to execute a trajectory given the set of weights;
* _ellipsoid.py_ implements the method using to extract the minimum volume enclosing ellipsoid algorithm [1];
* _exponential_integration.py_ contains the functions needed to perform the numerical integration method "Exponential Euler";
* _point_obstacle.py_ is the class which implements obstacle avoidance for point-like obstacles using the methods presented in [2] and [3];
* _obstacle_superquadric.py_ is the class which implements volumetric obstacle avoidance for ellipses (2 dimensions) and ellipsoids (3 dimensions) using both static (for our first work) and dynamic (for our second work) potential functions

### The _demos_ folder ###

The _demos_ folder contains all the scripts performing the tests proposed in our works.
the folder contains two sub-folders, one for our work with static potentials, and one fo the dynamic potentials.
See the comments at the beginning of the code to associate the test to the figure in the paper

## Theory: quick recall ##

_Dynamic Movement Primitives_ are a framework for trajectory learning. It is based on a second order differential equation of spring-mass-damper type:
\[
\begin{cases}
    \tau \dot{\mathbf{v}} = \mathbf{K} (\mathbf{g} - \mathbf{x}) - \mathbf{D} \mathbf{v} - \mathbf{K} (\mathbf{g} - \mathbf{x}_0) s + \mathbf{K} \mathbf{f}(s) \\
    \tau \dot{\mathbf{x}} = \mathbf{v}
\end{cases},
\]

where $ \mathbf{x}, \mathbf{v} \in \mathbb{R}^d $ are position and velocity of the system; $\mathbf{x}_0 , \mathbf{g} \in \mathbb{R}^d $ are initial and goal position, respectively; $\mathbf{K}, \mathbf{D} \in \mathbb{R}^{d \times d}$ are diagonal matrices, representing the elastic and damping terms, respectively; $\mathbf{f} (s) \in \mathbb{R}^d $ is the "forcing term"; $\tau \in \mathbb{R}^+$ is a parameter used to make the execution faster or slower. Parameter $s \in \mathbb{R}$ is a re-parametrization of time, governed by the _canonical system_
\[ \tau \dot{s} = -\alpha s, \quad \alpha \in \mathbb{R}^+ . \]

### Learning Phase ###

During the learning phase, a desired curve $ \tilde\mathbf{x}(t) $ is shown. This permit to compute the forcing term $\mathbf{f}(s)$. Thus, the forcing term is approximated using Gaussian Radial Basis functions $ \{ \psi_i(s) \}_{i=0,1,\ldots, N} $ as
\[ \mathbf{f}(s) = \frac{ \sum_{i=0}^N \omega_i \psi_i(s) }{ \sum_{i=0}^N \psi_i(s) } s , \]
with $ \omega_i \in \mathbb{R} $, $ i=0,1,\ldots, N. $

### Execution Phase ###

Once the _weights_ $\omega_i$ have been learned, the forcing term can be computed. The dynamical system can be integrated changing $\mathbf{x}_0, \mathbf{g}$, and $\tau$, thus being able to generalize the trajectory changing initial and final positions, and execution time.

### Obstacle Avoidance ###

To avoid obstacles, a _copupling term_ $\bm{\varphi} (\mathbf{x}, \mathbf{v})$ is added to the first equation of the DMP system.
In our approach, we use _Superquadric Potential Functions_. An isopotential
\[ C(\mathbf{x}) = \left( \left( \frac{x_1}{f_1(\mathbf{x})} \right) ^ {2n} + \left( \frac{x_2}{f_2(\mathbf{x})} \right) ^ {2n} \right) ^ \frac{2m}{2n} + \left( \frac{x_3}{f_3(\mathbf{x})} \right) ^ {2m} - 1 \]
that vanishes on the boundary of the obstacle is created by setting parameters $m,n \in \mathbb{N}$ and functions $f_i(\mathbf{x})$.
Then, a potential is created as
\[ U(\mathbf{x}) = \frac{A \exp (-\eta C(\mathbf{x})) }{ C(\mathbf{x}) }, \]
with $ A, \eta \in \mathbb{R}^+ $.
Finally, the coupling term is computed as
\[ \bm{\varphi} (\mathbf{x}, \mathbf{v}) \equiv \bm{\varphi} (\mathbf{x}) = - \nabla_\mathbf{x} U(\mathbf{x}) . \]

## References ##

[1] https://github.com/minillinim/ellipsoid

[2] _Hoffmann, H., Pastor, P., Park, D. H., and Schaal, S._. **Biologically-inspired dynamical systems for movement generation: automatic real-time goal adaptation and obstacle avoidance**. In Robotics and Automation, 2009. ICRA'09. IEEE International Conference on (pp. 2587-2592). IEEE.
