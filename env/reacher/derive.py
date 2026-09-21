---
title: "Rigid body dynamics simulation"
project_type : blog
permalink: /blog/rbds
date: 2026-09-21
excerpt: 'blah blah.<br><br><img src="https://adi3e08.github.io/files/research/pimbrl/envs.png" width="100%"/>'
author_profile: False
---
Adithya Ramesh.<br>
Published at <a href="https://l4dc.seas.upenn.edu/" target="_blank">Learning for Dynamics & Control Conference (L4DC), 2023</a>.

<h3> <a href="https://arxiv.org/abs/2212.02179" target="_blank">Paper</a> <span>&#124;</span> <a href="https://github.com/adi3e08/Physics_Informed_Model_Based_RL" target="_blank">Code</a> </h3> 

## Abstract
We focus on the robotics domain, mainly systems undergoing rigid body motion such as robotic manipulators. For such systems, it is easier to derive the equations of motion using Lagrangian or Hamiltonian mechanics rather than Newtonian mechanics. Conventionally, the Lagrangian approach is taken.

## Environments
We focus on robotic systems undergoing rigid body motion without contacts. We also assume that there is no friction. The environments considered are shown below. We implement our own simulations from first principles.
<p align="center">
<img src="https://adi3e08.github.io/files/research/pimbrl/envs.png" width="100%"/>
</p>

In future work, we plan to include both friction as well as contacts. 

## Lagrangian Mechanics
These systems obey Lagrangian mechanics. Their state consists of generalized coordinates $\textbf{q}$, which describe the configuration of the system, and generalized velocities $\dot{\textbf{q}}$, which are the time derivatives of $\textbf{q}$. Let the motor torques be $\boldsymbol\tau$. The Lagrangian equations of motion are given by,
\\[\textbf{M}(\textbf{q}) \, \ddot{\textbf{q}} + \textbf{C}(\textbf{q},\dot{\textbf{q}}) \, \dot{\textbf{q}} + \textbf{G}(\textbf{q}) = \boldsymbol\tau\\]

where, $\textbf{M}(\textbf{q})$ is the mass matrix, $\textbf{C}(\textbf{q},\dot{\textbf{q}}) \, \dot{\textbf{q}} = \frac{\partial }{\partial \textbf{q}} \big(\textbf{M}(\textbf{q})\, \dot{\textbf{q}} \big) \, \dot{\textbf{q}} - \frac{\partial }{\partial \textbf{q}} \big( \frac{1}{2} \, \dot{\textbf{q}}^{T} \, \textbf{M}(\textbf{q})\, \dot{\textbf{q}} \big)$ is the centripetal / Coriolis term and $\textbf{G}(\textbf{q}) = \frac{\partial \mathcal{V}(\textbf{q})}{\partial \textbf{q}}$ is the gravitational term, where $\mathcal{V}(\textbf{q})$ is the potential energy.

## Sympy kinematics model
Below we show how to derive the mass matrix $\textbf{M}(\textbf{q})$, the centripetal / Coriolis term $\textbf{C}(\textbf{q},\dot{\textbf{q}}) \, \dot{\textbf{q}}$ and the gravitational term $\textbf{G}(\textbf{q})$.
```
from sympy import symbols,cos,sin,simplify,diff,Matrix,linsolve,expand,nsimplify,zeros,flatten
from sympy.utilities.lambdify import lambdify
from sympy.matrices.dense import matrix_multiply_elementwise
import dill as pickle
pickle.settings['recurse'] = True

def get_C_G(n,M,V,q,qdot):
    C = zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += (diff(M[i,j],q[k])+diff(M[i,k],q[j])-diff(M[k,j],q[i]))*qdot[k]/2

    G = Matrix([diff(V,q[i]) for i in range(n)])

    return C,G
    
def derive():
    lambda_dict = {}
    n = 2
    m1,m2 = symbols('m1,m2')
    l1,l2 = symbols('l1,l2')
    r1,r2 = symbols('r1,r2')
    I1,I2 = symbols('I1,I2')
    g = symbols('g')
    q1,q2 = symbols('q1,q2')
    q1dot,q2dot = symbols('q1dot,q2dot')

    m = [m1,m2]
    l = [l1,l2]
    r = [r1,r2]
    I = [I1,I2]
    inertials = m+l+r+I+[g]

    q = Matrix([q1,q2])
    qdot = Matrix([q1dot,q2dot])
    state = [q1,q2,q1dot,q2dot]

    J_w = Matrix([[1,0],
                  [1,1]
                 ])

    angles = J_w * q

    V = 0
    M = zeros(n)
    J = []
    for i in range(n):
        if i == 0:
            joint = Matrix([[0, 0]])
            center = joint + (r[i])*Matrix([[sin(angles[i]),cos(angles[i])]])
            joints = joint
            centers = center
        else:
            joint = joint + l[i-1]*Matrix([[sin(angles[i-1]),cos(angles[i-1])]])
            center = joint + (r[i])*Matrix([[sin(angles[i]),cos(angles[i])]])
            joints = Matrix.vstack(joints, joint)
            centers = Matrix.vstack(centers, center)
        
        J_v = center.jacobian(q)
        # J.append(J_v)
        M_i = m[i] * J_v.T * J_v + I[i] * J_w[i,:].T * J_w[i,:]
        M += M_i
             
        V += m[i]*g*center[0,1]

    # print(cse([centers,joints,J_w]+J, optimizations='basic'))

    C,G = get_C_G(n,M,V,q,qdot)
    lambda_dict['kinematics'] = lambdify([tuple(inertials+state)],[centers,joints,angles],'numpy',cse=True)
    lambda_dict['dynamics'] = lambdify([tuple(inertials+state)],[M,C,G],'numpy',cse=True)
    
    with open("./env/reacher/robot.p", "wb") as outf:
        pickle.dump(lambda_dict, outf)

    print("Done")

if __name__ == '__main__':
    derive()

```

## Numerical simulation

### RK1 / Euler integration
$$\begin{bmatrix}\boldsymbol{q}_{t+1} \\ \dot{\boldsymbol{q}}_{t+1}\end{bmatrix} = \begin{bmatrix}\boldsymbol{q}_{t} \\ \dot{\boldsymbol{q}}_{t}\end{bmatrix} + \Delta_{t} \, f(\boldsymbol{q}_{t}, \dot{\boldsymbol{q}}_{t}, \boldsymbol{\tau})$$

### RK2
$$\boldsymbol{k1} = f(\boldsymbol{q}_{t}, \dot{\boldsymbol{q}}_{t}, \boldsymbol{\tau})\\
\boldsymbol{k2} = f((\boldsymbol{q}_{t}, \dot{\boldsymbol{q}}_{t}) + \Delta_{t} \, k1, \boldsymbol{\tau})\\
\begin{bmatrix}\boldsymbol{q}_{t+1} \\ \dot{\boldsymbol{q}}_{t+1}\end{bmatrix} = \begin{bmatrix}\boldsymbol{q}_{t} \\ \dot{\boldsymbol{q}}_{t}\end{bmatrix} + \Delta_{t} \, (k1+k2)$$

### RK4
$$\boldsymbol{k1} = f(\boldsymbol{q}_{t}, \dot{\boldsymbol{q}}_{t}, \boldsymbol{\tau})\\
\boldsymbol{k2} = f((\boldsymbol{q}_{t}, \dot{\boldsymbol{q}}_{t}) + \frac{\Delta_{t}}{2} \, k1, \boldsymbol{\tau})\\
\boldsymbol{k3} = f((\boldsymbol{q}_{t}, \dot{\boldsymbol{q}}_{t}) + \frac{\Delta_{t}}{2} \, k2, \boldsymbol{\tau})\\\boldsymbol{k4} = f((\boldsymbol{q}_{t}, \dot{\boldsymbol{q}}_{t}) + \Delta_{t} \, k3, \boldsymbol{\tau})\\
\begin{bmatrix}\boldsymbol{q}_{t+1} \\ \dot{\boldsymbol{q}}_{t+1}\end{bmatrix} = \begin{bmatrix}\boldsymbol{q}_{t} \\ \dot{\boldsymbol{q}}_{t}\end{bmatrix} + \frac{\Delta_{t}}{6} \, (k1+2\,k2+2\,k3+k4)$$

### Energy check
To verify the correctness of the simulation, we plot the total energy as a function of time.


## Videos
We show the behaviours learnt by our physics-informed model-based RL approach below.
<div align="center"> 
 <video title="Reacher" style="outline:none; width:24.5%;" controls>
  <source src="https://adi3e08.github.io/files/research/pimbrl/reacher.mp4" type="video/mp4">
</video> 
 <video title="Pendulum" style="outline:none; width:24.5%;" controls>
  <source src="https://adi3e08.github.io/files/research/pimbrl/pendulum.mp4" type="video/mp4">
</video> 
 <video title="Cartpole" style="outline:none; width:24.5%;" controls>
  <source src="https://adi3e08.github.io/files/research/pimbrl/cartpole.mp4" type="video/mp4">
</video> 
 <video title="Cart2pole" style="outline:none; width:24.5%;" controls>
  <source src="https://adi3e08.github.io/files/research/pimbrl/cart2pole.mp4" type="video/mp4">
</video> 
</div>

<div align="center"> 
 <video title="Acrobot" style="outline:none; width:24.5%;" controls>
  <source src="https://adi3e08.github.io/files/research/pimbrl/acrobot.mp4" type="video/mp4">
</video> 
 <video title="Cart3pole" style="outline:none; width:24.5%;" controls>
  <source src="https://adi3e08.github.io/files/research/pimbrl/cart3pole.mp4" type="video/mp4">
</video> 
 <video title="Acro3bot" style="outline:none; width:24.5%;" controls>
  <source src="https://adi3e08.github.io/files/research/pimbrl/acro3bot.mp4" type="video/mp4">
</video> 
</div>
