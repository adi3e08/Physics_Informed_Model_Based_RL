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

    J_w = Matrix([[0,0],
                  [0,1]
                 ])

    angles = J_w * q

    V = 0
    M = zeros(n)
    J = []
    for i in range(n):
        if i == 0:
            joint = Matrix([[q1, 0]])
            center = joint
            joints = joint
            centers = center
        elif i == 1:
            joint = joint
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
    
    with open("./env/cartpole/robot.p", "wb") as outf:
        pickle.dump(lambda_dict, outf)

    print("Done")

if __name__ == '__main__':
    derive()
