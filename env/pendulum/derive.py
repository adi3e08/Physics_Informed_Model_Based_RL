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
    n = 1
    m1 = symbols('m1')
    l1 = symbols('l1')
    r1 = symbols('r1')
    I1 = symbols('I1')
    g = symbols('g')
    q1 = symbols('q1')
    q1dot = symbols('q1dot')

    m = [m1]
    l = [l1]
    r = [r1]
    I = [I1]
    inertials = m+l+r+I+[g]

    q = Matrix([q1])
    qdot = Matrix([q1dot])
    state = [q1,q1dot]

    J_w = Matrix([[1]
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
        
        M_i = I[i] * J_w[i,:].T * J_w[i,:]
        M += M_i
             
        V += m[i]*g*center[0,1]

    # print(cse([centers,joints,J_w]+J, optimizations='basic'))

    C,G = get_C_G(n,M,V,q,qdot)
    lambda_dict['kinematics'] = lambdify([tuple(inertials+state)],[centers,joints,angles],'numpy',cse=True)
    lambda_dict['dynamics'] = lambdify([tuple(inertials+state)],[M,C,G],'numpy',cse=True)
    
    with open("./env/pendulum/robot.p", "wb") as outf:
        pickle.dump(lambda_dict, outf)

    print("Done")

if __name__ == '__main__':
    derive()
