from sympy import symbols,cos,sin,simplify,diff,Matrix,linsolve,expand,nsimplify,zeros
from sympy.utilities.lambdify import lambdify
import dill as pickle
pickle.settings['recurse'] = True

def derive():
    n = 1
    m1,l1,r1,I1,g = symbols('m1,l1,r1,I1,g')
    q1,q1dot = symbols('q1,q1dot')
    a1 = symbols('a1')

    q = Matrix([q1])
    qdot = Matrix([q1dot])   

    T = 0.5*I1*(q1dot**2)

    V = m1*g*r1*cos(q1)

    L = T-V

    M1 = expand(diff(L,q1dot))
    M = nsimplify(Matrix([[M1.coeff(q1dot)]]))
    det = M[0,0]
    Minv = Matrix([[1]])/det

    C = zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += (diff(M[i,j],q[k])+diff(M[i,k],q[j])-diff(M[k,j],q[i]))*qdot[k]/2

    G = Matrix([diff(V,q[i]) for i in range(n)])

    qddot = Minv*(Matrix([a1]) - C*qdot - G)

    F = Matrix([qdot[i] for i in range(n)]+[qddot[i] for i in range(n)])
    # print(F)
    H = T + V
    s = Matrix([q[i] for i in range(n)]+[qdot[i] for i in range(n)])
    D_s = F.jacobian(s)
    D_a = F.jacobian(Matrix([a1]))
    # print(D)

    H_lambda = lambdify([tuple([m1,l1,r1,I1,g]+[q1,q1dot])],H,'numpy',cse=True)
    F_lambda = lambdify([tuple([m1,l1,r1,I1,g]+[q1,q1dot]+[a1])],F,'numpy',cse=True)
    D_lambda = lambdify([tuple([m1,l1,r1,I1,g]+[q1,q1dot]+[a1])],(D_s,D_a),'numpy',cse=True)
    
    with open("./env/pendulum/dynamics.p", "wb") as outf:
        pickle.dump({'H':H_lambda,'F':F_lambda,'D':D_lambda}, outf)

    print("Done")

def check():
    inertial_params = [1.0,1.0,1.0,1.0,9.8]
    s = [1.0,1.0]
    a = [1.0]

    with open("./env/pendulum/dynamics.p", "rb") as inf:
        funcs = pickle.load(inf)
    
    H, F, D = funcs['H'], funcs['F'], funcs['D']        
    print(H(inertial_params+s))
    print(F(inertial_params+s+a).flatten())
    print(D(inertial_params+s+a))

if __name__ == '__main__':
    derive()
    check()