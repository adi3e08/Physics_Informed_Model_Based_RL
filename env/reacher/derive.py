from sympy import symbols,cos,sin,simplify,diff,Matrix,linsolve,expand,nsimplify,zeros
from sympy.utilities.lambdify import lambdify
import dill as pickle
pickle.settings['recurse'] = True

def derive():
    n = 2
    m1,l1,r1,I1,m2,l2,r2,I2,g = symbols('m1 l1 r1 I1 m2 l2 r2 I2 g')
    q1,q2,q1dot,q2dot = symbols('q1 q2 q1dot q2dot')
    a1,a2 = symbols('a1 a2')

    q = Matrix([q1, q2])
    qdot = Matrix([q1dot, q2dot])

    T1 = 0.5*m1*((r1*cos(q1)*q1dot)**2 + (-r1*sin(q1)*q1dot)**2) + 0.5*I1*(q1dot**2)
    T2 = 0.5*m2*((l1*cos(q1)*q1dot+r2*cos(q1+q2)*(q1dot+q2dot))**2 + (-l1*sin(q1)*q1dot-r2*sin(q1+q2)*(q1dot+q2dot))**2) + 0.5*I2*((q1dot+q2dot)**2)
    T = T1+T2

    V = m1*g*r1*cos(q1)+m2*g*(l1*cos(q1)+r2*cos(q1+q2))

    L = T-V

    M1 = expand(diff(L,q1dot))
    M2 = expand(diff(L,q2dot))
    M = nsimplify(Matrix([[M1.coeff(q1dot),M1.coeff(q2dot)],\
                          [M2.coeff(q1dot),M2.coeff(q2dot)]]))
    
    det = simplify(M[0,0]*M[1,1]-M[0,1]**2)
    Minv = Matrix([[M[1,1],-M[0,1]],\
                   [-M[0,1],M[0,0]]])/det

    C = zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += (diff(M[i,j],q[k])+diff(M[i,k],q[j])-diff(M[k,j],q[i]))*qdot[k]/2
    C = simplify(C)

    G = Matrix([diff(V,q[i]) for i in range(n)])
    G = simplify(G)

    qddot = Minv*(Matrix([a1, a2]) - C*qdot - G)

    F = Matrix([qdot[i] for i in range(n)]+[qddot[i] for i in range(n)])
    F = simplify(F)
    # print(F)
    H = T + V
    s = Matrix([q[i] for i in range(n)]+[qdot[i] for i in range(n)])
    D_s = F.jacobian(s)
    D_a = F.jacobian(Matrix([a1, a2]))
    # print(D)

    H_lambda = lambdify([tuple([m1,l1,r1,I1,m2,l2,r2,I2,g]+[q1,q2,q1dot,q2dot])],H,'numpy',cse=True)
    F_lambda = lambdify([tuple([m1,l1,r1,I1,m2,l2,r2,I2,g]+[q1,q2,q1dot,q2dot]+[a1, a2])],F,'numpy',cse=True)
    D_lambda = lambdify([tuple([m1,l1,r1,I1,m2,l2,r2,I2,g]+[q1,q2,q1dot,q2dot]+[a1, a2])],(D_s,D_a),'numpy',cse=True)
    
    with open("./env/reacher/dynamics.p", "wb") as outf:
        pickle.dump({'H':H_lambda,'F':F_lambda,'D':D_lambda}, outf)

    print("Done")

def check():
    inertial_params = [0.1,1.0,0.5,0.1/12,0.1,1.0,0.5,0.1/12,9.8]
    s = [1,1,0,0]
    a = [0,1]

    with open("./env/reacher/dynamics.p", "rb") as inf:
        funcs = pickle.load(inf)
    
    H, F, D = funcs['H'], funcs['F'], funcs['D']        
    print(H(inertial_params+s))
    print(F(inertial_params+s+a).flatten())
    print(D(inertial_params+s+a))

if __name__ == '__main__':
    derive()
    check()
