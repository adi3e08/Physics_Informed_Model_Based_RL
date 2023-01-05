from sympy import symbols,cos,sin,simplify,diff,Matrix,linsolve,expand,nsimplify,zeros
from sympy.utilities.lambdify import lambdify
import dill as pickle
pickle.settings['recurse'] = True

def derive():
    n = 3
    m1,m2,l2,r2,I2,m3,l3,r3,I3,g = symbols('m1 m2 l2 r2 I2 m3 l3 r3 I3 g')
    q1,q2,q3,q1dot,q2dot,q3dot = symbols('q1 q2 q3 q1dot q2dot q3dot')
    a1 = symbols('a1')

    q = Matrix([q1, q2, q3])
    qdot = Matrix([q1dot, q2dot, q3dot])

    T1 = 0.5*m1*(q1dot**2)
    T2 = 0.5*m2*((q1dot+r2*cos(q2)*q2dot)**2 + (-r2*sin(q2)*q2dot)**2) + 0.5*I2*(q2dot**2)
    T3 = 0.5*m3*((q1dot+l2*cos(q2)*q2dot+r3*cos(q2+q3)*(q2dot+q3dot))**2 + (-l2*sin(q2)*q2dot-r3*sin(q2+q3)*(q2dot+q3dot))**2) + 0.5*I3*((q2dot+q3dot)**2)
    T = T1+T2+T3

    V = m2*g*r2*cos(q2)+m3*g*(l2*cos(q2)+r3*cos(q2+q3))

    L = T-V

    M1 = expand(diff(L,q1dot))
    M2 = expand(diff(L,q2dot))
    M3 = expand(diff(L,q3dot))
    M = nsimplify(Matrix([[M1.coeff(q1dot),M1.coeff(q2dot),M1.coeff(q3dot)],\
                          [M2.coeff(q1dot),M2.coeff(q2dot),M2.coeff(q3dot)],\
                          [M3.coeff(q1dot),M3.coeff(q2dot),M3.coeff(q3dot)]]))


    a11 = simplify(M[2,2] * M[1,1] - M[1,2]**2)  
    a12 = simplify(M[0,2] * M[1,2] - M[2,2] * M[0,1])  
    a13 = simplify(M[0,1] * M[1,2] - M[0,2] * M[1,1])  

    a22 = simplify(M[2,2] * M[0,0] - M[0,2]**2)  
    a23 = simplify(M[0,1] * M[0,2] - M[0,0] * M[1,2])

    a33 = simplify(M[0,0] * M[1,1] - M[0,1]**2)

    det = simplify((M[0,0] * a11) + (M[0,1] * a12) + (M[0,2] * a13))

    Minv = Matrix([[a11,a12,a13],\
                   [a12,a22,a23],\
                   [a13,a23,a33]])/det

    C = zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += (diff(M[i,j],q[k])+diff(M[i,k],q[j])-diff(M[k,j],q[i]))*qdot[k]/2
    C = simplify(C)

    G = Matrix([diff(V,q[i]) for i in range(n)])
    G = simplify(G)

    qddot = Minv*(Matrix([a1, 0, 0]) - C*qdot - G)

    F = Matrix([qdot[i] for i in range(n)]+[qddot[i] for i in range(n)])
    # F = simplify(F)
    # print(F)
    H = T + V
    s = Matrix([q[i] for i in range(n)]+[qdot[i] for i in range(n)])
    D_s = F.jacobian(s)
    D_a = F.jacobian(Matrix([a1]))
    # print(D)

    H_lambda = lambdify([tuple([m1,m2,l2,r2,I2,m3,l3,r3,I3,g]+[q1,q2,q3,q1dot,q2dot,q3dot])],H,'numpy',cse=True)
    F_lambda = lambdify([tuple([m1,m2,l2,r2,I2,m3,l3,r3,I3,g]+[q1,q2,q3,q1dot,q2dot,q3dot]+[a1])],F,'numpy',cse=True)
    D_lambda = lambdify([tuple([m1,m2,l2,r2,I2,m3,l3,r3,I3,g]+[q1,q2,q3,q1dot,q2dot,q3dot]+[a1])],(D_s,D_a),'numpy',cse=True)
    
    with open("./env/cart2pole/dynamics.p", "wb") as outf:
        pickle.dump({'H':H_lambda,'F':F_lambda,'D':D_lambda}, outf)

    print("Done")

def check():
    inertial_params = [1.0,0.1,1.0,0.5,0.1/12,0.1,1.0,0.5,0.1/12,9.8]
    s = [1,1,1,0,0,0]
    a = [1,0,0]

    with open("./env/cart2pole/dynamics.p", "rb") as inf:
        funcs = pickle.load(inf)
    
    H, F, D = funcs['H'], funcs['F'], funcs['D']        
    print(H(inertial_params+s))
    print(F(inertial_params+s+a).flatten())
    print(D(inertial_params+s+a))

if __name__ == '__main__':
    derive()
    check()