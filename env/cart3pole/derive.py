from sympy import symbols,cos,sin,simplify,diff,Matrix,linsolve,expand,nsimplify,zeros
from sympy.utilities.lambdify import lambdify
from sympy.matrices.dense import matrix_multiply_elementwise
import dill as pickle
pickle.settings['recurse'] = True

def derive():
    n = 4
    m1,m2,l2,r2,I2,m3,l3,r3,I3,m4,l4,r4,I4,g = symbols('m1 m2 l2 r2 I2 m3 l3 r3 I3 m4 l4 r4 I4 g')
    q1,q2,q3,q4,q1dot,q2dot,q3dot,q4dot = symbols('q1 q2 q3 q4 q1dot q2dot q3dot q4dot')
    a1, a4 = symbols('a1 a4')

    q = Matrix([q1, q2, q3, q4])
    qdot = Matrix([q1dot, q2dot, q3dot, q4dot])

    T1 = 0.5*m1*(q1dot**2)
    T2 = 0.5*m2*((q1dot+r2*cos(q2)*q2dot)**2 + (-r2*sin(q2)*q2dot)**2) + 0.5*I2*(q2dot**2)
    T3 = 0.5*m3*((q1dot+l2*cos(q2)*q2dot+r3*cos(q2+q3)*(q2dot+q3dot))**2 + (-l2*sin(q2)*q2dot-r3*sin(q2+q3)*(q2dot+q3dot))**2) + 0.5*I3*((q2dot+q3dot)**2)
    T4 = 0.5*m4*((q1dot+l2*cos(q2)*q2dot+l3*cos(q2+q3)*(q2dot+q3dot)+r4*cos(q2+q3+q4)*(q2dot+q3dot+q4dot))**2 +\
                 (-l2*sin(q2)*q2dot-l3*sin(q2+q3)*(q2dot+q3dot)-r4*sin(q2+q3+q4)*(q2dot+q3dot+q4dot))**2) +\
         0.5*I4*((q2dot+q3dot+q4dot)**2)
    T = T1+T2+T3+T4

    V = m2*g*r2*cos(q2)+m3*g*(l2*cos(q2)+r3*cos(q2+q3))+m4*g*(l2*cos(q2)+l3*cos(q2+q3)+r4*cos(q2+q3+q4))

    L = T-V

    M1 = expand(diff(L,q1dot))
    M2 = expand(diff(L,q2dot))
    M3 = expand(diff(L,q3dot))
    M4 = expand(diff(L,q4dot))

    M11,M12,M13,M14 = M1.coeff(q1dot),M1.coeff(q2dot),M1.coeff(q3dot),M1.coeff(q4dot)
    M22,M23,M24 = M2.coeff(q2dot),M2.coeff(q3dot),M2.coeff(q4dot)
    M33,M34 = M3.coeff(q3dot),M3.coeff(q4dot)
    M44 = M4.coeff(q4dot)

    M11,M12,M13,M14 = simplify(M11),simplify(M12),simplify(M13),simplify(M14)
    M22,M23,M24 = simplify(M22),simplify(M23),simplify(M24)
    M33,M34 = simplify(M33),simplify(M34)
    M44 = simplify(M44)

    M = Matrix([[M11,M12,M13,M14],\
                [M12,M22,M23,M24],\
                [M13,M23,M33,M34],\
                [M14,M24,M34,M44]])

    a11 = -(M24**2*M33) + 2*M23*M24*M34 - M22*M34**2 - M23**2*M44 + M22*M33*M44
    a12 = M14*M24*M33 - M14*M23*M34 - M13*M24*M34 + M12*M34**2 + M13*M23*M44 - M12*M33*M44
    a13 = -(M14*M23*M24) + M13*M24**2 + M14*M22*M34 - M12*M24*M34 - M13*M22*M44 + M12*M23*M44
    a14 = M14*M23**2 - M13*M23*M24 - M14*M22*M33 + M12*M24*M33 + M13*M22*M34 - M12*M23*M34

    a22 = -(M14**2*M33) + 2*M13*M14*M34 - M11*M34**2 - M13**2*M44 + M11*M33*M44
    a23 = M14**2*M23 - M13*M14*M24 - M12*M14*M34 + M11*M24*M34 + M12*M13*M44 - M11*M23*M44
    a24 = -(M13*M14*M23) + M13**2*M24 + M12*M14*M33 - M11*M24*M33 - M12*M13*M34 + M11*M23*M34

    a33 = -(M14**2*M22) + 2*M12*M14*M24 - M11*M24**2 - M12**2*M44 + M11*M22*M44
    a34 = M13*M14*M22 - M12*M14*M23 - M12*M13*M24 + M11*M23*M24 + M12**2*M34 - M11*M22*M34

    a44 = -(M13**2*M22) + 2*M12*M13*M23 - M11*M23**2 - M12**2*M33 + M11*M22*M33

    det = M11*a11 + M12*a12 + M13*a13 + M14*a14

    Minv = Matrix([[a11,a12,a13,a14],\
                   [a12,a22,a23,a24],\
                   [a13,a23,a33,a34],\
                   [a14,a24,a34,a44]])/det

    C = zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += (diff(M[i,j],q[k])+diff(M[i,k],q[j])-diff(M[k,j],q[i]))*qdot[k]/2
    C = simplify(C)

    G = Matrix([diff(V,q[i]) for i in range(n)])
    G = simplify(G)

    qddot = Minv*(Matrix([a1, 0, 0, a4]) - C*qdot - G)

    F = Matrix([qdot[i] for i in range(n)]+[qddot[i] for i in range(n)])
    # F = simplify(F)
    # print(F)
    H = T + V
    s = Matrix([q[i] for i in range(n)]+[qdot[i] for i in range(n)])
    D_s = F.jacobian(s)
    D_a = F.jacobian(Matrix([a1,a4]))
    # print(D)

    H_lambda = lambdify([tuple([m1,m2,l2,r2,I2,m3,l3,r3,I3,m4,l4,r4,I4,g]+[q1,q2,q3,q4,q1dot,q2dot,q3dot,q4dot])],H,'numpy',cse=True)
    F_lambda = lambdify([tuple([m1,m2,l2,r2,I2,m3,l3,r3,I3,m4,l4,r4,I4,g]+[q1,q2,q3,q4,q1dot,q2dot,q3dot,q4dot]+[a1, a4])],F,'numpy',cse=True)
    D_lambda = lambdify([tuple([m1,m2,l2,r2,I2,m3,l3,r3,I3,m4,l4,r4,I4,g]+[q1,q2,q3,q4,q1dot,q2dot,q3dot,q4dot]+[a1, a4])],(D_s,D_a),'numpy',cse=True)
    
    with open("./env/cart3pole/dynamics.p", "wb") as outf:
        pickle.dump({'H':H_lambda,'F':F_lambda,'D':D_lambda}, outf)

    print("Done")

def check():
    inertial_params = [1.0,0.1,1.0,0.5,0.1/12,0.1,1.0,0.5,0.1/12,0.1,1.0,0.5,0.1/12,9.8]
    s = [1,1,1,1,0,0,0,0]
    a = [1,0,0,1]

    with open("./env/cart3pole/dynamics.p", "rb") as inf:
        funcs = pickle.load(inf)
    
    H, F, D = funcs['H'], funcs['F'], funcs['D']        
    print(H(inertial_params+s))
    print(F(inertial_params+s+a).flatten())
    print(D(inertial_params+s+a))

if __name__ == '__main__':
    derive()
    check()
