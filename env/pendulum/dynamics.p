???      }?(?H??
dill._dill??_create_function???(h?_create_code???(KK K KKKCC6| \}}}}}}}d| |d  || | t |?  S ?NG??      K???cos???(?	_Dummy_24??m1??l1??r1??I1??g??q1??q1dot?t??<lambdifygenerated-1>??_lambdifygenerated?KC ?))t?R?}??__name__?NshNNt?R?}?}?(?__doc__?X  Created with lambdify. Signature:

func(arg_0)

Expression:

0.5*I1*q1dot**2 + g*m1*r1*cos(q1)

Source code:

def _lambdifygenerated(_Dummy_24):
    [m1, l1, r1, I1, g, q1, q1dot] = _Dummy_24
    return 0.5*I1*q1dot**2 + g*m1*r1*cos(q1)


Imported modules:

??__annotations__?}?u??bh?cos??numpy.core._multiarray_umath??cos???s0?F?h(h(KK K K	KKCC<| \}}}}}}}}t |g||| | t|?  | gg?S ?N???array??sin???(?	_Dummy_25?hhhhhhh?a1?t??<lambdifygenerated-2>?hKC ?))t?R?}?hNshNNt?R?}?}?(hX  Created with lambdify. Signature:

func(arg_0)

Expression:

Matrix([[q1dot], [(a1 + g*m1*r1*sin(q1))/I1]])

Source code:

def _lambdifygenerated(_Dummy_25):
    [m1, l1, r1, I1, g, q1, q1dot, a1] = _Dummy_25
    return array([[q1dot], [(a1 + g*m1*r1*sin(q1))/I1]])


Imported modules:

?h!}?u??bh5(?array??numpy??array????sin?h%?sin???u0?D?h(h(KK K K
KKCCT| \}}}}}}}}|d }	t ddg|| | |	 t|? dgg?t dg|	gg?fS ?(NG??      K Kt?h+h	??(?	_Dummy_26?hhhhhhhh/?x0?t??<lambdifygenerated-3>?hKC ?))t?R?}?hNshNNt?R?}?}?(hXa  Created with lambdify. Signature:

func(arg_0)

Expression:

(Matrix([ [                 0, 1], [g*m1*r1*cos(q1)/I1, 0]]), Matrix([ [...

Source code:

def _lambdifygenerated(_Dummy_26):
    [m1, l1, r1, I1, g, q1, q1dot, a1] = _Dummy_26
    x0 = I1**(-1.0)
    return (array([[0, 1], [g*m1*r1*x0*cos(q1), 0]]), array([[0], [x0]]))


Imported modules:

?h!}?u??bhO(?cos?h'?array?h@u0?V?h(h(KK K KKKCC| \}}}}}}}t |gg?S ?N??h+??(?	_Dummy_27?hhhhhhht??<lambdifygenerated-4>?hKC ?))t?R?}?hNshNNt?R?}?}?(h??Created with lambdify. Signature:

func(arg_0)

Expression:

Matrix([[q1dot]])

Source code:

def _lambdifygenerated(_Dummy_27):
    [m1, l1, r1, I1, g, q1, q1dot] = _Dummy_27
    return array([[q1dot]])


Imported modules:

?h!}?u??bhc?array?h@s0u.