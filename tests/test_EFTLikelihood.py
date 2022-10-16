from EFTLikelihood.EFTLikelihood import *

def test_Constant():
    c = Constant(2)
    print('Testing', type(c).__name__)
    print(c)
    print('2*2 + 2 =', (c*c + c).eval())
    print((c*c + c).ln())
    assert ((c*c + c).ln().eval() - np.log(6))<1e-18
    assert type(c+c)==Constant
    assert type(c-c)==Constant
    assert type(c*c)==Constant
    assert type(c/c)==Constant

def test_Variable():
    x = Variable('x')
    print('Testing', type(x).__name__)
    print('    function: ', end='')
    print('   ' + str(x))
    print('    ln: ', end='')
    print('  ' + str(x.ln()))
    print('    derivative: ', end='')
    print('  ' + str(x.derivative()))
    print('    f\'(ln(x)): ', end='')
    print('  ' + str(x.ln().derivative()))
    assert ((Constant(2)*x-x).ln().eval(2) - np.log(2*2 - 2))<1e-18
    assert ((Constant(2)*x-x).ln().eval(1) - np.log(2*1 - 1))<1e-18
    assert ((Constant(2)*x-x).ln().eval(2) - np.log(2*2 - 2))<1e-18
    assert type(x.eval(1)+x.eval(1))==Constant
    assert type(x.eval(1)-x.eval(1))==Constant
    assert type(x.eval(1)*x.eval(1))==Constant
    assert type(x.eval(1)/x.eval(1))==Constant

def test_Power():
    x = Power('x', 2).simplify()
    print('Testing', type(x).__name__)
    print('    function: ', end='')
    print(x)
    print('    function: ', end='')
    print('   ' + str(x))
    print('    ln: ', end='')
    print('  ' + str(x.ln()))
    p = Power('x', 2).simplify()
    print('    f(x)+f(x):', end='')
    print('   ', p+Power('x', 1))
    print('    derivative: ', end='')
    print('   ' + str(x.derivative()))
    assert (x.eval(2) - 2**2)<1e-18
    assert type(x.eval(1)+x.eval(1))==Constant
    assert type(x.eval(1)-x.eval(1))==Constant
    assert type(x.eval(1)*x.eval(1))==Constant
    assert type(x.eval(1)/x.eval(1))==Constant

def test_Polynomial():
    x = Polynomial('x', [2,1,1],  2)
    print('Testing', type(x).__name__)
    print('    function: ', end='')
    print('   ' + str(x))
    print('    ln: ', end='')
    print('  ' + str(x.ln()))
    print('    derivative: ', end='')
    print('   ' + str(x.derivative()))
    assert (x.eval(2) - (2**2 + 2**1 + 1))<1e-18
    assert type(x.eval(1)+x.eval(1))==Constant
    assert type(x.eval(1)-x.eval(1))==Constant
    assert type(x.eval(1)*x.eval(1))==Constant
    assert type(x.eval(1)/x.eval(1))==Constant

def test_Expo():
    x = Expo(Prod(Constant(-1), Variable('x')))
    print('Testing', type(x).__name__)
    print('    function: ', end='')
    print('   ' + str(x))
    print('    ln: ', end='')
    print('  ' + str(x.ln()))
    print('    derivative: ', end='')
    print('   ' + str(x.derivative()))
    assert (x.eval(2) - np.exp(-2))<1e-18
    assert (x.derivative().eval(2) - -np.exp(-2))<1e-18
    assert type(x.eval(1)+x.eval(1))==Constant
    assert type(x.eval(1)-x.eval(1))==Constant
    assert type(x.eval(1)*x.eval(1))==Constant
    assert type(x.eval(1)/x.eval(1))==Constant

def test_Poisson():
    x = Poisson('x', 'k')
    print('Testing', type(x).__name__)
    print('    function: ', end='')
    print('   ' + str(x))
    print('    ln: ', end='')
    print('  ' + str(x.ln()))
    print('    derivative: ', end='')
    print('   ' + str(x.derivative()))
    print('    f\'(ln(x)): ', end='')
    print('  ' + str(x.ln().derivative()))
    assert (x.eval(1,2) - (1**2 * np.exp(-1) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    assert (x.derivative().eval(1,2) - np.exp(-1)/2)<1e-18 # float has precision of 1e-18
    assert (x.derivative().eval(1,10) - (np.exp(-1)*(10-1)*(1**(10-1)/np.math.factorial(10))))<1e-18 # float has precision of 1e-18
    assert (x.ln().derivative().eval(2,10) - (10/2-1))<1e-18 # float has precision of 1e-18

def test_EFTPoisson():
    x = EFTPoisson('x', [1, 1, 1], 'k')
    print('Testing', type(x).__name__)
    print('    function: ', end='')
    print('   ' + str(x))
    print('    ln: ', end='')
    print('  ' + str(x.ln()))
    print('    derivative: ', end='')
    print('   ' + str(x.derivative()))
    print('    f\'(ln(x)): ', end='')
    print('  ' + str(x.ln().derivative()))
    assert (x.eval(1,2) - ((1**2 + 1**1 + 1)**2 * np.exp(-1*(1**2 + 1**1 +1)) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    '''
    ln:   ((k * ln((((x^2) + x) + 1))) + ((-1 * (((x^2) + x) + 1)) - ln(k)))
        derivative:    (((k * ((((x^2) + x) + 1)^(k - 1))) * (e^(-1 * (((x^2) + x) + 1)) / k!)) + (((((x^2) + x) + 1)^k) * ((((0 * (((x^2) + x) + 1)) + (-1 * (((2 * x) + 1) + 0))) * e^(-1 * (((x^2) + x) + 1))) / k!)))
    f'(ln(x)):   (((0 * ln((((x^2) + x) + 1))) + (k * ((((2 * x) + 1) + 0) / (((x^2) + x) + 1)))) + (((0 * (((x^2) + x) + 1)) + (-1 * (((2 * x) + 1) + 0))) - 0))
    '''
    assert (x.ln().eval(1,2) - (2 * np.log(1**2 + 1**1 + 1) - (1**2 + 1**1 + 1) - np.log(2)))<1e-18 # float has precision of 1e-18
    print(x.ln().derivative().eval(1,2), (2 * (2*1 + 1) * 1./(1**2 + 1**1 + 1) - (2*1 + 1)))
    assert (x.ln().derivative().eval(1,2) - (2 * (2*1 + 1) * 1./(1**2 + 1**1 + 1) - (2*1 + 1)))<1e-18 # float has precision of 1e-18
