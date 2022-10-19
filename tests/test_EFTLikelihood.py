from eft_likelihood.eft_likelihood import *

def test_Constant():
    const_test_test = Constant(2)
    print('Testing', type(const_test_test).__name__)
    print('    const: ', end='')
    print('   ' + str(const_test_test))
    print('    2*2 + 2 =', (const_test_test*const_test_test + const_test_test).eval())
    print('    ln: ', end='')
    print('   ' + str((const_test_test*const_test_test + const_test_test).ln()))
    assert ((const_test_test*const_test_test + const_test_test).ln().eval() - np.log(6))<1e-18
    assert type(const_test_test+const_test_test)==Constant
    assert type(const_test_test-const_test_test)==Constant
    assert type(const_test_test*const_test_test)==Constant
    assert type(const_test_test/const_test_test)==Constant
    assert ((const_test_test*const_test_test + const_test_test).eval().value() - 6)<1e-18
    assert ((const_test_test*const_test_test + const_test_test).ln().eval().value() - np.log(6))<1e-18

def test_Variable():
    var_test_test = Variable('x')
    print('Testing', type(var_test_test).__name__)
    print('    function: ', end='')
    print('   ' + str(var_test_test))
    print('    ln: ', end='')
    print('  ' + str(var_test_test.ln()))
    print('    derivative: ', end='')
    print('  ' + str(var_test_test.derivative()))
    print('    f\'(ln(var_test_test)): ', end='')
    print('  ' + str(var_test_test.ln().derivative()))
    assert ((Constant(2)*var_test_test-var_test_test).ln().eval(2) - np.log(2*2 - 2))<1e-18
    assert ((Constant(2)*var_test_test-var_test_test).ln().eval(1) - np.log(2*1 - 1))<1e-18
    assert ((Constant(2)*var_test_test-var_test_test).ln().eval(2) - np.log(2*2 - 2))<1e-18
    assert type(var_test_test.eval(1)+var_test_test.eval(1))==Constant
    assert type(var_test_test.eval(1)-var_test_test.eval(1))==Constant
    assert type(var_test_test.eval(1)*var_test_test.eval(1))==Constant
    assert type(var_test_test.eval(1)/var_test_test.eval(1))==Constant
    var_y = Variable('y')
    assert var_test_test.derivative('y').value() == 0
    assert var_y.derivative('x').value() == 0
    assert var_y.derivative('y').value() == 1

def test_Power():
    pow_test_test = Power('x', 2)
    print('Testing', type(pow_test_test).__name__)
    print('    function: ', end='')
    print(pow_test_test)
    print('    function: ', end='')
    print('   ' + str(pow_test_test))
    print('    ln: ', end='')
    print('  ' + str(pow_test_test.ln()))
    other_pow_test_test = Power('x', 2)
    print('    f(pow_test_test)+f(pow_test_test):', end='')
    print('   ', other_pow_test_test+Power('x', 1))
    print('    derivative: ', end='')
    print('   ' + str(pow_test_test.derivative()))
    assert (pow_test_test.eval(2) - 2**2)<1e-18
    assert type(pow_test_test.eval(1)+pow_test_test.eval(1))==Constant
    assert type(pow_test_test.eval(1)-pow_test_test.eval(1))==Constant
    assert type(pow_test_test.eval(1)*pow_test_test.eval(1))==Constant
    assert type(pow_test_test.eval(1)/pow_test_test.eval(1))==Constant

def test_Polynomial():
    poly_test_test = Polynomial('x', [2,1,1],  2)
    print('Testing', type(poly_test_test).__name__)
    print('    function: ', end='')
    print('   ' + str(poly_test_test))
    print('    ln: ', end='')
    print('  ' + str(poly_test_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(poly_test_test.derivative()))
    assert (poly_test_test.eval(2) - (2 * 2**2 + 2**1 + 1))<1e-18
    assert type(poly_test_test.eval(1)+poly_test_test.eval(1))==Constant
    assert type(poly_test_test.eval(1)-poly_test_test.eval(1))==Constant
    assert type(poly_test_test.eval(1)*poly_test_test.eval(1))==Constant
    assert type(poly_test_test.eval(1)/poly_test_test.eval(1))==Constant
    print(poly_test_test.derivative('x'))
    assert (poly_test_test.derivative().eval(2) - (2 * 2 * 2**1 + 1))<1e-18
    assert poly_test_test.derivative('y').eval(2).value() == 0

def test_Expo():
    expo_test_test = Expo(Prod(Constant(-1), Variable('x')))
    print('Testing', type(expo_test_test).__name__)
    print('    function: ', end='')
    print('   ' + str(expo_test_test))
    print('    ln: ', end='')
    print('  ' + str(expo_test_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(expo_test_test.derivative()))
    assert (expo_test_test.eval(2) - np.exp(-2))<1e-18
    assert (expo_test_test.derivative().eval(2) - -np.exp(-2))<1e-18
    assert type(expo_test_test.eval(1)+expo_test_test.eval(1))==Constant
    assert type(expo_test_test.eval(1)-expo_test_test.eval(1))==Constant
    assert type(expo_test_test.eval(1)*expo_test_test.eval(1))==Constant
    assert type(expo_test_test.eval(1)/expo_test_test.eval(1))==Constant

def test_Poisson():
    pois_test_test = Poisson('x', 'k')
    print('Testing', type(pois_test_test).__name__)
    print('    function: ', end='')
    print('   ' + str(pois_test_test))
    print('    ln: ', end='')
    print('  ' + str(pois_test_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(pois_test_test.derivative()))
    print('    f\'(ln(pois_test_test)): ', end='')
    print('  ' + str(pois_test_test.ln().derivative()))
    assert (pois_test_test.eval(1,2) - (1**2 * np.exp(-1) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    assert (pois_test_test.derivative().eval(1,2) - (np.exp(-1) - np.exp(-1)/2))<1e-18 # float has precision of 1e-18
    assert (pois_test_test.derivative().eval(1,10) - (np.exp(-1)*(10-1)*(1**(10-1)/np.math.factorial(10))))<1e-18 # float has precision of 1e-18
    assert (pois_test_test.ln().derivative().eval(2,10) - (10/2-1))<1e-18 # float has precision of 1e-18

def test_EFTPoisson():
    poly_test_test = Polynomial('x', [1,1,1],  2)
    eft_pois_test_test = EFTPoisson(poly_test_test, param='k')
    print('Testing', type(eft_pois_test_test).__name__)
    print('    function: ', end='')
    print('   ' + str(eft_pois_test_test))
    print('    ln: ', end='')
    print('  ' + str(eft_pois_test_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(eft_pois_test_test.derivative()))
    print('    f\'(ln(x)): ', end='')
    print('  ' + str(eft_pois_test_test.ln().derivative()))
    assert (eft_pois_test_test.eval(1,2) - ((1**2 + 1**1 + 1)**2 * np.exp(-1*(1**2 + 1**1 +1)) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    '''
    ln:   ((k * ln((((eft_pois_test_test^2) + eft_pois_test_test) + 1))) + ((-1 * (((eft_pois_test_test^2) + eft_pois_test_test) + 1)) - ln(k)))
        derivative:    (((k * ((((eft_pois_test_test^2) + eft_pois_test_test) + 1)^(k - 1))) * (e^(-1 * (((eft_pois_test_test^2) + eft_pois_test_test) + 1)) / k!)) + (((((eft_pois_test_test^2) + eft_pois_test_test) + 1)^k) * ((((0 * (((eft_pois_test_test^2) + eft_pois_test_test) + 1)) + (-1 * (((2 * eft_pois_test_test) + 1) + 0))) * e^(-1 * (((eft_pois_test_test^2) + eft_pois_test_test) + 1))) / k!)))
    f'(ln(eft_pois_test_test)):   (((0 * ln((((eft_pois_test_test^2) + eft_pois_test_test) + 1))) + (k * ((((2 * eft_pois_test_test) + 1) + 0) / (((eft_pois_test_test^2) + eft_pois_test_test) + 1)))) + (((0 * (((eft_pois_test_test^2) + eft_pois_test_test) + 1)) + (-1 * (((2 * eft_pois_test_test) + 1) + 0))) - 0))
    '''
    assert (eft_pois_test_test.ln().eval(1,2) - (2 * np.log(1**2 + 1**1 + 1) - (1**2 + 1**1 + 1) - np.log(2)))<1e-18 # float has precision of 1e-18
    assert (eft_pois_test_test.ln().derivative().eval(1,2) - (2 * (2*1 + 1) * 1./(1**2 + 1**1 + 1) - (2*1 + 1)))<1e-18 # float has precision of 1e-18
