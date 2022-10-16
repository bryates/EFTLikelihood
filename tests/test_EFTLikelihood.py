from EFTLikelihood.EFTLikelihood import *

def test_constant():
    const_test = constant(2)
    print('Testing', type(const_test).__name__)
    print(const_test)
    print('2*2 + 2 =', (const_test*const_test + const_test).eval())
    print((const_test*const_test + const_test).ln())
    assert ((const_test*const_test + const_test).ln().eval() - np.log(6))<1e-18
    assert type(const_test+const_test)==constant
    assert type(const_test-const_test)==constant
    assert type(const_test*const_test)==constant
    assert type(const_test/const_test)==constant

def test_variable():
    var_test = variable('var_test')
    print('Testing', type(var_test).__name__)
    print('    function: ', end='')
    print('   ' + str(var_test))
    print('    ln: ', end='')
    print('  ' + str(var_test.ln()))
    print('    derivative: ', end='')
    print('  ' + str(var_test.derivative()))
    print('    f\'(ln(var_test)): ', end='')
    print('  ' + str(var_test.ln().derivative()))
    assert ((constant(2)*var_test-var_test).ln().eval(2) - np.log(2*2 - 2))<1e-18
    assert ((constant(2)*var_test-var_test).ln().eval(1) - np.log(2*1 - 1))<1e-18
    assert ((constant(2)*var_test-var_test).ln().eval(2) - np.log(2*2 - 2))<1e-18
    assert type(var_test.eval(1)+var_test.eval(1))==constant
    assert type(var_test.eval(1)-var_test.eval(1))==constant
    assert type(var_test.eval(1)*var_test.eval(1))==constant
    assert type(var_test.eval(1)/var_test.eval(1))==constant

def test_power():
    pow_test = power('pow_test', 2).simplify()
    print('Testing', type(pow_test).__name__)
    print('    function: ', end='')
    print(pow_test)
    print('    function: ', end='')
    print('   ' + str(pow_test))
    print('    ln: ', end='')
    print('  ' + str(pow_test.ln()))
    other_pow_test = power('pow_test', 2).simplify()
    print('    f(pow_test)+f(pow_test):', end='')
    print('   ', other_pow_test+power('pow_test', 1))
    print('    derivative: ', end='')
    print('   ' + str(pow_test.derivative()))
    assert (pow_test.eval(2) - 2**2)<1e-18
    assert type(pow_test.eval(1)+pow_test.eval(1))==constant
    assert type(pow_test.eval(1)-pow_test.eval(1))==constant
    assert type(pow_test.eval(1)*pow_test.eval(1))==constant
    assert type(pow_test.eval(1)/pow_test.eval(1))==constant

def test_polynomial():
    poly_test = polynomial('poly_test', [2,1,1],  2)
    print('Testing', type(poly_test).__name__)
    print('    function: ', end='')
    print('   ' + str(poly_test))
    print('    ln: ', end='')
    print('  ' + str(poly_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(poly_test.derivative()))
    assert (poly_test.eval(2) - (2**2 + 2**1 + 1))<1e-18
    assert type(poly_test.eval(1)+poly_test.eval(1))==constant
    assert type(poly_test.eval(1)-poly_test.eval(1))==constant
    assert type(poly_test.eval(1)*poly_test.eval(1))==constant
    assert type(poly_test.eval(1)/poly_test.eval(1))==constant

def test_expo():
    expo_test = expo(prod(constant(-1), variable('expo_test')))
    print('Testing', type(expo_test).__name__)
    print('    function: ', end='')
    print('   ' + str(expo_test))
    print('    ln: ', end='')
    print('  ' + str(expo_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(expo_test.derivative()))
    assert (expo_test.eval(2) - np.exp(-2))<1e-18
    assert (expo_test.derivative().eval(2) - -np.exp(-2))<1e-18
    assert type(expo_test.eval(1)+expo_test.eval(1))==constant
    assert type(expo_test.eval(1)-expo_test.eval(1))==constant
    assert type(expo_test.eval(1)*expo_test.eval(1))==constant
    assert type(expo_test.eval(1)/expo_test.eval(1))==constant

def test_poisson():
    pois_test = poisson('pois_test', 'k')
    print('Testing', type(pois_test).__name__)
    print('    function: ', end='')
    print('   ' + str(pois_test))
    print('    ln: ', end='')
    print('  ' + str(pois_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(pois_test.derivative()))
    print('    f\'(ln(pois_test)): ', end='')
    print('  ' + str(pois_test.ln().derivative()))
    assert (pois_test.eval(1,2) - (1**2 * np.exp(-1) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    assert (pois_test.derivative().eval(1,2) - np.exp(-1)/2)<1e-18 # float has precision of 1e-18
    assert (pois_test.derivative().eval(1,10) - (np.exp(-1)*(10-1)*(1**(10-1)/np.math.factorial(10))))<1e-18 # float has precision of 1e-18
    assert (pois_test.ln().derivative().eval(2,10) - (10/2-1))<1e-18 # float has precision of 1e-18

def test_eft_poisson():
    eft_pois_test = eft_poisson('eft_pois_test', [1, 1, 1], 'k')
    print('Testing', type(eft_pois_test).__name__)
    print('    function: ', end='')
    print('   ' + str(eft_pois_test))
    print('    ln: ', end='')
    print('  ' + str(eft_pois_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(eft_pois_test.derivative()))
    print('    f\'(ln(eft_pois_test)): ', end='')
    print('  ' + str(eft_pois_test.ln().derivative()))
    assert (eft_pois_test.eval(1,2) - ((1**2 + 1**1 + 1)**2 * np.exp(-1*(1**2 + 1**1 +1)) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    '''
    ln:   ((k * ln((((eft_pois_test^2) + eft_pois_test) + 1))) + ((-1 * (((eft_pois_test^2) + eft_pois_test) + 1)) - ln(k)))
        derivative:    (((k * ((((eft_pois_test^2) + eft_pois_test) + 1)^(k - 1))) * (e^(-1 * (((eft_pois_test^2) + eft_pois_test) + 1)) / k!)) + (((((eft_pois_test^2) + eft_pois_test) + 1)^k) * ((((0 * (((eft_pois_test^2) + eft_pois_test) + 1)) + (-1 * (((2 * eft_pois_test) + 1) + 0))) * e^(-1 * (((eft_pois_test^2) + eft_pois_test) + 1))) / k!)))
    f'(ln(eft_pois_test)):   (((0 * ln((((eft_pois_test^2) + eft_pois_test) + 1))) + (k * ((((2 * eft_pois_test) + 1) + 0) / (((eft_pois_test^2) + eft_pois_test) + 1)))) + (((0 * (((eft_pois_test^2) + eft_pois_test) + 1)) + (-1 * (((2 * eft_pois_test) + 1) + 0))) - 0))
    '''
    assert (eft_pois_test.ln().eval(1,2) - (2 * np.log(1**2 + 1**1 + 1) - (1**2 + 1**1 + 1) - np.log(2)))<1e-18 # float has precision of 1e-18
    print(eft_pois_test.ln().derivative().eval(1,2), (2 * (2*1 + 1) * 1./(1**2 + 1**1 + 1) - (2*1 + 1)))
    assert (eft_pois_test.ln().derivative().eval(1,2) - (2 * (2*1 + 1) * 1./(1**2 + 1**1 + 1) - (2*1 + 1)))<1e-18 # float has precision of 1e-18
