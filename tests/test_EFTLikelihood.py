from eft_likelihood.eft_likelihood import *

def test_Constant():
    const_test = Constant(2)
    print('Testing', type(const_test).__name__)
    print('    const: ', end='')
    print('   ' + str(const_test))
    print('    2*2 + 2 =', (const_test*const_test + const_test).eval())
    print('    ln: ', end='')
    print('   ' + str((const_test*const_test + const_test).ln()))
    assert ((const_test*const_test + const_test).ln().eval() - np.log(6))<1e-18
    assert type(const_test+const_test)==Constant
    assert type(const_test-const_test)==Constant
    assert type(const_test*const_test)==Constant
    assert type(const_test/const_test)==Constant
    assert ((const_test*const_test + const_test).eval().value() - 6)<1e-18
    assert ((const_test*const_test + const_test).ln().eval().value() - np.log(6))<1e-18
    assert(Pi().eval() - np.math.pi)<1e-18
    assert(LogPi().eval() - np.log(np.math.pi))<1e-18

def test_Variable():
    var_test = Variable('x')
    print('Testing', type(var_test).__name__)
    print('    function: ', end='')
    print('   ' + str(var_test))
    print('    ln: ', end='')
    print('  ' + str(var_test.ln()))
    print('    derivative: ', end='')
    print('  ' + str(var_test.derivative()))
    print('    f\'(ln(var_test)): ', end='')
    print('  ' + str(var_test.ln().derivative()))
    assert ((Constant(2)*var_test-var_test).ln().eval(x_in=2) - np.log(2*2 - 2))<1e-18
    assert ((Constant(2)*var_test-var_test).ln().eval(x_in=1) - np.log(2*1 - 1))<1e-18
    assert ((Constant(2)*var_test-var_test).ln().eval(x_in=2) - np.log(2*2 - 2))<1e-18
    assert type(var_test.eval(x_in=1)+var_test.eval(x_in=1))==Constant
    assert type(var_test.eval(x_in=1)-var_test.eval(x_in=1))==Constant
    assert type(var_test.eval(x_in=1)*var_test.eval(x_in=1))==Constant
    assert type(var_test.eval(x_in=1)/var_test.eval(x_in=1))==Constant
    var_y = Variable('y')
    assert var_test.derivative('y').value() == 0
    assert var_y.derivative('x').value() == 0
    assert var_y.derivative('y').value() == 1

def test_Power():
    pow_test = Power('x', 2)
    print('Testing', type(pow_test).__name__)
    print('    function: ', end='')
    print(pow_test)
    print('    function: ', end='')
    print('   ' + str(pow_test))
    print('    ln: ', end='')
    print('  ' + str(pow_test.ln()))
    other_pow_test = Power('x', 2)
    print('    f(pow_test)+f(pow_test):', end='')
    print('   ', other_pow_test+Power('x', 1))
    print('    derivative: ', end='')
    print('   ' + str(pow_test.derivative()))
    assert (pow_test.eval(x_in=2) - 2**2)<1e-18
    assert type(pow_test.eval(x_in=1)+pow_test.eval(x_in=1))==Constant
    assert type(pow_test.eval(x_in=1)-pow_test.eval(x_in=1))==Constant
    assert type(pow_test.eval(x_in=1)*pow_test.eval(x_in=1))==Constant
    assert type(pow_test.eval(x_in=1)/pow_test.eval(x_in=1))==Constant

def test_Polynomial():
    poly_test = Polynomial('x', [2,1,1],  2)
    print('Testing', type(poly_test).__name__)
    print('    function: ', end='')
    print('   ' + str(poly_test))
    print('    ln: ', end='')
    print('  ' + str(poly_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(poly_test.derivative()))
    assert (poly_test.eval(x_in=2) - (2 * 2**2 + 2**1 + 1))<1e-18
    assert type(poly_test.eval(x_in=1)+poly_test.eval(x_in=1))==Constant
    assert type(poly_test.eval(x_in=1)-poly_test.eval(x_in=1))==Constant
    assert type(poly_test.eval(x_in=1)*poly_test.eval(x_in=1))==Constant
    assert type(poly_test.eval(x_in=1)/poly_test.eval(x_in=1))==Constant
    print(poly_test.derivative('x'))
    assert (poly_test.derivative().eval(x_in=2) - (2 * 2 * 2**1 + 1))<1e-18
    assert poly_test.derivative('y').eval(x_in=2).value() == 0

def test_Expo():
    expo_test = Expo(Prod(Constant(-1), Variable('x')))
    print('Testing', type(expo_test).__name__)
    print('    function: ', end='')
    print('   ' + str(expo_test))
    print('    ln: ', end='')
    print('  ' + str(expo_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(expo_test.derivative()))
    assert (expo_test.eval(x_in=2) - np.exp(-2))<1e-18
    assert (expo_test.derivative().eval(x_in=2) - -np.exp(-2))<1e-18
    assert type(expo_test.eval(x_in=1)+expo_test.eval(x_in=1))==Constant
    assert type(expo_test.eval(x_in=1)-expo_test.eval(x_in=1))==Constant
    assert type(expo_test.eval(x_in=1)*expo_test.eval(x_in=1))==Constant
    assert type(expo_test.eval(x_in=1)/expo_test.eval(x_in=1))==Constant

def test_Poisson():
    pois_test = Poisson('x', 'k')
    print('Testing', type(pois_test).__name__)
    print('    function: ', end='')
    print('   ' + str(pois_test))
    print('    ln: ', end='')
    print('  ' + str(pois_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(pois_test.derivative()))
    print('    f\'(ln(pois_test)): ', end='')
    print('  ' + str(pois_test.ln().derivative()))
    assert (pois_test.eval(x_in=1, k_in=2) - (1**2 * np.exp(-1) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    assert (pois_test.derivative().eval(x_in=1, k_in=2) - (np.exp(-1) - np.exp(-1)/2))<1e-18 # float has precision of 1e-18
    assert (pois_test.derivative().eval(x_in=1, k_in=10) - (np.exp(-1)*(10-1)*(1**(10-1)/np.math.factorial(10))))<1e-18 # float has precision of 1e-18
    assert (pois_test.ln().derivative().eval(x_in=2, k_in=10) - (10/2-1))<1e-18 # float has precision of 1e-18

def test_EFTPoisson():
    poly_test = Polynomial('x', [1,1,1],  2)
    eft_pois_test = EFTPoisson(poly_test, param='k')
    print('Testing', type(eft_pois_test).__name__)
    print('    function: ', end='')
    print('   ' + str(eft_pois_test))
    print('    ln: ', end='')
    print('  ' + str(eft_pois_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(eft_pois_test.derivative()))
    print('    f\'(ln(x)): ', end='')
    print('  ' + str(eft_pois_test.ln().derivative()))
    assert (eft_pois_test.eval(x_in=1,k_in=2) - ((1**2 + 1**1 + 1)**2 * np.exp(-1*(1**2 + 1**1 +1)) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    assert (eft_pois_test.ln().eval(x_in=1,k_in=2) - (2 * np.log(1**2 + 1**1 + 1) - (1**2 + 1**1 + 1) - np.log(2)))<1e-18 # float has precision of 1e-18
    assert (eft_pois_test.ln().derivative().eval(x_in=1,k_in=2) - (2 * (2*1 + 1) * 1./(1**2 + 1**1 + 1) - (2*1 + 1)))<1e-18 # float has precision of 1e-18

def test_LogNormal():
    log_normal_test = LogNormal('x', 'u', 's')
    print('Testing', type(log_normal_test).__name__)
    print('    function: ', end='')
    print('   ' + str(log_normal_test))
    print('    ln: ', end='')
    print('  ' + str(log_normal_test.ln()))
    print('    f\'(x): ', end='')
    print('   ' + str(log_normal_test.derivative('x')))
    print('    f\'(mu): ', end='')
    print('   ' + str(log_normal_test.derivative('u')))
    print('    f\'(sigma): ', end='')
    print('   ' + str(log_normal_test.derivative('s')))
    print('    f\'(ln(x)): ', end='')
    print('  ' + str(log_normal_test.ln().derivative('x')))
    print('    f\'(ln(mu)): ', end='')
    print('  ' + str(log_normal_test.ln().derivative('u')))
    print('    f\'(ln(sigma)): ', end='')
    print('  ' + str(log_normal_test.ln().derivative('s')))
    assert (log_normal_test.eval(x_in=100, u_in=100, s_in=1.05) - 0.0817669) < 1e-7
    assert (log_normal_test.ln().eval(x_in=100, u_in=100, s_in=1.05) - -2.50388) < 1e-5
