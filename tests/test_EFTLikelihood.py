from EFTLikelihood.EFTLikelihood import *

def test_Constant():
    const_test = Constant(2)
    print('Testing', type(const_test).__name__)
    print(const_test)
    print('2*2 + 2 =', (const_test*const_test + const_test).eval())
    print((const_test*const_test + const_test).ln())
    assert ((const_test*const_test + const_test).ln().eval() - np.log(6))<1e-18
    assert type(const_test+const_test)==Constant
    assert type(const_test-const_test)==Constant
    assert type(const_test*const_test)==Constant
    assert type(const_test/const_test)==Constant

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
    assert ((Constant(2)*var_test-var_test).ln().eval(2) - np.log(2*2 - 2))<1e-18
    assert ((Constant(2)*var_test-var_test).ln().eval(1) - np.log(2*1 - 1))<1e-18
    assert ((Constant(2)*var_test-var_test).ln().eval(2) - np.log(2*2 - 2))<1e-18
    assert type(var_test.eval(1)+var_test.eval(1))==Constant
    assert type(var_test.eval(1)-var_test.eval(1))==Constant
    assert type(var_test.eval(1)*var_test.eval(1))==Constant
    assert type(var_test.eval(1)/var_test.eval(1))==Constant
    var_y = Variable('y')
    assert var_test.derivative('y').value() == 0
    assert var_y.derivative('x').value() == 0
    assert var_y.derivative('y').value() == 1

def test_Power():
    Pow_test = Power('x', 2).simplify()
    print('Testing', type(Pow_test).__name__)
    print('    function: ', end='')
    print(Pow_test)
    print('    function: ', end='')
    print('   ' + str(Pow_test))
    print('    ln: ', end='')
    print('  ' + str(Pow_test.ln()))
    other_Pow_test = Power('x', 2).simplify()
    print('    f(Pow_test)+f(Pow_test):', end='')
    print('   ', other_Pow_test+Power('x', 1))
    print('    derivative: ', end='')
    print('   ' + str(Pow_test.derivative()))
    assert (Pow_test.eval(2) - 2**2)<1e-18
    assert type(Pow_test.eval(1)+Pow_test.eval(1))==Constant
    assert type(Pow_test.eval(1)-Pow_test.eval(1))==Constant
    assert type(Pow_test.eval(1)*Pow_test.eval(1))==Constant
    assert type(Pow_test.eval(1)/Pow_test.eval(1))==Constant

def test_Polynomial():
    Poly_test = Polynomial('x', [2,1,1],  2)
    print('Testing', type(Poly_test).__name__)
    print('    function: ', end='')
    print('   ' + str(Poly_test))
    print('    ln: ', end='')
    print('  ' + str(Poly_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(Poly_test.derivative()))
    assert (Poly_test.eval(2) - (2 * 2**2 + 2**1 + 1))<1e-18
    assert type(Poly_test.eval(1)+Poly_test.eval(1))==Constant
    assert type(Poly_test.eval(1)-Poly_test.eval(1))==Constant
    assert type(Poly_test.eval(1)*Poly_test.eval(1))==Constant
    assert type(Poly_test.eval(1)/Poly_test.eval(1))==Constant
    print(Poly_test.derivative())
    assert (Poly_test.derivative().eval(2) - (2 * 2 * 2**1 + 2))<1e-18

def test_Expo():
    Expo_test = Expo(Prod(Constant(-1), Variable('x')))
    print('Testing', type(Expo_test).__name__)
    print('    function: ', end='')
    print('   ' + str(Expo_test))
    print('    ln: ', end='')
    print('  ' + str(Expo_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(Expo_test.derivative()))
    assert (Expo_test.eval(2) - np.exp(-2))<1e-18
    assert (Expo_test.derivative().eval(2) - -np.exp(-2))<1e-18
    assert type(Expo_test.eval(1)+Expo_test.eval(1))==Constant
    assert type(Expo_test.eval(1)-Expo_test.eval(1))==Constant
    assert type(Expo_test.eval(1)*Expo_test.eval(1))==Constant
    assert type(Expo_test.eval(1)/Expo_test.eval(1))==Constant

def test_Poisson():
    Pois_test = Poisson('x', 'k')
    print('Testing', type(Pois_test).__name__)
    print('    function: ', end='')
    print('   ' + str(Pois_test))
    print('    ln: ', end='')
    print('  ' + str(Pois_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(Pois_test.derivative()))
    print('    f\'(ln(Pois_test)): ', end='')
    print('  ' + str(Pois_test.ln().derivative()))
    assert (Pois_test.eval(1,2) - (1**2 * np.exp(-1) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    assert (Pois_test.derivative().eval(1,2) - np.exp(-1)/2)<1e-18 # float has precision of 1e-18
    assert (Pois_test.derivative().eval(1,10) - (np.exp(-1)*(10-1)*(1**(10-1)/np.math.factorial(10))))<1e-18 # float has precision of 1e-18
    assert (Pois_test.ln().derivative().eval(2,10) - (10/2-1))<1e-18 # float has precision of 1e-18

def test_EFTPoisson():
    eft_Pois_test = EFTPoisson('x', consts=[1, 1, 1], param='k')
    print('Testing', type(eft_Pois_test).__name__)
    print('    function: ', end='')
    print('   ' + str(eft_Pois_test))
    print('    ln: ', end='')
    print('  ' + str(eft_Pois_test.ln()))
    print('    derivative: ', end='')
    print('   ' + str(eft_Pois_test.derivative()))
    print('    f\'(ln(x)): ', end='')
    print('  ' + str(eft_Pois_test.ln().derivative()))
    assert (eft_Pois_test.eval(1,2) - ((1**2 + 1**1 + 1)**2 * np.exp(-1*(1**2 + 1**1 +1)) / np.math.factorial(2)))<1e-18 # float has precision of 1e-18
    '''
    ln:   ((k * ln((((eft_Pois_test^2) + eft_Pois_test) + 1))) + ((-1 * (((eft_Pois_test^2) + eft_Pois_test) + 1)) - ln(k)))
        derivative:    (((k * ((((eft_Pois_test^2) + eft_Pois_test) + 1)^(k - 1))) * (e^(-1 * (((eft_Pois_test^2) + eft_Pois_test) + 1)) / k!)) + (((((eft_Pois_test^2) + eft_Pois_test) + 1)^k) * ((((0 * (((eft_Pois_test^2) + eft_Pois_test) + 1)) + (-1 * (((2 * eft_Pois_test) + 1) + 0))) * e^(-1 * (((eft_Pois_test^2) + eft_Pois_test) + 1))) / k!)))
    f'(ln(eft_Pois_test)):   (((0 * ln((((eft_Pois_test^2) + eft_Pois_test) + 1))) + (k * ((((2 * eft_Pois_test) + 1) + 0) / (((eft_Pois_test^2) + eft_Pois_test) + 1)))) + (((0 * (((eft_Pois_test^2) + eft_Pois_test) + 1)) + (-1 * (((2 * eft_Pois_test) + 1) + 0))) - 0))
    '''
    assert (eft_Pois_test.ln().eval(1,2) - (2 * np.log(1**2 + 1**1 + 1) - (1**2 + 1**1 + 1) - np.log(2)))<1e-18 # float has precision of 1e-18
    assert (eft_Pois_test.ln().derivative().eval(1,2) - (2 * (2*1 + 1) * 1./(1**2 + 1**1 + 1) - (2*1 + 1)))<1e-18 # float has precision of 1e-18
