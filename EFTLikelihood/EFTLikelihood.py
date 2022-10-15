'''
EFT likelihood fit
Brent R. Yates
The Ohio State University
Department of Physics
Oct. 8 2022
'''

import numpy as np

class Constant():
    def __init__(self, val):
        if type(val) == Constant:
            self.val_ = val.value()
        else:
            self.val_ = val
        self.simplify()

    def value(self):
        return self.val_

    def simplify(self):
        return self

    def __str__(self):
        return str(self.value())

    def __add__(self, rhs):
        if not issubclass(type(rhs), Constant):
            rhs = Constant(rhs)
        if type(self) == Constant and type(rhs) == Constant:
            return Constant(self.value() + rhs.value())
        return Sum(self, rhs).simplify()

    def __sub__(self, rhs):
        if not issubclass(type(rhs), Constant):
            rhs = Constant(rhs)
        if type(self) == Constant and type(rhs) == Constant:
            return Constant(self.value() - rhs.value())
        return Diff(self, rhs).simplify()

    def __mul__(self, rhs):
        if not issubclass(type(rhs), Constant):
            rhs = Constant(rhs)
        if type(self) == Constant and type(rhs) == Constant:
            return Constant(self.value() * rhs.value())
        if self.value() == 0 or rhs.value() == 0:
            return Constant(0)
        return Prod(self, rhs).simplify()

    def __div__(self, rhs):
        if not issubclass(type(rhs), Constant):
            rhs = Constant(rhs)
        if type(self) == Constant and type(rhs) == Constant:
            return Constant(self.value() / rhs.value())
        if self.value() == 0 or rhs.value() == 0:
            return Constant(0)
        return Quotient(self, rhs).simplify()

    def __truediv__(self, rhs):
        if not issubclass(type(rhs), Constant):
            rhs = Constant(rhs)
        if type(self) == Constant and type(rhs) == Constant:
            return Constant(self.value() / rhs.value())
        if self.value() == 0 or rhs.value() == 0:
            return Constant(0)
        return Quotient(self, rhs).simplify()

    def __floordiv__(self, rhs):
        return Quotient(self, rhs)

    def __eq__(self, rhs):
        if isinstance(type(self), Constant) and issubclass(type(rhs), int):
            return self.value() == rhs
        elif not isinstance(type(rhs), type(self)):
            return False
        else:
            return self.value() == rhs.value()

    def __lt__(self, rhs):
        if not issubclass(type(rhs), Constant):
            rhs = Constant(rhs)
        return np.abs(self.value()) < rhs.value()

    def set_param(self, k=None):
        return self

    def ln(self):
        #return Constant(np.log(self.value()))
        return Log(self.value())

    def derivative(self):
        if issubclass(type(self), Constant) or issubclass(type(self.val_), Constant):
            return Constant(0)
        else:
            return self.val_.derivative()

    def eval(self, x=None):
        return self.simplify()


class Log(Constant):
    def __init__(self, val):
        super().__init__(val)

    def __str__(self):
        return 'ln(' + str(self.value()) + ')'

    def eval(self, k=None):
        if not issubclass(type(self.value()), Constant):
            return Constant(np.log(self.value()))
        else:
            return Constant(np.log(self.value().eval(k).value()))


class Parameter(Constant):
    def __init__(self, param):
        self.param_ = param

    def value(self):
        return Constant(-999)#self.param_

    def __str__(self):
        return self.param_

    def set_param(self, k):
        return Constant(k)

    def ln(self):
        return LogParameter(self.param_)

    def derivative(self):
        return Constant(0)


class LogParameter(Parameter):
    def __init__(self, param):
        super().__init__(param)

    def __str__(self):
        return 'ln(' + str(self.param_) + ')'

    def set_param(self, k):
        return Constant(np.log(k))

    def eval(self, k):
        return Constant(np.log(np.math.factorial(k)))


class Factorial(Parameter):
    def __str__(self):
        return self.param_ + '!'

    def set_param(self, k):
        return Constant(np.math.factorial(k))

    def eval(self, k):
        return Constant(np.math.factorial(k))


class Sum(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs#.simplify()
        self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_.value()) == Constant and type(self.rhs_.value()) == Constant:
            return Constant(self.lhs_.value() + self.rhs_.value())
        elif type(self.lhs_.value()) == Constant:
            return Sum(self.lhs_, self.rhs_.value())
        elif type(self.rhs_.value()) == Constant:
            return Sum(self.lhs_.value(), self.rhs_)
        else:
            return Sum(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' + ' + str(self.rhs_) + ')'

    def set_param(self, k):
        return Sum(self.lhs_.set_param(k), self.rhs_.set_param(k))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Prod(Constant(2), self.lhs_)
        elif type(self.lhs_.value()) == Constant:
            return Sum(self.lhs_, self.rhs_.simplify())
        elif type(self.rhs_.value()) == Constant:
            return Sum(self.lhs_.simplify(), self.rhs_)
        elif self.lhs_ is None or self.lhs_.value() == 0:
            return self.rhs_
        elif self.rhs_ is None or self.rhs_.value() == 0:
            return self.lhs_
        else:
            lhs = self.lhs_.simplify()
            rhs = self.rhs_.simplify()
            if type(lhs)==Constant and type(rhs)==Constant:
                return Constant(lhs.value() + rhs.value())
            return Sum(self.lhs_.simplify(), self.rhs_.simplify())

    def derivative(self):
        return Sum(self.lhs_.derivative(), self.rhs_.derivative())

    def eval(self, x=None):
        return Constant(self.lhs_.simplify().eval(x) + self.rhs_.simplify().eval(x))


class Diff(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs#.simplify()
        self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_) == Constant and type(self.rhs_) == Constant:
            return Constant(self.lhs_ + self.rhs_)
        elif type(self.lhs_) == Constant:
            return Diff(self.lhs_, self.rhs_)
        elif type(self.rhs_.value()) == Constant:
            return Diff(self.lhs_.value(), self.rhs_)
        else:
            return Diff(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' - ' + str(self.rhs_) + ')'

    def set_param(self, k):
        return Diff(self.lhs_.set_param(k), self.rhs_.set_param(k))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Constant(0)
        elif self.lhs_.value() == 0:
            return self.rhs_
        elif issubclass(type(self.rhs_), Constant) and self.rhs_.value() == 0:
            return self.lhs_
        else:
            lhs = self.lhs_.simplify()
            rhs = self.rhs_
            if issubclass(type(rhs), Constant):
                rhs = rhs.simplify()
            if type(lhs)==Constant and type(rhs)==Constant:
                return Constant(lhs.value() - rhs.value())
            return Diff(lhs, rhs)

    def derivative(self):
        return Diff(self.lhs_.derivative(), self.rhs_.derivative())

    def eval(self, x=None):
        return Constant(self.lhs_.eval(x) - self.rhs_.eval(x))


class Prod(Constant):
    def __init__(self, lhs, rhs):
        #if lhs.value() == 0 or rhs.value() == 0:
        #    self.lhs_ = Constant(0)
        #    self.rhs_ = Constant(0)
        #else:
            self.lhs_ = lhs#.simplify()
            self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_) == Constant and type(self.rhs_) == Constant:
            return Constant(self.lhs_.value() * self.rhs_.value())
        elif type(self.lhs_) == Constant:
            return Prod(self.lhs_, self.rhs_.value())
        elif type(self.rhs_) == Constant:
            return Prod(self.lhs_.value(), self.rhs_)
        else:
            return Prod(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' * ' + str(self.rhs_) + ')'

    def set_param(self, k):
        return Prod(self.lhs_.set_param(k).simplify(), self.rhs_.set_param(k).simplify())

    def simplify(self):
        if self.lhs_.value() == 0 or self.rhs_.value() == 0:
            return Constant(0)
        elif self.lhs_ == self.rhs_:
            return Prod(self.lhs_.symbol_, Constant(2))
        elif self.lhs_ == 1:
            return self.rhs_
        elif self.rhs_ == 1:
            return self.lhs_
        else:
            lhs = self.lhs_.simplify()
            rhs = self.rhs_.simplify()
            if type(lhs)==Constant and type(rhs)==Constant:
                return Constant(lhs.value() * rhs.value())
            return Prod(self.lhs_.simplify(), self.rhs_.simplify())

    def ln(self):
        return Sum(self.lhs_.ln(), self.rhs_.ln())

    def derivative(self):
        return Sum(Prod(self.lhs_.derivative(), self.rhs_), Prod(self.lhs_, self.rhs_.derivative()))

    def eval(self, x=None):
        '''
        if type(self.lhs_) == Constant:
            return Constant(self.lhs_ * self.rhs_.eval(x))
        elif type(self.rhs_) == Constant:
            return Constant(self.lhs_.eval(x) * self.rhs_)
        else:
            return Constant(self.lhs_.eval(x) * self.rhs_.eval(x))
        '''
        return Constant(self.lhs_.simplify().eval(x) * self.rhs_.simplify().eval(x))
        return Constant(self.lhs_.eval(x) * self.rhs_.eval(x))


class Quotient(Constant):
    def __init__(self, lhs, rhs):
        if lhs.value() == 0:
            self.lhs_ = Constant(0)
            self.rhs_ = Constant(0)
        elif rhs.value() == 0:
            raise Exception('Trying to divide by 0!')
        else:
             self.lhs_ = lhs#.simplify()
             self.rhs_ = rhs#.simplify()

    def value(self):
        if self.rhs_.value() == 0:
            raise Exception('Trying to divide by 0!')
        if type(self.lhs_) == Constant and type(self.rhs_) == Constant:
            return Constant(self.lhs_ / self.rhs_)
        elif type(self.lhs_) == Constant:
            return Quotient(self.lhs_, self.rhs_.value())
        elif type(self.rhs_) == Constant:
            return Quotient(self.lhs_.value(), self.rhs_)
        else:
            return Quotient(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' / ' + str(self.rhs_) + ')'

    def set_param(self, k):
        return Quotient(self.lhs_.set_param(k), self.rhs_.set_param(k))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Constant(1)
        elif self.lhs_.value() == 1:
            return Power(self.rhs_, Constant(-1))
            return Constant(1)/self.rhs_
        elif self.rhs_.value() == 1:
            return Power(self.lhs_, Constant(-1))
            return Constant(1)/self.lhs_
        else:
            lhs = self.lhs_.simplify()
            rhs = self.rhs_.simplify()
            if type(lhs)==Constant and type(rhs)==Constant:
                return Constant(lhs.value() / rhs.value())
            return Quotient(self.lhs_.simplify(), self.rhs_.simplify())

    def ln(self):
        return Diff(self.lhs_.ln(), self.rhs_.ln())

    def derivative(self):
        # FIXME check for constant lhs/rhs
        if isinstance(type(self.rhs_.value()), Constant) or type(self.rhs_.value()) == Constant or type(self.rhs_) == Parameter:
            return Quotient(self.lhs_.derivative(), self.rhs_)
        else:
            return Diff(Quotient(self.lhs_.derivative(), self.rhs_), Quotient(Prod(self.lhs_, self.rhs_.derivative()), self.rhs_))

    def eval(self, x=None):
        if type(self.lhs_) == Constant:
            return Constant(self.lhs_ / self.rhs_.eval(x))
        elif type(self.rhs_) == Constant:
            return Constant(self.lhs_.eval(x) / self.rhs_.value())
        else:
            return Constant(self.lhs_.eval(x) / self.rhs_.eval(x))


class Variable(Constant):
    def __init__(self, symbol='x'):
        self.symbol_ = symbol

    def __eq__(self, rhs):
        if isinstance(rhs, Variable):
            return self.symbol_ == rhs.symbol_
        else:
            return False

    def __str__(self):
        return str(self.symbol_)

    def __call__(self, x):
        return Constant(self.eval(x))

    '''
    def set_param(self, k):
        return Variable(self.symbol_.set_param(k))
    '''

    def value(self):
        return self

    def ln(self):
        return LogVariable(self.symbol_)

    def derivative(self):
        return Constant(1)

    def eval(self, x):
        return Constant(x)


class LogVariable(Variable):
    def __init__(self, symbol='x'):
        super().__init__(symbol)

    def __str__(self):
        return 'ln(' + str(self.symbol_) + ')'

    '''
    def set_param(self, k):
        return LogVariable(self.val_.set_param(k))
    '''

    def derivative(self):
        return Quotient(Constant(1), Variable(self.symbol_))


class Power(Variable):
    def __init__(self, symbol='x', power=1):
        if not issubclass(type(power), Constant):
            self.val_ = Constant(power)
        else:
            self.val_ = power
        self.symbol_ = symbol

    def __eq__(self, rhs):
        if isinstance(rhs, Power) and self.val_ == rhs.val_:
            return self.symbol_ == rhs.symbol_
        else:
            return False

    def __str__(self):
        if self.val_.value() == -1:
            return '(1 / ' + str(self.symbol_) + ')'
        return '(' + str(self.symbol_) + '^' + str(self.val_) + ')'

    def simplify(self):
        if self.val_.value() == 0:
            return Constant(1)
        elif self.val_.value() == 1:
            return Variable(self.symbol_)
        else:
            return self

    def set_param(self, k):
        return Power(self.val_.set_param(k))

    def ln(self):
        return self.val_ * Variable(self.symbol_).ln()

    def derivative(self):
        return self.val_ * Power(self.symbol_, self.val_-1) # FIXME be carfule with x^1

    def eval(self, x):
        return Constant(x**self.val_.eval(x).value())


class Polynomial(Constant):
    def __init__(self, symbol='x', const=[2,1,1], order=2):
        self.symbol_ = symbol
        if len(const) != order+1:
            raise Exception('Please specify enough constants!')
        self.val_ = Power(symbol, order)
        for i in range(1, order+1):
            self.val_ += Power(symbol, order-i)
            self.val_.simplify()

    def __str__(self):
        return str(self.val_)

    def ln(self):
        return LogPolynomial(self)

    def derivative(self):
        return self.val_.derivative()

    def eval(self, x):
        return Constant(self.val_.eval(x))


class LogPolynomial(Polynomial):
    def __init__(self, var):
        self.symbol_ = var.symbol_
        self.val_ = var.val_

    def __str__(self):
        return 'ln(' + str(self.val_) + ')'

    def eval(self, x):
        return Constant(np.log(self.val_.eval(x)))


class Expo(Constant):
    def __init__(self, var):
        self.val_ = var

    def __str__(self):
        return 'e^' + str(self.val_)

    def ln(self):
        return self.val_

    def derivative(self):
        return Prod(self.val_.derivative(), self)

    def eval(self, x):
        return Constant(np.exp(self.val_.eval(x).value()))


class Poisson(Prod):
    def __init__(self, symbol='x', param=1):
        self.symbol_ = symbol
        var = Prod(Constant(-1), Variable(self.symbol_))
        self.lhs_ = Power(symbol, Parameter(param))
        self.rhs_ = Quotient(Expo(var), Factorial(param))
        self.param_ = param

    def ln(self):
        return LogPoisson(Sum(self.lhs_.ln(), self.rhs_.ln()), self.param_)

    def derivative(self):
        return DerivativePoisson(Prod(self.lhs_, self.rhs_).derivative(), self.param_)

    def eval(self, x, k):
        var = Constant(-1) * Variable(k)
        lhs = Power(self.symbol_, Constant(k)).eval(x)
        rhs = Expo(var).eval(x) / Factorial(k).eval(k)
        return Constant(lhs * rhs)


class LogPoisson(Sum):
    def __init__(self, pois_l, k):
        self.lhs_ = pois_l.lhs_
        self.rhs_ = pois_l.rhs_
        self.k_ = k

    def derivative(self):
        return DerivativePoisson(Sum(self.lhs_.derivative(), self.rhs_.derivative()), self.k_)

    def eval(self, x, k):
        lhs_ = self.lhs_.set_param(k)
        rhs_ = self.rhs_.set_param(k)
        return Constant(lhs_.eval(x) + rhs_.eval(x))


class DerivativePoisson(Sum):
    def __init__(self, pois_d, k):
        self.lhs_ = pois_d.lhs_
        self.rhs_ = pois_d.rhs_
        self.k_ = k

    def eval(self, x, k):
        lhs_ = self.lhs_.set_param(k)
        rhs_ = self.rhs_.set_param(k)
        return Constant(lhs_.eval(x) + rhs_.eval(x))


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


if __name__ == '__main__':
    test_Constant()
    test_Variable()
    test_Power()
    test_Polynomial()
    test_Expo()
    test_Poisson()