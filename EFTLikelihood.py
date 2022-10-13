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
        if isinstance(val, Constant):
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
        if issubclass(type(rhs), int):
            rhs = Constant(rhs)
        return Sum(self, rhs).simplify()

    def __sub__(self, rhs):
        if issubclass(type(rhs), int):
            rhs = Constant(rhs)
        return Diff(self, rhs).simplify()

    def __mul__(self, rhs):
        return Prod(self, rhs).simplify()

    def __div__(self, rhs):
        return Quotient(self, rhs).simplify()

    def __truediv__(self, rhs):
        return Quotient(self, rhs)

    def __floordiv__(self, rhs):
        return Quotient(self, rhs)

    def __eq__(self, rhs):
        if issubclass(type(self), Constant) and issubclass(type(rhs), int):
            return self.value() == rhs
        elif not isinstance(type(rhs), type(self)):
            return False
        else:
            return self.value() == rhs.value()

    def eval(self, x=None):
        return self


class Parameter(Constant):
    def __init__(self, param):
        self.param_ = param

    def value(self):
        return self.param_

    def __str__(self):
        return self.param_


class Factorial(Parameter):
    def __str__(self):
        return self.param_ + '!'

    def eval(self, k):
        return np.math.factorial(k)


class Sum(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs.simplify()
        self.rhs_ = rhs.simplify()

    def value(self):
        return self.lhs_.value() + self.rhs_.value()

    def __str__(self):
        return '(' + str(self.lhs_) + ' + ' + str(self.rhs_) + ')'

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Prod(Constant(2), self.lhs_)
        if issubclass(type(self.lhs_), int):
            return self.lhs_ + self.rhs_.simplify()
        if issubclass(type(self.rhs_), int):
            return self.lhs_.simplify() + self.rhs_
        elif self.lhs_ is None or self.lhs_.value() == 0:
            return self.rhs_
        elif self.rhs_ is None or self.rhs_.value() == 0:
            return self.lhs_
        else:
            return Sum(self.lhs_.simplify(), self.rhs_.simplify())

    def eval(self, x=None):
        return Constant(self.lhs_.eval(x) + self.rhs_.eval(x))


class Diff(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs
        self.rhs_ = rhs

    def value(self):
        return self.lhs_.value() - self.rhs_.value()

    def __str__(self):
        return '(' + str(self.lhs_) + ' - ' + str(self.rhs_) + ')'

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Constant(0)
        elif self.lhs_.value() == 0:
            return self.rhs_
        elif self.rhs_.value() == 0:
            return self.lhs_
        else:
            return Diff(self.lhs_.simplify(), self.rhs_.simplify())

    def eval(self, x=None):
        return Constant(self.lhs_.eval(x) - self.rhs_.eval(x))


class Prod(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs
        self.rhs_ = rhs

    def value(self):
        return self.lhs_.value() * self.rhs_.value()

    def __str__(self):
        return '(' + str(self.lhs_) + ' * ' + str(self.rhs_) + ')'

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Power(self.lhs_.symbol_, Constant(2))
        elif self.lhs_.value() == 1:
            return self.rhs_
        elif self.rhs_.value() == 1:
            return self.lhs_
        else:
            return Prod(self.lhs_.simplify(), self.rhs_.simplify())

    def eval(self, x=None):
        return Constant(self.lhs_.eval(x) * self.rhs_.eval(x))


class Quotient(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs
        self.rhs_ = rhs

    def value(self):
        if self.rhs_.value() == 0:
            raise Exception('Trying to divide by 0!')
        return self.lhs_.value() / self.rhs_.value()

    def __str__(self):
        return '(' + str(self.lhs_) + ' / ' + str(self.rhs_) + ')'

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Constant(1)
        elif self.lhs_.value() == 1:
            return self.rhs_
        elif self.rhs_.value() == 1:
            return self.lhs_
        else:
            return Quotient(self.lhs_.simplify(), self.rhs_.simplify())

    def eval(self, x=None):
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
        return self.symbol_

    def __call__(self, x):
        return self.eval(x)

    def value(self):
        return self

    def eval(self, x):
        return Constant(x)


class Power(Variable):
    def __init__(self, symbol='x', power=1):
        self.power_ = Constant(power)
        self.symbol_ = symbol

    def __eq__(self, rhs):
        if isinstance(rhs, Power) and self.power_ == rhs.power_:
            return self.symbol_ == rhs.symbol_
        else:
            return False

    def __str__(self):
        return '(' + self.symbol_ + '^' + str(self.power_) + ')'

    def simplify(self):
        if self.power_.value() == 0:
            return Constant(1)
        elif self.power_.value() == 1:
            return Variable(self.symbol_)
        else:
            return self

    def eval(self, x):
        return Constant(x**self.power_.value())


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

class Expo(Variable):
    def __init__(self, var):
        self.val_ = var

    def __str__(self):
        return 'e^' + str(self.val_)

    def eval(self, x):
        return Constant(np.exp(self.val_.eval(x).value()))


class Poisson(Prod):
    def __init__(self, symbol='x', param=1):
        self.symbol_ = symbol
        var = Constant(-1) * Variable(self.symbol_)
        self.lhs_ = Power(symbol, Parameter(param))
        self.rhs_ = Expo(var) / Factorial(param)

    def eval(self, x, k):
        var = Constant(-1) * Variable(self.symbol_)
        self.lhs_ = Power(self.symbol_, Constant(k))
        self.rhs_ = Expo(var) / Constant(Factorial(k).eval(k))
        return Constant(self.lhs_.eval(x) * self.rhs_.eval(x))


if __name__ == '__main__':
    print(Constant(1))
    print(Constant(1)+Constant(2))
    print(Constant(1)-Constant(2))
    print(Constant(1)*Constant(2))
    print(Constant(1)/Constant(2))
    #print(Constant(1)/Constant(0))
    c = Constant(1)/Constant(2)
    print((Constant(1)/Constant(2)).eval())
    x = Variable('x')
    print(x)
    print(Variable('x').eval(1))
    print(x(2))
    print(x+x)
    print(x+x.eval(1))
    print((x+x).eval(1))
    print(x-x.eval(1))
    print((x-x).eval(1))
    p = Power('x', 2)
    print(p+p)
    print((p+p).eval(2))
    print(x*x)
    print(x*x.eval(1))
    print(x/x.eval(1))
    print((x*x)(2))
    p = Polynomial('x', [2,1,1],  2)
    print(p)
    print p+x
    print(Power('x', 0).simplify())
    p = Power('x', 1).simplify()
    print(p)
    print(Expo(Variable('x')))
    print(Factorial('k'))
    t = Constant(-1) * Variable('x')
    print(t)
    print(t.eval(1))
    pois = Poisson('x', 'k')
    print(pois)
    print(pois.eval(1,1))
    print(pois.eval(1,2))
    assert Expo(Constant(-1) * Variable('x')).eval(1), np.exp(-1)
    assert (pois.eval(1,2).value() - 1**2 * np.exp(-1) / np.math.factorial(2))<1e-5
