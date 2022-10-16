'''
EFT Likelihood fit
Brent R. Yates
The Ohio State University
Department of Physics
Oct. 8 2022
'''

import numpy as np

class constant():
    def __init__(self, val):
        if type(val) == constant:
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
        if not issubclass(type(rhs), constant):
            rhs = constant(rhs)
        if type(self) == constant and type(rhs) == constant:
            return constant(self.value() + rhs.value())
        return sum(self, rhs).simplify()

    def __sub__(self, rhs):
        if not issubclass(type(rhs), constant):
            rhs = constant(rhs)
        if type(self) == constant and type(rhs) == constant:
            return constant(self.value() - rhs.value())
        return diff(self, rhs).simplify()

    def __mul__(self, rhs):
        if not issubclass(type(rhs), constant):
            rhs = constant(rhs)
        if type(self) == constant and type(rhs) == constant:
            return constant(self.value() * rhs.value())
        if self.value() == 0 or rhs.value() == 0:
            return constant(0)
        return prod(self, rhs).simplify()

    def __div__(self, rhs):
        if not issubclass(type(rhs), constant):
            rhs = constant(rhs)
        if type(self) == constant and type(rhs) == constant:
            return constant(self.value() / rhs.value())
        if self.value() == 0 or rhs.value() == 0:
            return constant(0)
        return quotient(self, rhs).simplify()

    def __truediv__(self, rhs):
        if not issubclass(type(rhs), constant):
            rhs = constant(rhs)
        if type(self) == constant and type(rhs) == constant:
            return constant(self.value() / rhs.value())
        if self.value() == 0 or rhs.value() == 0:
            return constant(0)
        return quotient(self, rhs).simplify()

    def __floordiv__(self, rhs):
        return quotient(self, rhs)

    def __eq__(self, rhs):
        if isinstance(type(self), constant) and issubclass(type(rhs), int):
            return self.value() == rhs
        elif not isinstance(type(rhs), type(self)):
            return False
        else:
            return self.value() == rhs.value()

    def __lt__(self, rhs):
        if not issubclass(type(rhs), constant):
            rhs = constant(rhs)
        return np.abs(self.value()) < rhs.value()

    def set_param(self, k=None):
        return self

    def ln(self):
        #return constant(np.log(self.value()))
        return log(self.value())

    def derivative(self):
        if issubclass(type(self), constant) or issubclass(type(self.val_), constant):
            return constant(0)
        else:
            return self.val_.derivative()

    def eval(self, x=None):
        return self.simplify()


class log(constant):
    def __init__(self, val):
        super().__init__(val)

    def __str__(self):
        return 'ln(' + str(self.value()) + ')'

    def eval(self, k=None):
        if not issubclass(type(self.value()), constant):
            return constant(np.log(self.value()))
        else:
            return constant(np.log(self.value().eval(k).value()))


class parameter(constant):
    def __init__(self, param):
        self.param_ = param

    def value(self):
        return constant(-999)#self.param_

    def __str__(self):
        return self.param_

    def set_param(self, k):
        return constant(k)

    def ln(self):
        return log_parameter(self.param_)

    def derivative(self):
        return constant(0)


class log_parameter(parameter):
    def __init__(self, param):
        super().__init__(param)

    def __str__(self):
        return 'ln(' + str(self.param_) + ')'

    def set_param(self, k):
        return constant(np.log(k))

    def eval(self, k):
        return constant(np.log(np.math.factorial(k)))


class Factorial(parameter):
    def __str__(self):
        return self.param_ + '!'

    def set_param(self, k):
        return constant(np.math.factorial(k))

    def eval(self, k):
        return constant(np.math.factorial(k))


class sum(constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs#.simplify()
        self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_.value()) == constant and type(self.rhs_.value()) == constant:
            return constant(self.lhs_.value() + self.rhs_.value())
        elif type(self.lhs_.value()) == constant:
            return sum(self.lhs_, self.rhs_.value())
        #elif type(self.rhs_.value()) == constant:
        elif issubclass(type(self.rhs_), constant):
            return sum(self.lhs_.value(), self.rhs_)
        else:
            return sum(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' + ' + str(self.rhs_) + ')'

    def set_param(self, k):
        return sum(self.lhs_.set_param(k), self.rhs_.set_param(k))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return prod(constant(2), self.lhs_)
        elif type(self.lhs_.value()) == constant:
            return sum(self.lhs_, self.rhs_.simplify())
        elif type(self.rhs_.value()) == constant:
            return sum(self.lhs_.simplify(), self.rhs_)
        elif self.lhs_ is None or self.lhs_.value() == 0:
            return self.rhs_
        elif self.rhs_ is None or self.rhs_.value() == 0:
            return self.lhs_
        else:
            lhs = self.lhs_.simplify()
            rhs = self.rhs_.simplify()
            if type(lhs) == constant and type(rhs) == constant:
                return constant(lhs.value() + rhs.value())
            return sum(self.lhs_.simplify(), self.rhs_.simplify())

    def derivative(self):
        return sum(self.lhs_.derivative(), self.rhs_.derivative())

    def eval(self, x=None):
        return constant(self.lhs_.simplify().eval(x) + self.rhs_.simplify().eval(x))


class diff(constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs#.simplify()
        self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_) == constant and type(self.rhs_) == constant:
            return constant(self.lhs_ + self.rhs_)
        elif type(self.lhs_) == constant:
            return diff(self.lhs_, self.rhs_)
        elif type(self.rhs_.value()) == constant:
            return diff(self.lhs_.value(), self.rhs_)
        else:
            return diff(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' - ' + str(self.rhs_) + ')'

    def set_param(self, k):
        return diff(self.lhs_.set_param(k), self.rhs_.set_param(k))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return constant(0)
        elif self.lhs_.value() == 0:
            return self.rhs_
        elif issubclass(type(self.rhs_), constant) and self.rhs_.value() == 0:
            return self.lhs_
        else:
            lhs = self.lhs_.simplify()
            rhs = self.rhs_
            if issubclass(type(rhs), constant):
                rhs = rhs.simplify()
            if type(lhs) == constant and type(rhs) == constant:
                return constant(lhs.value() - rhs.value())
            return diff(lhs, rhs)

    def derivative(self):
        return diff(self.lhs_.derivative(), self.rhs_.derivative())

    def eval(self, x=None):
        return constant(self.lhs_.eval(x) - self.rhs_.eval(x))


class prod(constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs#.simplify()
        self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_) == constant and type(self.rhs_) == constant:
            return constant(self.lhs_.value() * self.rhs_.value())
        elif type(self.lhs_) == constant:
            return prod(self.lhs_, self.rhs_.value())
        elif type(self.rhs_) == constant:
            return prod(self.lhs_.value(), self.rhs_)
        else:
            return prod(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' * ' + str(self.rhs_) + ')'

    def set_param(self, k):
        return prod(self.lhs_.set_param(k).simplify(), self.rhs_.set_param(k).simplify())

    def simplify(self):
        if self.lhs_.value() == 0 or self.rhs_.value() == 0:
            return constant(0)
        elif self.lhs_ == self.rhs_:
            return prod(self.lhs_.symbol_, constant(2))
        elif self.lhs_ == 1:
            return self.rhs_
        elif self.rhs_ == 1:
            return self.lhs_
        else:
            lhs = self.lhs_.simplify()
            rhs = self.rhs_.simplify()
            if type(lhs) == constant and type(rhs) == constant:
                return constant(lhs.value() * rhs.value())
            return prod(self.lhs_.simplify(), self.rhs_.simplify())

    def ln(self):
        return sum(self.lhs_.ln(), self.rhs_.ln())

    def derivative(self):
        return sum(prod(self.lhs_.derivative(), self.rhs_), prod(self.lhs_, self.rhs_.derivative()))

    def eval(self, x=None):
        '''
        if type(self.lhs_) == constant:
            return constant(self.lhs_ * self.rhs_.eval(x))
        elif type(self.rhs_) == constant:
            return constant(self.lhs_.eval(x) * self.rhs_)
        else:
            return constant(self.lhs_.eval(x) * self.rhs_.eval(x))
        '''
        return constant(self.lhs_.simplify().eval(x) * self.rhs_.simplify().eval(x))
        return constant(self.lhs_.eval(x) * self.rhs_.eval(x))


class quotient(constant):
    def __init__(self, lhs, rhs):
        if lhs.value() == 0:
            self.lhs_ = constant(0)
            self.rhs_ = constant(0)
        elif rhs.value() == 0:
            raise Exception('Trying to divide by 0!')
        else:
            self.lhs_ = lhs#.simplify()
            self.rhs_ = rhs#.simplify()

    def value(self):
        if self.rhs_.value() == 0:
            raise Exception('Trying to divide by 0!')
        if type(self.lhs_) == constant and type(self.rhs_) == constant:
            return constant(self.lhs_ / self.rhs_)
        elif type(self.lhs_) == constant:
            return quotient(self.lhs_, self.rhs_.value())
        elif type(self.rhs_) == constant:
            return quotient(self.lhs_.value(), self.rhs_)
        else:
            return quotient(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' / ' + str(self.rhs_) + ')'

    def set_param(self, k):
        return quotient(self.lhs_.set_param(k), self.rhs_.set_param(k))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return constant(1)
        elif self.lhs_.value() == 1:
            return power(self.rhs_, constant(-1))
            return constant(1)/self.rhs_
        elif self.rhs_.value() == 1:
            return power(self.lhs_, constant(-1))
            return constant(1)/self.lhs_
        else:
            lhs = self.lhs_.simplify()
            rhs = self.rhs_.simplify()
            if type(lhs) == constant and type(rhs) == constant:
                return constant(lhs.value() / rhs.value())
            return quotient(self.lhs_.simplify(), self.rhs_.simplify())

    def ln(self):
        return diff(self.lhs_.ln(), self.rhs_.ln())

    def derivative(self):
        # FIXME check for constant lhs/rhs
        if isinstance(type(self.rhs_.value()), constant) or type(self.rhs_.value()) == constant or type(self.rhs_) == parameter:
            return quotient(self.lhs_.derivative(), self.rhs_)
        else:
            return diff(quotient(self.lhs_.derivative(), self.rhs_), quotient(prod(self.lhs_, self.rhs_.derivative()), self.rhs_))

    def eval(self, x=None):
        if type(self.lhs_) == constant:
            return constant(self.lhs_ / self.rhs_.eval(x))
        elif type(self.rhs_) == constant:
            return constant(self.lhs_.eval(x) / self.rhs_.value())
        else:
            return constant(self.lhs_.eval(x) / self.rhs_.eval(x))


class variable(constant):
    def __init__(self, symbol='x'):
        self.symbol_ = symbol

    def __eq__(self, rhs):
        if isinstance(rhs, variable):
            return self.symbol_ == rhs.symbol_
        else:
            return False

    def __str__(self):
        return str(self.symbol_)

    def value(self):
        return self

    def ln(self):
        return log_variable(self.symbol_)

    def derivative(self):
        return constant(1)

    def eval(self, x):
        return constant(x)


class log_variable(variable):
    def __init__(self, symbol='x'):
        super().__init__(symbol)

    def __str__(self):
        return 'ln(' + str(self.symbol_) + ')'

    def derivative(self):
        return quotient(constant(1), variable(self.symbol_))


class power(variable):
    def __init__(self, symbol='x', power=1):
        if not issubclass(type(power), constant):
            self.val_ = constant(power)
        else:
            self.val_ = power
        if isinstance(symbol, constant):
            self.symbol_ = symbol
        else:
            self.symbol_ = variable(symbol)

    def __eq__(self, rhs):
        if isinstance(rhs, power) and self.val_ == rhs.val_:
            return self.symbol_ == rhs.symbol_
        else:
            return False

    def __str__(self):
        if self.val_.value() == -1:
            return '(1 / ' + str(self.symbol_) + ')'
        return '(' + str(self.symbol_) + '^' + str(self.val_) + ')'

    def simplify(self):
        if self.val_.value() == 0:
            return constant(1)
        elif self.val_.value() == 1:
            return variable(self.symbol_)
        else:
            return self

    def set_param(self, k):
        return power(self.val_.set_param(k))

    def ln(self):
        return self.val_ * self.symbol_.ln()

    def derivative(self):
        return self.val_ * power(self.symbol_, self.val_-1) # FIXME be carfule with x^1

    def eval(self, x):
        return constant(self.symbol_.eval(x).value()**self.val_.eval(x).value())


class polynomial(constant):
    def __init__(self, symbol='x', const=[2, 1, 1], order=2):
        self.symbol_ = symbol
        if len(const) != order+1:
            raise Exception('Please specify enough constants!')
        self.val_ = power(symbol, order)
        for i in range(1, order+1):
            self.val_ += power(symbol, order-i)
            self.val_.simplify()

    def __str__(self):
        return str(self.val_)

    def ln(self):
        return log_polynomial(self)

    def derivative(self):
        return self.val_.derivative()

    def eval(self, x):
        return constant(self.val_.eval(x))


class log_polynomial(polynomial):
    def __init__(self, var):
        self.symbol_ = var.symbol_
        self.val_ = var.val_

    def __str__(self):
        return 'ln(' + str(self.val_) + ')'

    def derivative(self):
        return quotient(self.val_.derivative(), self.val_)

    def eval(self, x):
        return constant(np.log(self.val_.eval(x).value()))


class expo(constant):
    def __init__(self, var):
        self.val_ = var

    def __str__(self):
        return 'e^' + str(self.val_)

    def ln(self):
        return self.val_

    def derivative(self):
        return prod(self.val_.derivative(), self)

    def eval(self, x):
        return constant(np.exp(self.val_.eval(x).value()))


class poisson(prod):
    def __init__(self, symbol='x', param=1):
        self.symbol_ = symbol
        self.var_ = prod(constant(-1), variable(self.symbol_))
        self.lhs_ = power(symbol, parameter(param))
        self.rhs_ = quotient(expo(self.var_), Factorial(param))
        self.param_ = param

    def ln(self):
        return log_poisson(sum(self.lhs_.ln(), self.rhs_.ln()), self.param_)

    def derivative(self):
        return derivative_poisson(prod(self.lhs_, self.rhs_).derivative(), self.param_)

    def eval(self, x, k):
        var = constant(-1) * variable(k)
        lhs = power(self.symbol_, constant(k)).eval(x)
        rhs = expo(var).eval(x) / Factorial(k).eval(k)
        return constant(lhs * rhs)


class log_poisson(sum):
    def __init__(self, pois_l, k):
        self.lhs_ = pois_l.lhs_
        self.rhs_ = pois_l.rhs_
        self.k_ = k

    def derivative(self):
        return derivative_poisson(sum(self.lhs_.derivative(), self.rhs_.derivative()), self.k_)

    def eval(self, x, k):
        lhs_ = self.lhs_.set_param(k)
        rhs_ = self.rhs_.set_param(k)
        return constant(lhs_.eval(x) + rhs_.eval(x))


class derivative_poisson(sum):
    def __init__(self, pois_d, k):
        self.lhs_ = pois_d.lhs_
        self.rhs_ = pois_d.rhs_
        self.k_ = k

    def eval(self, x, k):
        lhs_ = self.lhs_.set_param(k)
        rhs_ = self.rhs_.set_param(k)
        return constant(lhs_.eval(x) + rhs_.eval(x))


class eft_poisson(poisson):
    def __init__(self, symbol='x', consts=[1, 1, 1], param=1):
        super().__init__(symbol, param)
        self.var_ = prod(constant(-1), polynomial(self.symbol_, consts, 2))
        self.lhs_ = power(polynomial(self.symbol_), parameter(param))
        self.rhs_ = quotient(expo(self.var_), Factorial(param))

    def eval(self, x, k):
        var = self.var_.eval(x)
        lhs = power(polynomial(self.symbol_), constant(k)).eval(x)
        rhs = expo(var).eval(x) / Factorial(k).eval(k)
        return constant(lhs * rhs)


if __name__ == '__main__':
    pass
