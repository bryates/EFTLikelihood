'''
EFT Likelihood fit
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
        if not isinstance(type(rhs), type(self)):
            return False
        return self.value() == rhs.value()

    def __lt__(self, rhs):
        if not issubclass(type(rhs), Constant):
            rhs = Constant(rhs)
        return np.abs(self.value()) < rhs.value()

    def set_param(self, k_in=None):
        return self

    def ln(self):
        return Log(self.value())

    def derivative(self, var='x'):
        if issubclass(type(self), Constant) or issubclass(type(self.val_), Constant):
            return Constant(0)
        return self.val_.derivative(var)

    def eval(self, x_in=None):
        return self.simplify()


class Log(Constant):
    def __init__(self, val):
        super().__init__(val)

    def __str__(self):
        return 'ln(' + str(self.value()) + ')'

    def eval(self, k_in=None):
        if not issubclass(type(self.value()), Constant):
            return Constant(np.log(self.value()))
        return Constant(np.log(self.value().eval(k_in).value()))


class Parameter(Constant):
    def __init__(self, param):
        self.param_ = param

    def value(self):
        return Constant(-999)#self.param_

    def __str__(self):
        return self.param_

    def set_param(self, k_in):
        return Constant(k_in)

    def ln(self):
        return LogParameter(self.param_)

    def derivative(self, var='x'):
        return Constant(0)


class LogParameter(Parameter):
    def __init__(self, param):
        super().__init__(param)

    def __str__(self):
        return 'ln(' + str(self.param_) + ')'

    def set_param(self, k_in):
        return Constant(np.log(k_in))

    def eval(self, k_in):
        return Constant(np.log(np.math.factorial(k_in)))


class Factorial(Parameter):
    def __str__(self):
        return self.param_ + '!'

    def set_param(self, k_in):
        return Constant(np.math.factorial(k_in))

    def ln(self):
        return LogFactorial(self.param_)

    def eval(self, k_in):
        return Constant(np.math.factorial(k_in))


class LogFactorial(Factorial):
    def __init__(self, param):
        super().__init__(param)

    def __str__(self):
        return 'ln(' + str(self.param_) + '!)'

    def set_param(self, k_in):
        return Constant(np.log(float(np.math.factorial(k_in))))

    def eval(self, k_in):
        return Constant(np.log(float(np.math.factorial(k_in))))


class Sum(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs#.simplify()
        self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_) == Constant and type(self.rhs_) == Constant:
            return Constant(self.lhs_.value() + self.rhs_.value())
        if type(self.lhs_) == Constant:
            return Sum(self.lhs_, self.rhs_.value())
        if issubclass(type(self.rhs_), Constant):
            return Sum(self.lhs_.value(), self.rhs_)
        return Sum(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' + ' + str(self.rhs_) + ')'

    def set_param(self, k_in):
        return Sum(self.lhs_.set_param(k_in), self.rhs_.set_param(k_in))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Prod(Constant(2), self.lhs_)
        if type(self.lhs_.value()) == Constant:
            return Sum(self.lhs_, self.rhs_.simplify())
        if type(self.rhs_.value()) == Constant:
            return Sum(self.lhs_.simplify(), self.rhs_)
        if self.lhs_ is None or self.lhs_.value() == 0:
            return self.rhs_
        if self.rhs_ is None or self.rhs_.value() == 0:
            return self.lhs_
        lhs = self.lhs_.simplify()
        rhs = self.rhs_.simplify()
        if type(lhs) == Constant and type(rhs) == Constant:
            return Constant(lhs.value() + rhs.value())
        return Sum(self.lhs_.simplify(), self.rhs_.simplify())

    def derivative(self, var='x'):
        return Sum(self.lhs_.derivative(var), self.rhs_.derivative(var))

    def eval(self, x_in=None):
        return Constant(self.lhs_.simplify().eval(x_in) + self.rhs_.simplify().eval(x_in))


class Diff(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs#.simplify()
        self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_) == Constant and type(self.rhs_) == Constant:
            return Constant(self.lhs_ + self.rhs_)
        if type(self.lhs_) == Constant:
            return Diff(self.lhs_, self.rhs_)
        if type(self.rhs_.value()) == Constant:
            return Diff(self.lhs_.value(), self.rhs_)
        return Diff(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' - ' + str(self.rhs_) + ')'

    def set_param(self, k_in):
        return Diff(self.lhs_.set_param(k_in), self.rhs_.set_param(k_in))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Constant(0)
        if self.lhs_.value() == 0:
            return self.rhs_
        if issubclass(type(self.rhs_), Constant) and self.rhs_.value() == 0:
            return self.lhs_
        lhs = self.lhs_.simplify()
        rhs = self.rhs_
        if issubclass(type(rhs), Constant):
            rhs = rhs.simplify()
        if type(lhs) == Constant and type(rhs) == Constant:
            return Constant(lhs.value() - rhs.value())
        return Diff(lhs, rhs)

    def derivative(self, var='x'):
        return Diff(self.lhs_.derivative(var), self.rhs_.derivative(var))

    def eval(self, x_in=None):
        return Constant(self.lhs_.eval(x_in) - self.rhs_.eval(x_in))


class Prod(Constant):
    def __init__(self, lhs, rhs):
        self.lhs_ = lhs#.simplify()
        self.rhs_ = rhs#.simplify()

    def value(self):
        if type(self.lhs_) == Constant and type(self.rhs_) == Constant:
            return Constant(self.lhs_.value() * self.rhs_.value())
        if type(self.lhs_) == Constant:
            return Prod(self.lhs_, self.rhs_.value())
        if type(self.rhs_) == Constant:
            return Prod(self.lhs_.value(), self.rhs_)
        return Prod(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' * ' + str(self.rhs_) + ')'

    def set_param(self, k_in):
        return Prod(self.lhs_.set_param(k_in).simplify(), self.rhs_.set_param(k_in).simplify())

    def simplify(self):
        if self.lhs_.value() == 0 or self.rhs_.value() == 0:
            return Constant(0)
        if self.lhs_ == self.rhs_:
            return Prod(self.lhs_.symbol_, Constant(2))
        if self.lhs_ == 1:
            return self.rhs_
        if self.rhs_ == 1:
            return self.lhs_
        lhs = self.lhs_.simplify()
        rhs = self.rhs_.simplify()
        if type(lhs) == Constant and type(rhs) == Constant:
            return Constant(lhs.value() * rhs.value())
        return Prod(self.lhs_.simplify(), self.rhs_.simplify())

    def ln(self):
        return Sum(self.lhs_.ln(), self.rhs_.ln())

    def derivative(self, var='x'):
        if type(self.lhs_) == Constant:
            return Prod(self.lhs_, self.rhs_.derivative(var))
        if type(self.rhs_) == Constant:
            return Prod(self.lhs_.derivative(var), self.rhs_)
        return Sum(Prod(self.lhs_.derivative(var), self.rhs_),
                   Prod(self.lhs_, self.rhs_.derivative(var)))

    def eval(self, x_in=None):
        return Constant(self.lhs_.simplify().eval(x_in) * self.rhs_.simplify().eval(x_in))


class Quotient(Constant):
    def __init__(self, lhs, rhs):
        if lhs.value() == 0:
            self.lhs_ = Constant(0)
            self.rhs_ = Constant(1)
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
        if type(self.lhs_) == Constant:
            return Quotient(self.lhs_, self.rhs_.value())
        if type(self.rhs_) == Constant:
            return Quotient(self.lhs_.value(), self.rhs_)
        return Quotient(self.lhs_.value(), self.rhs_.value())

    def __str__(self):
        return '(' + str(self.lhs_) + ' / ' + str(self.rhs_) + ')'

    def set_param(self, k_in):
        return Quotient(self.lhs_.set_param(k_in), self.rhs_.set_param(k_in))

    def simplify(self):
        if self.lhs_ == self.rhs_:
            return Constant(1)
        if self.lhs_.value() == 1:
            return Power(self.rhs_, Constant(-1))
            return Constant(1)/self.rhs_
        if self.rhs_.value() == 1:
            return Power(self.lhs_, Constant(-1))
            return Constant(1)/self.lhs_
        lhs = self.lhs_.simplify()
        rhs = self.rhs_.simplify()
        if type(lhs) == Constant and type(rhs) == Constant:
            return Constant(lhs.value() / rhs.value())
        return Quotient(self.lhs_.simplify(), self.rhs_.simplify())

    def ln(self):
        return Diff(self.lhs_.ln(), self.rhs_.ln())

    def derivative(self, var='x'):
        if isinstance(type(self.rhs_.value()), Constant) or type(self.rhs_.value()) == Constant or type(self.rhs_) == Parameter:
            return Quotient(self.lhs_.derivative(var), self.rhs_)
        if type(self.rhs_) == Variable:
            return Diff(Quotient(self.lhs_.derivative(var), self.rhs_),
                       Quotient(Prod(self.lhs_, self.rhs_.derivative(var)), Power(self.rhs_, Constant(2))))
        return Diff(Quotient(self.lhs_.derivative(var), self.rhs_),
                   Quotient(Prod(self.lhs_, self.rhs_.derivative(var)), self.rhs_))

    def eval(self, x_in=None):
        if type(self.lhs_) == Constant:
            return Constant(self.lhs_ / self.rhs_.eval(x_in))
        if type(self.rhs_) == Constant:
            return Constant(self.lhs_.eval(x_in) / self.rhs_.value())
        return Constant(self.lhs_.eval(x_in) / self.rhs_.eval(x_in))


class Variable(Constant):
    def __init__(self, symbol='x'):
        self.symbol_ = symbol

    def __eq__(self, rhs):
        if isinstance(rhs, Variable):
            return self.symbol_ == rhs.symbol_
        return False

    def __str__(self):
        return str(self.symbol_)

    def value(self):
        return self

    def ln(self):
        return LogVariable(self.symbol_)

    def derivative(self, var='x'):
        if var == self.symbol_:
            return Constant(1)
        return Constant(0)

    def eval(self, x_in):
        return Constant(x_in)


class LogVariable(Variable):
    def __init__(self, symbol='x'):
        super().__init__(symbol)

    def __str__(self):
        return 'ln(' + str(self.symbol_) + ')'

    def derivative(self, var='x'):
        return Quotient(Constant(1), Variable(self.symbol_))


class Power(Variable):
    def __init__(self, symbol='x', Power=1):
        if not issubclass(type(Power), Constant):
            self.val_ = Constant(Power)
        else:
            self.val_ = Power
        if isinstance(symbol, Constant):
            self.symbol_ = symbol
        else:
            self.symbol_ = Variable(symbol)

    def __eq__(self, rhs):
        if isinstance(rhs, Power) and self.val_ == rhs.val_:
            return self.symbol_ == rhs.symbol_
        return False

    def __str__(self):
        if self.val_.value() == -1:
            return '(1 / ' + str(self.symbol_) + ')'
        return '(' + str(self.symbol_) + '^' + str(self.val_) + ')'

    def simplify(self):
        if self.val_.value() == 0:
            return Constant(1)
        if self.val_.value() == 1:
            return Variable(self.symbol_)
        return self

    def set_param(self, k_in):
        return Power(self.symbol_.set_param(k_in), self.val_.set_param(k_in))

    def ln(self):
        return self.val_ * self.symbol_.ln()

    def derivative(self, var='x'):
        if self.val_.value() == 1:
            return self.symbol_.derivative(var)
        if self.val_.value() == 0:
            return Constant(0)
        return Prod(Prod(self.val_, Power(self.symbol_, self.val_-1)),
                   self.symbol_.derivative(var))

    def eval(self, x_in):
        if self.symbol_.eval(x_in).value() == 0:
            return Constant(0)
        return Constant(self.symbol_.eval(x_in).value()**self.val_.eval(x_in).value())


class Polynomial(Constant):
    def __init__(self, symbol='x', const=[2, 1, 1], order=2):
        self.symbol_ = symbol
        if len(const) != order+1:
            raise Exception('Please specify enough constants!')
        #self.val_ = Prod(Constant(const[0]), Power(symbol, order))
        #for i in range(1, order):
        #    self.val_ = Sum(self.val_, Prod(Constant(const[i]), Power(symbol, order-i)))
        #self.val_ = Sum(self.val_, Constant(const[-1]))
        #self.val_ = self.val_.simplify()
        self.val_ = Prod(Constant(const[0]), Power(symbol, order))
        for i in range(1, order+1):
            self.val_ = Sum(self.val_, Prod(Constant(const[i]), Power(symbol, order-i)))
            #self.val_ = self.val_.simplify()

    def __str__(self):
        return str(self.val_)

    def ln(self):
        return LogPolynomial(self)

    def derivative(self, var='x'):
        return self.val_.derivative(var)

    def eval(self, x_in):
        return Constant(self.val_.eval(x_in))


class LogPolynomial(Polynomial):
    def __init__(self, var):
        self.symbol_ = var.symbol_
        self.val_ = var.val_

    def __str__(self):
        return 'ln(' + str(self.val_) + ')'

    def derivative(self, var='x'):
        return Quotient(self.val_.derivative(var), self.val_)

    def eval(self, x_in):
        return Constant(np.log(self.val_.eval(x_in).value()))


class Expo(Constant):
    def __init__(self, var):
        self.val_ = var

    def __str__(self):
        return 'e^' + str(self.val_)

    def ln(self):
        return self.val_

    def derivative(self, var='x'):
        return Prod(self.val_.derivative(var), self)

    def eval(self, x_in):
        return Constant(np.exp(self.val_.eval(x_in).value()))


class Poisson(Prod):
    def __init__(self, symbol='x', param=1):
        self.symbol_ = symbol
        self.var_ = Prod(Constant(-1), Variable(self.symbol_))
        self.lhs_ = Power(symbol, Parameter(param))
        self.rhs_ = Quotient(Expo(self.var_), Factorial(param))
        self.param_ = param

    def ln(self):
        return LogPoisson(Sum(self.lhs_.ln(), self.rhs_.ln()), self.param_)

    def derivative(self, var='x'):
        return DerivativePoisson(Prod(self.lhs_, self.rhs_).derivative(var), self.param_)

    def eval(self, x_in, k_in):
        var = Constant(-1) * Variable(k_in)
        lhs = Power(self.symbol_, Constant(k_in)).eval(x_in)
        rhs = Expo(var).eval(x_in) / Factorial(k_in).eval(k_in)
        return Constant(lhs * rhs)


class LogPoisson(Sum):
    def __init__(self, Pois_l, k_in):
        self.lhs_ = Pois_l.lhs_
        self.rhs_ = Pois_l.rhs_
        self.param_k_ = k_in

    def derivative(self, var='x'):
        return DerivativePoisson(Sum(self.lhs_.derivative(var),
                   self.rhs_.derivative(var)), self.param_k_)

    def eval(self, x_in, k_in):
        lhs_ = self.lhs_.set_param(k_in)
        rhs_ = self.rhs_.set_param(k_in)
        return Constant(lhs_.eval(x_in) + rhs_.eval(x_in))

    def minimize_nll(self, var, x_in, k_in, nll_sigma=0.5, iterations=1000, epsilon=1e-8, rate=1e-1,
                    temperature=None, debug=False, doError=False, doHess=False):

        def hessian(derivative, var, minimum, data):
            hess = derivative.derivative(var) 
            print(hess)
            print('hess', hess.eval(minimum, data).value())
            return np.sqrt(-1 * hess.eval(minimum, data).value())

        derivative = self.derivative(var)
        grad = Constant(0)
        minimum = Constant(x_in)
        in_rate = rate
        for istep in range(iterations):
            grad = derivative.eval(minimum, k_in)
            min_val = derivative.eval(minimum, k_in)
            minimum = minimum + grad * rate
            if debug:
                print('istep=', istep, 'minimum=', minimum, 'min_val=', min_val,
                    'grad=', grad, 'grad*rate==', grad*rate,
                    'minimum-grad*rate==', minimum-grad*rate)
            if abs(min_val.value()) < epsilon or abs(grad.value()) < epsilon:
                if not doError:
                    return minimum
                if doHess:
                    return (minimum, hessian(derivative, var, minimum, k_in))
                return (minimum, self.error(var, minimum, k_in, nll_sigma, iterations, epsilon, rate,
                    temperature, debug))
            if temperature is not None:
                decay = np.exp(-1*istep/temperature) # cooling
                rate = max(rate * decay, in_rate*.001)
                if debug: print('rate after decay', rate, 'decay', decay)
        if not doError:
            return minimum
        if doHess:
            return (minimum, hessian(derivative, var, minimum, k_in))
        return (minimum, self.error(var, minimum, k_in, nll_sigma, iterations, epsilon, rate,
            temperature, debug))

    def error(self, var, true_min, k_in, nll_sigma=0.5, iterations=1000, epsilon=1e-8, rate=1e-1,
                    temperature=None, debug=False, doError=False):
        grad = Constant(1)
        target = self.eval(true_min, k_in) - nll_sigma * 2
        print(self.derivative(var))
        minimum = true_min + np.sqrt(true_min.value()) * nll_sigma * 2 # Poisson guess
        in_rate = rate
        for istep in range(iterations):
            grad = target - self.eval(minimum, k_in)
            min_val = self.eval(minimum, k_in)
            if debug:
                print('min_val', min_val, 'target', target)
                print('istep=', istep, 'minimum=', minimum, 'Delta min_val=', target - min_val,
                    'nll_sigma=', nll_sigma,
                    'target=', target, 'min_val=', min_val,
                    'grad=', grad, 'grad*rate==', grad*rate,
                    'minimum-grad*rate==', minimum-grad*rate)
                print('diff', target.value() - min_val.value(), grad.value()-nll_sigma)
            if abs(target.value() - min_val.value()) < epsilon:
            #if abs(target.value() - min_val.value() - nll_sigma) < epsilon or grad.value() < epsilon:
                return np.sqrt(minimum.value())
            #minimum = minimum + Constant(1)/grad# * rate# / true_min
            print('direction', (min_val - target) / np.abs(target.value() - min_val.value()))
            print('direction/grad', (min_val - target) / np.abs(target.value() - min_val.value()) / grad)
            print('min + direction/grad', minimum + (min_val - target) / np.abs(target.value() - min_val.value()) / grad)
            #minimum = minimum + (min_val - target) / np.abs(target.value() - min_val.value()) / grad
            minimum = minimum - grad
            print('minimum is now', minimum)
            if temperature is not None:
                decay = np.exp(-1*istep/temperature) # cooling
                rate = max(rate * decay, in_rate*.001)
                if debug: print('rate after decay', rate, 'decay', decay)
        return np.sqrt(minimum.value())


class DerivativePoisson(Sum):
    def __init__(self, Pois_d, k_in):
        self.lhs_ = Pois_d.lhs_
        self.rhs_ = Pois_d.rhs_
        self.param_k_ = k_in

    def derivative(self, var):
        return DerivativePoisson(Sum(self.lhs_.derivative(var), self.rhs_.derivative(var)), self.param_k_)

    def eval(self, x_in, k_in):
        lhs_ = self.lhs_.set_param(k_in)
        rhs_ = self.rhs_.set_param(k_in)
        return Constant(lhs_.eval(x_in) + rhs_.eval(x_in))


class EFTPoisson(Poisson):
    def __init__(self, poly=Polynomial(symbol='x', const=[1, 1, 1], order=2), param=1):
        self.param_ = param
        self.var_ = poly
        self.lhs_ = Power(self.var_, Parameter(param))
        self.rhs_ = Quotient(Expo(Prod(Constant(-1), self.var_)), Factorial(param))

    def eval(self, x_in, k_in):
        var = self.var_.eval(x_in)
        lhs = Power(var, Constant(k_in)).eval(x_in)
        rhs = Expo(Prod(Constant(-1), var)).eval(x_in) / Factorial(k_in).eval(k_in)
        return Constant(lhs * rhs)


if __name__ == '__main__':
    pass
