'''
EFT Likelihood fit
Brent R. Yates
The Ohio State University
Department of Physics
Oct. 8 2022
'''

import warnings
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

    def gradient(self, var=[]):
        grad = self.derivative(var[0])
        for variable in var[1:]:
            grad = Sum(grad, self.derivative(variable))
        return grad

    def eval(self, **kwargs):
        return self.simplify()


class Pi(Constant):
    def __init__(self):
        super().__init__(Constant(np.math.pi))

    def __str__(self):
        return 'pi'

    def ln(self):
        return LogPi()

    def eval(self, **kwargs):
        return self.value()


class LogPi(Pi):
    def __init__(self):
        self.val_ = Constant(np.log(np.math.pi))

    def __str__(self):
        return 'ln(pi)'


class Log(Constant):
    def __init__(self, val):
        super().__init__(val)

    def __str__(self):
        return 'ln(' + str(self.value()) + ')'

    def eval(self, **kwargs):
        if not issubclass(type(self.value()), Constant):
            return Constant(np.log(self.value()))
        if 'x_in' not in kwargs:
            raise Exception(f'Please provide k_in!')
        return Constant(np.log(self.value().eval(**kwargs).value()))


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

    def eval(self, **kwargs):
        if 'k_in' not in kwargs:
            raise Exception(f'Please provide {self.symbol_}!')
        k_in = kwargs['k_in']
        return Constant(np.log(np.math.factorial(k_in)))


class Factorial(Parameter):
    def __str__(self):
        return self.param_ + '!'

    def set_param(self, k_in):
        return Constant(np.math.factorial(k_in))

    def ln(self):
        return LogFactorial(self.param_)

    def eval(self, **kwargs):
        if 'k_in' not in kwargs:
            raise Exception(f'Please provide {self.symbol_}!')
        k_in = kwargs['k_in']
        return Constant(np.math.factorial(k_in))


class LogFactorial(Factorial):
    def __init__(self, param):
        super().__init__(param)

    def __str__(self):
        return 'ln(' + str(self.param_) + '!)'

    def set_param(self, k_in):
        return Constant(np.log(float(np.math.factorial(k_in))))

    def eval(self, **kwargs):
        if 'k_in' not in kwargs:
            raise Exception(f'Please provide {self.symbol_}!')
        k_in = kwargs['k_in']
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

    def eval(self, **kwargs):
        return Constant(self.lhs_.simplify().eval(**kwargs) + self.rhs_.simplify().eval(**kwargs))


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

    def eval(self, **kwargs):
        return Constant(self.lhs_.eval(**kwargs) - self.rhs_.eval(**kwargs))


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

    def eval(self, **kwargs):
        return Constant(self.lhs_.simplify().eval(**kwargs) * self.rhs_.simplify().eval(**kwargs))


class Quotient(Constant):
    def __init__(self, lhs, rhs):
        if type(lhs) == Constant and lhs.value() == 0:
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
        #(f/g)' = f'/g - g'*f/g^2
        return Diff(Quotient(self.lhs_.derivative(var), self.rhs_),
                   Quotient(Prod(self.lhs_, self.rhs_.derivative(var)), Power(self.rhs_, 2)))

    def eval(self, **kwargs):
        if type(self.lhs_) == Constant:
            return Constant(self.lhs_ / self.rhs_.eval(**kwargs))
        if type(self.rhs_) == Constant:
            return Constant(self.lhs_.eval(**kwargs) / self.rhs_.value())
        return Constant(self.lhs_.eval(**kwargs) / self.rhs_.eval(**kwargs))


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

    def eval(self, **kwargs):
        if isinstance(self.symbol_, str):
            if self.symbol_ + '_in' not in kwargs:
                raise Exception(f'please provide {self.symbol_}!')
            x_in = kwargs[self.symbol_ + '_in']
            return Constant(x_in)
        return self.symbol_.eval(**kwargs)

    def udate(self, delta, **kwargs):
        pass


class LogVariable(Variable):
    def __init__(self, symbol='x'):
        super().__init__(symbol)

    def __str__(self):
        return 'ln(' + str(self.symbol_) + ')'

    def ln(self):
        return LogVariable(self)

    def derivative(self, var='x'):
        if var == self.symbol_:
            return Quotient(Constant(1), Variable(self.symbol_))
        return Constant(0)

    def eval(self, **kwargs):
        if isinstance(self.symbol_, str):
            if self.symbol_ + '_in' not in kwargs:
                raise Exception(f'please provide {self.symbol_}!')
            x_in = Constant(kwargs[self.symbol_ + '_in']).value()
            return Constant(np.log(x_in))
        return Constant(np.log(self.symbol_.eval(**kwargs).value()))


class Power(Variable):
    def __init__(self, symbol='x', Power=1):
        if not issubclass(type(Power), Constant):
            self.val_ = Constant(Power)
        else:
            self.val_ = Power
        if isinstance(symbol, Constant):
            self.symbol_ = symbol
        elif isinstance(symbol, str):
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

    def raise_power_by_one(self):
        return Power(self.symbol_, self.val_+1)

    def ln(self):
        return self.val_ * self.symbol_.ln()

    def derivative(self, var='x'):
        if self.val_.value() == 1:
            return self.symbol_.derivative(var)
        if self.val_.value() == 0:
            return Constant(0)
        return Prod(Prod(self.val_, Power(self.symbol_, self.val_-1)),
                   self.symbol_.derivative(var))

    def eval(self, **kwargs):
        if self.symbol_.eval(**kwargs).value() == 0:
            return Constant(0)
        return Constant(self.symbol_.eval(**kwargs).value()**self.val_.eval(**kwargs).value())


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

    def eval(self, **kwargs):
        return Constant(self.val_.eval(**kwargs))


class LogPolynomial(Polynomial):
    def __init__(self, var):
        self.symbol_ = var.symbol_
        self.val_ = var.val_

    def __str__(self):
        return 'ln(' + str(self.val_) + ')'

    def derivative(self, var='x'):
        return Quotient(self.val_.derivative(var), self.val_)

    def eval(self, **kwargs):
        return Constant(np.log(self.val_.eval(**kwargs).value()))


class Expo(Constant):
    def __init__(self, var):
        self.val_ = var

    def __str__(self):
        return 'e^' + str(self.val_)

    def ln(self):
        return self.val_

    def derivative(self, var='x'):
        return Prod(self.val_.derivative(var), self)

    def eval(self, **kwargs):
        return Constant(np.exp(self.val_.eval(**kwargs).value()))


class Poisson(Prod):
    def __init__(self, symbol='x', param=1):
        self.symbol_ = symbol
        self.var_ = Prod(Constant(-1), Variable(self.symbol_))
        self.lhs_ = Power(symbol, Parameter(param))
        self.rhs_ = Quotient(Expo(self.var_), Factorial(param))
        self.param_ = param

    def ln(self):
        return LogPoisson(Sum(self.lhs_.ln(), self.rhs_.ln()), self.symbol_, self.param_)

    def derivative(self, var='x'):
        return DerivativePoisson(Prod(self.lhs_, self.rhs_).derivative(var),
                   self.symbol_, self.param_)

    def eval(self, **kwargs):
        if self.symbol_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.symbol_}!')
        var = Prod(Constant(-1), Variable(self.var_))
        if self.param_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.param_}!')
        k_in = kwargs[self.param_ + '_in']
        x_in = kwargs[self.symbol_ + '_in']
        lhs = Power(self.symbol_, Constant(k_in)).eval(**kwargs)
        rhs = Expo(self.var_).eval(**kwargs) / Factorial(k_in).eval(k_in=k_in)
        return Constant(lhs * rhs)


class LogPoisson(Sum):
    def __init__(self, Pois_l, symbol, k_in):
        self.symbol_ = symbol
        self.lhs_ = Pois_l.lhs_
        self.rhs_ = Pois_l.rhs_
        self.param_k_ = k_in

    def derivative(self, var='x'):
        return DerivativePoisson(Sum(self.lhs_.derivative(var),
                   self.rhs_.derivative(var)), self.symbol_, self.param_k_)

    def gradient(self, var=[]):
        grad = self.derivative(var[0])
        for variable in var[1:]:
            grad = Sum(grad, self.derivative(variable))
        return LogPoisson(grad, self.symbol_, self.param_k_)

    def eval(self, **kwargs):
        if self.symbol_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.symbol_}!')
        x_in = kwargs['x_in']
        if self.param_k_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.param_k_}!')
        k_in = kwargs[self.param_k_ + '_in']
        lhs_ = self.lhs_.set_param(k_in)
        rhs_ = self.rhs_.set_param(k_in)
        return Constant(lhs_.eval(x_in=x_in) + rhs_.eval(x_in=x_in))


class DerivativePoisson(Sum):
    def __init__(self, Pois_d, symbol, k_in):
        self.symbol_ = symbol
        self.lhs_ = Pois_d.lhs_
        self.rhs_ = Pois_d.rhs_
        self.param_k_ = k_in

    def derivative(self, var):
        return DerivativePoisson(Sum(self.lhs_.derivative(var), self.rhs_.derivative(var)),
                   self.symbol_, self.param_k_)

    def eval(self, **kwargs):
        if self.symbol_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.symbol_}!')
        x_in = kwargs[self.symbol_ + '_in']
        if self.param_k_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.param_k_}!')
        k_in = kwargs[self.param_k_ + '_in']
        lhs_ = self.lhs_.set_param(k_in)
        rhs_ = self.rhs_.set_param(k_in)
        return Constant(lhs_.eval(x_in=x_in) + rhs_.eval(x_in=x_in))


class EFTPoisson(Poisson):
    def __init__(self, poly=Polynomial(symbol='x', const=[1, 1, 1], order=2), param=1):
        self.symbol_ = poly.symbol_
        self.param_ = param
        self.var_ = poly
        self.lhs_ = Power(self.var_, Parameter(param))
        self.rhs_ = Quotient(Expo(Prod(Constant(-1), self.var_)), Factorial(param))

    def eval(self, **kwargs):
        if self.symbol_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.symbol_}!')
        x_in = kwargs[self.symbol_ + '_in']
        if self.param_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.param_}!')
        k_in = kwargs[self.param_ + '_in']
        var = self.var_.eval(x_in=x_in)
        lhs = Power(var, Constant(k_in)).eval(x_in=x_in)
        rhs = Expo(Prod(Constant(-1), var)).eval(x_in=x_in) / Factorial(k_in).eval(k_in=k_in)
        return Constant(lhs * rhs)


class LogLikelohood:
    def __init__(self, dist=Variable('x'), nuis=None, var='x', nuis_var=[]):
        if nuis is not None:
            self.log_likelihood_ = Sum(dist.ln(), nuis.ln())
        self.log_likelihood_ = dist.ln()
        if nuis is not None:
            self.nuis_ = nuis.ln()
        else:
            self.nuis_ = Constant(0)
        if not isinstance(var, list):
            var = [var]
        self.var_ = var
        self.nuis_var_ = nuis_var

    def minimize_nll(self, var='x', nll_sigma=0.5, iterations=1000, epsilon=1e-8, rate=1e-1,
                    temperature=None, debug=False, doError=False, doHess=False, **kwargs):

        def hessian(derivative, var, minimum, **data):
            tmp_kwargs = self.var_.copy()
            tmp_kwargs.append('k')
            if (nll_sigma - 0.5) > 1e-18:
                warnings.warn('Asked for Hessian at a value other than 2*deltaNLL=1,\
                               answer WILL be wrong')
            hess_matrix = {}
            for x in self.var_:
                hess = derivative.gradient(x)
                hess_matrix[x] = Constant(1/np.sqrt(-1 * hess.eval(**data).value()))
            hess_matrix['kappa'] = self.nuis_.best_kappa_.value()
            return hess_matrix

        derivative = self.log_likelihood_.gradient(self.var_)
        d_nuis = self.nuis_.gradient(self.var_ + self.nuis_var_)
        grad = [Constant(0)] * len(self.var_ + self.nuis_var_)
        minimum = {x.replace('_in', ''): Constant(kwargs[x]) for x in kwargs if '_in' in x}
        min_val = [Constant(0) for x in self.var_ + self.nuis_var_]
        in_rate = rate
        rate = Constant(rate)
        tmp_kwargs = kwargs.copy()
        for istep in range(iterations):
            for ix,x in enumerate(self.var_ + self.nuis_var_):
                tmp_kwargs[x + '_in'] = minimum[x].value()
                tmp_kwargs['symbol'] = var
                grad[ix] = derivative.eval(**tmp_kwargs) + d_nuis.eval(**tmp_kwargs)
                min_val[ix] = derivative.eval(**tmp_kwargs)
                minimum[x] = minimum[x] + grad[ix] * rate
                if isinstance(self.nuis_, Variable):
                    d_nuis.update(**{x: grad[ix] * rate})
                    #self.nuis_.update(**{x: grad[ix] * rate})
            global_min_val = Constant(0)
            global_grad = Constant(0)
            #for g in min_val:
            #    global_min_val += g.value()
            tmp_kwargs = kwargs.copy()
            for ix,x in enumerate(self.var_ + self.nuis_var_):
                tmp_kwargs[x + '_in'] = minimum[x]
            global_min_val = derivative.eval(**tmp_kwargs)
            for g in grad:
                global_grad += g.value()
            if debug:
                print('istep=', istep, 'minimum=', minimum[x], 'min_val=', global_min_val,
                    'grad=', global_grad, 'grad*rate==', global_grad*rate,
                    'minimum-grad*rate==', minimum[x]-global_grad*rate)
            if abs(global_min_val.value()) < epsilon or abs(global_grad.value()) < epsilon:
                if not doError:
                    return minimum
                if doHess:
                    return (minimum, hessian(derivative, var, minimum, **kwargs))
                return (minimum, self.error(var, minimum, nll_sigma, iterations, epsilon, rate,
                    temperature, debug, **kwargs))
            if temperature is not None:
                decay = np.exp(-1*istep/temperature) # cooling
                rate = max(rate * decay, in_rate*.001)
                if debug:
                    print('rate after decay', rate, 'decay', decay)
        if not doError:
            return minimum
        if doHess:
            return (minimum, hessian(derivative, var, minimum, **kwargs))
        return (minimum, self.error(var, minimum, nll_sigma, iterations, epsilon, rate,
            temperature, debug, **kwargs))

    def error(self, var='x', true_min=1, nll_sigma=0.5, iterations=1000, epsilon=1e-8, rate=1e-1,
                    temperature=None, debug=False, doError=False, **kwargs):
        #var = kwargs['var']
        #true_min = kwargs['true_min']
        #nll_sigma = kwargs['nll_sigma']
        #iterations=1000
        #epsilon = kwargs['epsilon']
        #rate = kwargs['rate']
        #temperature = kwargs['temperature']
        #debug = kwargs['debug']
        #doError = kwargs['doError']

        grad = [Constant(0)] * len(self.var_ + self.nuis_var_)
        min_val = [Constant(0) for x in self.var_ + self.nuis_var_]
        tmp_kwargs = kwargs.copy()
        minimum = {}
        for x in true_min:
            tmp_kwargs[x + '_in'] = true_min[x].value()
            minimum[x] = Constant(0)
        target = self.log_likelihood_.eval(**tmp_kwargs) - nll_sigma * 2
        for x in minimum:
            minimum[x] = Constant(true_min[x].value() +\
                                  np.sqrt(true_min[x].value()) * nll_sigma * 2) # Poisson guess
        in_rate = rate
        rate = Constant(rate)
        for istep in range(iterations):
            for ix,x in enumerate(self.var_ + self.nuis_var_):
                tmp_kwargs = kwargs.copy()
                tmp_kwargs[x + '_in'] = minimum[x].value()
                grad[ix] = target - self.log_likelihood_.eval(**tmp_kwargs) +\
                           self.nuis_.eval(**tmp_kwargs)
                min_val[ix] = self.log_likelihood_.eval(**tmp_kwargs)
                minimum[x] = Constant(minimum[x]) + grad[ix] * rate
            global_min_val = Constant(0)
            global_grad = Constant(0)
            for g in min_val:
                global_min_val += g.value()
            for g in grad:
                global_grad += g.value()
            if debug:
                print('min_val', global_min_val, 'target', target)
                print('istep=', istep, 'minimum=', minimum,
                    'Delta min_val=', target - global_min_val, 'nll_sigma=', nll_sigma,
                    'target=', target, 'min_val=', global_min_val,
                    'grad=', global_grad, 'grad*rate==', global_grad*rate,
                    'minimum-grad*rate==', minimum[x]-global_grad*rate)
                print('diff', target.value() - global_min_val.value(),
                      global_grad.value()-nll_sigma)
            if abs(target.value() - global_min_val.value()) < epsilon:
                minimum = {key: np.sqrt(val.value()) for key,val in minimum.items()}
                return minimum
            for ix,x in enumerate(self.var_ + self.nuis_var_):
                minimum[x] = minimum[x] - grad[ix]
                tmp_kwargs[x + '_in'] = minimum[x]
            if temperature is not None:
                decay = np.exp(-1*istep/temperature) # cooling
                rate = max(rate * decay, in_rate*.001)
                if debug:
                    print('rate after decay', rate, 'decay', decay)
        minimum = {key: np.sqrt(val.value()) for key,val in minimum.items()}
        return minimum


class LogNormal(Variable):
    def __init__(self, var='x', mu='u', kappa='kappa', mu_in=np.exp(1), kappa_in=1):
        self.symbol_ = var
        self.var_ = LogVariable(var)
        self.mu_ = mu
        self.kappa_ = kappa
        pow_var = Power(Diff(self.var_, LogVariable(self.mu_)), Constant(2))
        self.log_normal_ = Quotient(
                        Expo(
                        Quotient(
                        Prod(Constant(-1), pow_var),
                            Prod(Constant(2), Power(LogVariable(self.kappa_),
                                Constant(2))))),
                        Prod(Prod(Variable(self.symbol_), LogVariable(self.kappa_)),
                            Power(Prod(Constant(2), Pi()), Constant(1/2))))
        self.initial_mu_ = Constant(mu_in)
        self.initial_kappa_ = Constant(kappa_in)
        self.best_mu_ = Constant(mu_in)
        self.best_kappa_ = Constant(kappa_in)

    def __str__(self):
        return str(self.log_normal_)

    def ln(self):
        return LogLogNormal(self.log_normal_.ln(), self.symbol_, self.mu_, self.kappa_)

    def derivative(self, var='x'):
        return DerivativeLogNormal(self.log_normal_.derivative(var), self.symbol_,
                                   self.mu_, self.kappa_)

    def eval(self, **kwargs):
        if self.symbol_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.symbol_}!')
        x_in = kwargs[self.symbol_ + '_in']
        if self.mu_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.mu_}!')
        mu_in = kwargs[self.mu_ + '_in']
        if self.kappa_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.kappa_}!')
        kappa_in = kwargs[self.kappa_ + '_in']
        if kappa_in <= 0:
            raise Exception('kappa is negative!')
        return self.log_normal_.eval(**kwargs)

    def variance(self, **kwargs):
        if self.mu_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.mu_}!')
        mu_in = kwargs[self.mu_ + '_in']
        if self.kappa_ + '_in' not in kwargs:
            raise Exception(f'Please provide {self.kappa_}!')
        kappa_in = kwargs[self.kappa_ + '_in']
        return np.exp(2 * mu_in + np.power(kappa_in, 2)) *\
                (np.exp(np.power(kappa_in, 2) - 1))

    def update(self, **kwargs):
        if 'kappa' in kwargs:
            kappa_in = kwargs['kappa']
            self.best_kappa_ += kappa_in
        if 'mu' in kwargs:
            mu_in = kwargs['mu']
            self.best_mu_ += mu_in
        

class LogLogNormal(LogNormal):
    def __init__(self, log_norm, var, mu, kappa):
        self.symbol_ = var
        self.mu_ = mu
        self.kappa_ = kappa
        super().__init__(var, mu, kappa)
        self.log_normal_ = log_norm

    def derivative(self, var):
        return DerivativeLogNormal(self.log_normal_.derivative(var),
                   self.symbol_, self.mu_, self.kappa_)

    def gradient(self, var=[]):
        grad = self.derivative(var[0])
        for variable in var[1:]:
            grad = Sum(grad, self.derivative(variable))
        grad = DerivativeLogNormal(grad, self.symbol_, self.mu_, self.kappa_)
        return grad


class DerivativeLogNormal(LogNormal):
    def __init__(self, log_norm, var, mu, kappa):
        super().__init__(var, mu, kappa)
        self.symbol_ = var
        self.log_normal_ = log_norm

    def eval(self, **kwargs):
        return self.log_normal_.eval(**kwargs)

    def gradient(self, var=[]):
        grad = self.derivative(var[0])
        for variable in var[1:]:
            grad = Sum(grad, self.derivative(variable))
        grad = DerivativeLogNormal(grad, self.symbol_, self.mu_, self.kappa_)
        return grad


if __name__ == '__main__':
    pass
