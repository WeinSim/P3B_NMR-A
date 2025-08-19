import math

# Diese Datei enthält verschieden Klassen um arithmetische Operationen zu
# repräsentieren. Zu den Operationen zählen Addition (Add), Subtraktion (Sub),
# Multiplikation (Mult), Division (Div), Exponentiation (Pow),
# die Exponentialfunktion (Exp) Sinus (Sin), Cosinus (Cos) und der
# Zehnerlogarithmus (Log10).
# Diese Operationen bilden die Knoten der entstehenden Syntaxbäume.
# Jede Instanz besitzt ein bzw. zwei arithmetische Ausdrücke als "Kinder".
# Die Blätter bilden Konstanten (Const) und Variablen (Var).
# Konstanten besitzen einen festen Wert, während Variablen einen Wert
# und eine Unsicherheit haben.
# Ein arithmetischer Ausdruck kann mit der Funktion eval() ausgewertet werden.
# Mit derivative() wird die Ableitung nach der angegebenen Variable gebildet,
# welche wieder ein arithmetischer Ausdruck ist.
# Mit den Funktionen gaussian und minMax wird die gauß'sche Unsicherheit
# bzw. die Min-Max-Unsicherheit eines arithmetischen Ausdrucks
# mit den gegebenen Parametern berechnet.

class Add:

    def __init__(self, child1, child2):
        self.child1 = child1
        self.child2 = child2

    def eval(self):
        return self.child1.eval() + self.child2.eval()

    def derivative(self, var):
        return Add(self.child1.derivative(var), self.child2.derivative(var))

    def __str__(self):
        return toStr(self, "+")

    def isEqual(self, other):
        if not isinstance(other, Add):
            return False
        if self.child1.isEqual(other.child1) and self.child2.isEqual(other.child2):
            return True
        if self.child1.isEqual(other.child2) and self.child2.isEqual(other.child1):
            return True
        return False

    @staticmethod
    def priority():
        return 0

class Sub:

    def __init__(self, child1, child2):
        self.child1 = child1
        self.child2 = child2

    def eval(self):
        return self.child1.eval() - self.child2.eval()

    def derivative(self, var):
        return Sub(self.child1.derivative(var), self.child2.derivative(var))

    def __str__(self):
        return toStr(self, "-")

    def isEqual(self, other):
        if not isinstance(other, Sub):
            return False
        return self.child1.isEqual(other.child1) and self.child2.isEqual(other.child2)

    @staticmethod
    def priority():
        return 0

class Mult:

    def __init__(self, child1, child2):
        self.child1 = child1
        self.child2 = child2

    def eval(self):
        return self.child1.eval() * self.child2.eval()

    def derivative(self, var):
        return Add(Mult(self.child1, self.child2.derivative(var)), Mult(self.child1.derivative(var), self.child2))

    def __str__(self):
        return toStr(self, "*")

    def isEqual(self, other):
        if not isinstance(other, Mult):
            return False
        if self.child1.isEqual(other.child1) and self.child2.isEqual(other.child2):
            return True
        if self.child1.isEqual(other.child2) and self.child2.isEqual(other.child1):
            return True
        return False

    @staticmethod
    def priority():
        return 1

class Div:

    def __init__(self, child1, child2):
        self.child1 = child1
        self.child2 = child2

    def eval(self):
        c2 = self.child2.eval()
        if c2 == 0.0 or c2 == -0.0:
            return float('NaN')
        return self.child1.eval() / c2

    def derivative(self, var):
        num = Sub(Mult(self.child2, self.child1.derivative(var)), Mult(self.child2.derivative(var), self.child1))
        return Div(num, Pow(self.child2, 2))

    def __str__(self):
        return toStr(self, "/")

    def isEqual(self, other):
        if not isinstance(other, Div):
            return False
        return self.child1.isEqual(other.child1) and self.child2.isEqual(other.child2)

    @staticmethod
    def priority():
        return 1

class Pow:

    def __init__(self, child1, value):
        self.child1 = child1
        self.value = value

    def eval(self):
        return math.pow(self.child1.eval(), self.value)

    def derivative(self, var):
        return Mult(Mult(Const(self.value), Pow(self.child1, self.value - 1)), self.child1.derivative(var))

    def __str__(self):
        useParens = not (isinstance(self.child1, Var) or isinstance(self.child1, Const))
        baseStr = self.child1.__str__()
        if useParens:
            baseStr = f"({baseStr})"
        return f"{baseStr} ^ {self.value}"

    def isEqual(self, other):
        if not isinstance(other, Pow):
            return False
        return self.child1.isEqual(other.child1) and self.value == other.value

    @staticmethod
    def priority():
        return 1

class Exp:

    def __init__(self, child1):
        self.child1 = child1

    def eval(self):
        return math.exp(self.child1.eval())

    def derivative(self, var):
        return Mult(Exp(self.child1), self.child1.derivative(var))

    def __str__(self):
        c1 = self.child1.__str__()
        return f"exp({c1})"

    def isEqual(self, other):
        if not isinstance(other, Exp):
            return False
        return self.child1.isEqual(other.child1)

class Sin:

    def __init__(self, child1):
        self.child1 = child1

    def eval(self):
        return math.sin(self.child1.eval())

    def derivative(self, var):
        return Mult(Cos(self.child1), self.child1.derivative(var))

    def __str__(self):
        c1 = self.child1.__str__()
        return f"sin({c1})"

    def isEqual(self, other):
        if not isinstance(other, Sin):
            return False
        return self.child1.isEqual(other.child1)

class Cos:

    def __init__(self, child1):
        self.child1 = child1

    def eval(self):
        return math.cos(self.child1.eval())

    def derivative(self, var):
        return Mult(Mult(Const(-1), Sin(self.child1)), self.child1.derivative(var))

    def __str__(self):
        c1 = self.child1.__str__()
        return f"cos({c1})"

    def isEqual(self, other):
        if not isinstance(other, Cos):
            return False
        return self.child1.isEqual(other.child1)

class Log10:

    def __init__(self, child1):
        self.child1 = child1

    def eval(self):
        return math.log(self.child1.eval()) / math.log(10)

    def derivative(self, var):
        return Pow(Mult(self.child1, Const(math.log(10))), -1)

    def __str__(self):
        c1 = self.child1.__str__()
        return f"log_10({c1})"

    def isEqual(self, other):
        if not isinstance(other, Log10):
            return False
        return self.child1.isEqual(other.child1)

class Const:

    def __init__(self, value):
        self.value = value

    def eval(self):
        return self.value

    def derivative(self, var):
        return Const(1) if self is var else Const(0)

    def __str__(self):
        return f"{self.value}"

    @staticmethod
    def priority():
        return 3
    
    def isEqual(self, other):
        if not isinstance(other, Const):
            return False
        return self.value == other.value

class Var:

    def __init__(self, value, uncertainty, name):
        self.value = value
        self.name = name
        self.uncertainty = uncertainty

    def eval(self):
        return self.value

    def derivative(self, var):
        return Const(1) if self is var else Const(0)

    def __str__(self):
        return self.name
    
    def isEqual(self, other):
        return self is other

    @staticmethod
    def priority():
        return 3

# Hilfsfunktion zur Darstellung eines arithmetischen Ausdrucks als String.
def toStr(expr, infix):
    c1Parens = expr.child1.priority() <= expr.priority()
    c2Parens = expr.child2.priority() <= expr.priority()
    c1 = expr.child1.__str__()
    c2 = expr.child2.__str__()
    if c1Parens:
        c1 = f"({c1})"
    if c2Parens:
        c2 = f"({c2})"
    return f"{c1} {infix} {c2}"
    # return f"({self.child1.__str__()} + {self.child2.__str__()})"

# Vereinfachung eines arithmetischen Ausdrucks.
# Hauptsächlich werden Konstanten zusammengefasst bzw. entfernt
# (z.B. x + 0 = x, 1 * x = x, 0 * x = 0)
def simplify(expr):
    match expr:
        case Add() | Sub() | Mult() | Div():
            expr.child1 = simplify(expr.child1)
            expr.child2 = simplify(expr.child2)
            if isinstance(expr.child1, Const) and isinstance(expr.child2, Const):
                return Const(expr.eval())
        case Pow() | Sin() | Cos() | Log10():
            expr.child1 = simplify(expr.child1)
            if isinstance(expr.child1, Const):
                return Const(expr.eval())

    match expr:
        case Add():
            if isinstance(expr.child1, Const):
                if expr.child1.value == 0:
                    return expr.child2
            if isinstance(expr.child2, Const):
                if expr.child2.value == 0:
                    return expr.child1
                if expr.child2.value < 0:
                    return Sub(expr.child1, Const(-expr.child2.value))
            if expr.child1.isEqual(expr.child2):
                return Mult(Const(2), expr.child1)
        case Sub():
            if isinstance(expr.child1, Const):
                if expr.child1.value == 0:
                    return Mult(Const(-1), expr.child2)
            if isinstance(expr.child2, Const):
                if expr.child2.value == 0:
                    return expr.child1
                if expr.child2.value < 0:
                    return Add(expr.child1, Const(-expr.child2.value))
            if expr.child1.isEqual(expr.child2):
                return Const(0)
        case Mult():
            if isinstance(expr.child1, Const):
                if expr.child1.value == 0:
                    return Const(0)
                if expr.child1.value == 1:
                    return expr.child2
            if isinstance(expr.child2, Const):
                if expr.child2.value == 0:
                    return Const(0)
                if expr.child2.value == 1:
                    return expr.child1
            if expr.child1.isEqual(expr.child2):
                return Pow(expr.child1, 2)
        case Div():
            if isinstance(expr.child1, Const):
                if expr.child1.value == 0:
                    return Const(0)
                if expr.child1.value == 1:
                    return Pow(expr.child2, -1)
            if isinstance(expr.child2, Const):
                if expr.child2.value == 1:
                    return expr.child1
            if expr.child1.isEqual(expr.child2):
                return Const(1)
        case Pow():
            if expr.value == 0:
                return Const(1)
            if expr.value == 1:
                return expr.child1
    return expr

# Berechnung der gauß'schen Unsicherheit des Ausdrucks expr mit den Variablen
# params
def gaussian(expr, params):
    total = 0
    for param in params:
        if type(param) == Const:
            continue
        der = simplify(expr.derivative(param))
        derVal = der.eval()
        delta = derVal * param.uncertainty
        total += delta ** 2
    return total ** 0.5

# Berechnung der Unsicherheit des Ausdrucks expr mit der Min-Max-Methode
# und den Variablen params
def minMax(expr, params):
    minVal = expr.eval()
    maxVal = expr.eval()
    incDec = [False] * len(params)
    originalValues = []
    for param in params:
        originalValues.append(param.value)
    for j in range(2 ** len(params)):
        for i in range(len(incDec)):
            incDec[i] = not incDec[i]
            if incDec[i]:
                break
        for i in range(len(params)):
            sign = 1 if incDec[i] else -1
            params[i].value = originalValues[i] + sign * params[i].uncertainty
        newVal = expr.eval()
        minVal = min(minVal, newVal)
        maxVal = max(maxVal, newVal)
    for i in range(len(params)):
        params[i].value = originalValues[i]
    return (maxVal - minVal) / 2
