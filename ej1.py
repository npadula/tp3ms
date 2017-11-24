import scipy
import sympy
import numpy as np
import matplotlib.pyplot as plt

def f(t): #funcion original
    pi = np.pi
    return 10*pi*((t-pi)/((4 + (t-pi)**2)*(1 + (t-pi)**2))) - (pi/10)*np.arctan(-t)
    
def f1(t): #f'(t)
    pi = np.pi
    return pi/(10*(t**2 + 1)) - (20*pi*(t-pi)**2)/((((t-pi)**2 + 1)**2)*((t-pi)**2 + 4)) - (20*pi*(t-pi)**2)/(((t-pi)**2 + 1)*(((t-pi)**2 + 4)**2)) + (10*pi)/(((t-pi)**2 + 1)*(((t-pi)**2 + 4)))
    
def f2(t): #f''(t)
    pi = np.pi
    return -(pi*t)/(5*(t**2 + 1)**2) + (80*pi*(t-pi)**3)/((((t-pi)**2 + 1)**3)*((t-pi)**2 + 4)) + (80*pi*(t-pi)**3)/((((t-pi)**2 + 1)**2)*(((t-pi)**2 + 4)**2)) + (80*pi*(t-pi)**3)/(((t-pi)**2 + 1)*(((t-pi)**2 + 4)**3)) - (60*pi*(t-pi))/((((t-pi)**2 + 1)**2)*((t-pi)**2 + 4)) - (60*pi*(t-pi))/(((t-pi)**2 + 1)*(((t-pi)**2 + 4)**2))
    
def f2symp(t): #f''(t) obtenida a partir de sympy, deberia ser correcta
    x = sympy.symbols('x')
    fs = 10*sympy.pi*((x-sympy.pi)/((4 + (x-sympy.pi)**2)*(1 + (x-sympy.pi)**2))) - (sympy.pi/10)*sympy.atan(-x)
    f1s = sympy.diff(fs,x)
    f2s = sympy.diff(f1s,x)
    
    return f2s.evalf(subs={x:t})


#==============================================================================
# N-R
#==============================================================================

def newtonRaphson(x0,f,f1,maxIter,tol):
    x = x0
    it = 0    
    for i in range(maxIter):
        it = i
        xn = x - f(x)/f1(x)
        if (abs(xn - x)/abs(x)  <= tol):
            break
        x = xn
    return xn,it

#==============================================================================
# Biseccion
#==============================================================================
def bisect(a,b,f,maxIter,tol):
    it = 0
    for i in range(maxIter):
        it = i
        c = (a+b)/2
        if (abs((b-a)/2) <= tol):
            break
        if(np.sign(f(c)) == np.sign(f(a))):
            a=c
        else:
            b=c
    return c,it

   
    
        
def hibrido(a,b,f,f1,maxIter,tol): #Biseccion y despues N-R
    x,itB = bisect(a,b,f,2,tol)
    x, itNR = newtonRaphson(x,f,f1,maxIter - 3, tol)
    itH = itNR + itB
    return x,itH
            


#==============================================================================
# Puntos criticos
#==============================================================================

t1 = -9.678268787229119019094616134681374905158
t2 = 2.366839542110999434219935561592523363221
t3 = 3.912639867208552436782492507708752127339
t4 = 23.00986120244194885454102065182218513630

#==============================================================================
# Calculado por WA
# lim t->-inf f(t) = -pi^2/20 ==> menor que cero
# lim t->inf f(t) = pi^2/20 ==> mayor que cero
#==============================================================================

t_inf = -(np.pi**2)/20
tinf = (np.pi**2)/20

ft1 = f(t1)
ft2 = f(t2)
ft3 = f(t3)
ft4 = f(t4)

if(np.sign(ft2) != np.sign(ft3)): #La unica raiz
    print('existe raiz entre {0} y {1}'.format(ft2, ft3))


#==============================================================================
# No se puede garantizar convergencia de N-R ==> Achicar el intervalo
print(np.sign(f2(t2)) == np.sign(f(t2)))
print(np.sign(f2(t3)) == np.sign(f(t3)))

# Achicando el intervalo manualmente, el punto t en que f(t)f''(t) > 0 es ~3.1

# Verificando no haber perdido la raiz:

print(np.sign(f(t2)) != np.sign(f(3.1))) # ==> True ==> hay raiz entre t2 y 3.1

# N-R desde 3.1:

raizNR, itNR = newtonRaphson(3.1,f,f1,1000,1.48e-15)
raizB, itB = bisect(t1,t4,f,1000,1.48e-15)
raizH, itH = hibrido(t2,t3,f,f1,1000,1.48e-15)


print(raizNR)
print(itNR)

print(raizB)
print(itB)

print(raizH)
print(itH)


t = np.linspace(-5,5,100000)
y = f(t)
yprima = f1(t)
ysegunda = f2(t)

figure, axes = plt.subplots(3,1)

axes[0].plot(t,y)
axes[0].set_title('f(t)')
axes[0].spines['left'].set_position('zero')
axes[0].spines['right'].set_color('none')
axes[0].spines['bottom'].set_position('zero')
axes[0].spines['top'].set_color('none')

# remove the ticks from the top and right edges
axes[0].xaxis.set_ticks_position('bottom')
axes[0].yaxis.set_ticks_position('left')

axes[1].plot(t,yprima)
axes[1].set_title('f\'(t)')
axes[1].spines['left'].set_position('zero')
axes[1].spines['right'].set_color('none')
axes[1].spines['bottom'].set_position('zero')
axes[1].spines['top'].set_color('none')

# remove the ticks from the top and right edges
axes[1].xaxis.set_ticks_position('bottom')
axes[1].yaxis.set_ticks_position('left')


axes[2].plot(t,ysegunda)
axes[2].set_title('f\'\'(t)')
axes[2].spines['left'].set_position('zero')
axes[2].spines['right'].set_color('none')
axes[2].spines['bottom'].set_position('zero')
axes[2].spines['top'].set_color('none')

# remove the ticks from the top and right edges
axes[2].xaxis.set_ticks_position('bottom')
axes[2].yaxis.set_ticks_position('left')

#==============================================================================



    
    
    