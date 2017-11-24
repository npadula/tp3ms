import scipy
import time
from scipy import linalg as scplg
import numpy as np
import copy
import matplotlib.pyplot as plt

                        # a) Funcion make_sys
#==============================================================================
    
def make_sys(n):
    A = np.empty([n,n])
    b = np.empty([n])
    for i in range(1,n+1):
        b[i-1] = i
        k = 1
        for j in range(i,n+1):            
            if(i==j):
                A[i-1][j-1] = 1
            else:
                A[i-1][j-1] = (k+1.0)/k**2
                A[j-1][i-1] = A[i-1][j-1]
            k+=1
    return A,b


#==============================================================================

                            # d) Precision simple
#==============================================================================

def make_sys_simple(n):
    A = np.empty([n,n],np.float32)
    b = np.empty([n],np.float32)
    for i in range(1,n+1):
        b[i-1] = i
        k = 1
        for j in range(i,n+1):            
            if(i==j):
                A[i-1][j-1] = 1
            else:
                A[i-1][j-1] = (k+1.0)/k**2
                A[j-1][i-1] = A[i-1][j-1]
            k+=1
    return A,b


#==============================================================================

            # b) Reduccion Gaussiana y sustitucion hacia atras
#==============================================================================

def solve_gauss(A,b):
    n = A.shape[0]
    for col in range(n):
        aux = (1.0/A[col,col])
        A[col,:] = A[col,:]*aux # Dividimos de entrada toda la fila por su factor diagonal para que quede en 1.
        b[col] = b[col]*aux 
        for fila in range(col+1,n): 
            b[fila] -= b[col]*A[fila,col] # Puse esta linea primero y de pronto todo funciono magicamente.
            A[fila,:] -= A[col,:]*A[fila,col]
    return solve_upper_triang(A,b) #Sust. hacia atras

def solve_upper_triang(A,b):
    n = A.shape[0]
    sol = np.zeros(n)
    for fila in range(n): #Redondeamos bien los ceros
        for col in range(fila):
            A[fila,col] = 0
    for fila in range(n):
        sol[fila] = b[fila] #Se inicializa el vector solucion
        
    for fila in range(n-1,-1,-1): #Se "despeja" cada incognita usando las anteriores
        for col in range(fila+1,n):
            sol[fila] -= A[fila,col]*sol[col]
    return sol


                            # e) Gauss-Seidel
#==============================================================================
def gauss_seidel(A, b, x,maxIter,tol):
    N = np.tril(A)
    P = A - N
    xn = x
    for i in range(maxIter):
        xn = np.dot(scplg.solve_triangular(N,np.identity(N.shape[0]),lower=True), b - np.dot(P, xn)) # Esta linea es literalmente la generalizacion que dimos en clase
        if(abs(np.linalg.norm(xn-x)) <= tol):
            break
        x = xn
    return xn
    
def gauss_seidel2(A,b,x0,maxit=100,tol=1e-10):
    A=np.asarray(A,dtype=float)
    b=np.asarray(b,dtype=float)
    x0=np.asarray(x0,dtype=float)
    dim=x0.shape[0]
    aux=A.diagonal()
    M=-np.tril(A,-1)
    N=-np.triu(A,1)
    res=[np.linalg.norm(np.dot(A,x0)-b)]
    x=x0.copy()
    it=0
    while res[it]>tol and it<maxit:
        for i in range(dim):
            x[i]=(b[i]+np.dot(M[i,:],x)+np.dot(N[i,:],x))/aux[i]
        res.append(np.linalg.norm(np.dot(A,x)-b))
        it+=1
    return x


#==============================================================================

def residuo(A,b,x):
    return np.linalg.norm(np.dot(A,x) - b, inf)


maxN = 500
ns = []

residuosG = []
residuosGS = []
residuosGS2 = []
residuosGSimple = []
tiemposG = []
tiemposGS = []
tiemposGS2 = []
tiemposGSimple = []



for i in range(2,maxN): #Mediciones
    Ag, bg = make_sys(i)
    AgSimple, bgSimple = make_sys_simple(i)
    Ags,bgs = make_sys(i)
    
    x = np.zeros_like(bgs) #Aproximacion inicial como vector nulo
    
    ticG = time.clock() #Tiempo de Gauss
    solGauss = solve_gauss(Ag,bg)
    tocG = time.clock() - ticG
    
    resG = residuo(Ag,bg,solGauss) #Residuo de Gauss
    
    ticGS = time.clock() #Tiempo de Gauss-Seidel
    solGS = gauss_seidel(Ags,bgs,x,1000,1e-15)
    tocGS = time.clock() - ticGS
    
    resGS = residuo(Ags,bgs,solGS) #Residuo de Gauss-Seidel
    
    ticGS2 = time.clock() #Tiempo de Gauss-Seidel (segunda implementacion)
    solGS2 = gauss_seidel2(Ags,bgs,x,1000,1e-15)
    tocGS2 = time.clock() - ticGS2
    
    resGS2 = residuo(Ags,bgs,solGS2) #Residuo de Gauss-Seidel (segunda implementacion)
    
    ticGSimple = time.clock()  #Tiempo de Gauss con precision simple
    solGSimple = solve_gauss(AgSimple,bgSimple)
    tocGSimple =  time.clock() - ticGSimple
    
    resGSimple = residuo(AgSimple,bgSimple,solGSimple) #Residuo de Gauss con precision simple
    
    residuosG.append(resG)
    tiemposG.append(tocG)
    
    residuosGS.append(resGS)
    tiemposGS.append(tocGS)
    
    residuosGS2.append(resGS2)
    tiemposGS2.append(tocGS2)
    
    residuosGSimple.append(resGSimple)
    tiemposGSimple.append(tocGSimple)
    
    ns.append(i)
    


#Grafico de los datos obtenidos
figure, axes = plt.subplots(2,1)
axes[0].set_xlabel('Dimension de la matriz (n)')
axes[0].set_ylabel('||Residuo|| (log10)')
axes[0].set_yscale('log',basey=10)
axes[0].plot(ns,residuosG, color='blue', label='Gauss')
axes[0].plot(ns,residuosGS, color='red', label='Gauss-Seidel')
axes[0].plot(ns,residuosGS2, color='yellow', label='Gauss-Seidel (2)')
axes[0].plot(ns,residuosGSimple, color='green', label='Gauss (precision simple)')

axes[1].set_xlabel('Dimension de la matriz (n)')
axes[1].set_ylabel('Tiempo [seg]')
axes[1].plot(ns,tiemposG, color='blue', label='Gauss')
axes[1].plot(ns,tiemposGS,color='red',label='Gauss-Seidel')
axes[1].plot(ns,tiemposGS2,color='yellow',label='Gauss-Seidel (2)')
axes[1].plot(ns,tiemposGSimple, color='green', label='Gauss (precision simple)')

plt.legend(bbox_to_anchor=(0.5, 2.45), loc=2, borderaxespad=0.)
plt.show()

    
#==============================================================================
#==============================================================================