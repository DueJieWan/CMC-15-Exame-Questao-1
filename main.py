import numpy as np
import random

#Funcao do item a)
def fa(x:np.array):
    return pow(x[0],2) + 2*pow(x[1],3) + pow(x[2],2)

#Gradiente da funcao do item a)
def gradfa(x:np.array):
    return np.array([2*x[0], 4*x[1], 2*x[2]]) 

#Funcao auxiliar para calcular modulo de um vetor
def modulo(x:np.array):
    sum = np.sum(np.square(x))
    return np.sqrt(sum)

#Funcao auxiliar para calcular modulo ao quadrado de um vetor
def modulo2(x:np.array):
    sum = np.sum(np.square(x))
    return sum

#Funcao auxiliar para calcular Alpha apartir da funcao gradiente
def computAlpha(gradient, xn:np.array,xn_1:np.array):
    gradDif = gradient(xn) - gradient(xn_1)
    num = modulo(np.dot(xn-xn_1, gradDif))
    den = modulo2(gradDif)
    return num/den
    
#Funcao que computa o metodo de Gradiente modificado
def gradientDescent(gradient, start:np.array, n_iter:int, tolerance:int = 1e-06):
    vector = start
    print(vector)
    alpha = 1/gradient(start)
    for _ in range(n_iter):
        tmp = vector
        diff = -alpha*gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector = vector + diff
        print(vector)
        alpha = computAlpha(gradient, vector, tmp)
    return vector
    

#Funcao que computa o metodo de Newton modificado
def newtonDescent(gradient, H, start:np.array, n_iter:int, tolerance:int = 1e-06):
    vector = start
    print(vector)
    alpha = 1/2
    for _ in range(n_iter):
        h = np.linalg.solve(H, -gradient(vector))
        if np.all(np.abs(h) <= tolerance):
            break
        vector = vector + h*alpha
        print(vector)
    return vector



def main():
    x0 = np.array([80,-80,80])

    minimo = gradientDescent(gradfa, x0, 200)
    print('O minimo achado por Descida de Gradiente modificado:')
    print(minimo)
    print('\n')
    
    H = [[2, 0, 0], [0, 4, 0], [0, 0, 2]]

    minimo2 = newtonDescent(gradfa, H , x0, 200)
    print('O minimo achado por Descida de Gradiente Newton modificado:')
    print(minimo2)

main()