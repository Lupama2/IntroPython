#Resolución del ejercicio 08 (b) de la clase 04 del curso  Introducción al lenguaje Python orientado a ingenierías y física - Instituto Balseiro
#Pablo Chehade
#Creación: 06/03/2023
#Última modificación: 06/03/2023
#Comentarios:
#(1) en mi pc corre con python archivo.py, no con python3 archivo.py

import numpy as np

def trapz(x,y):
    '''
    Aplica la fórmula de los trapecios a los arrays unidimensionales x e y
    (https://en.wikipedia.org/wiki/Trapezoidal_rule)
    
    Parameters
    ----------
    x: ndarray
    y: ndarray

    Returns
    -------
    float: resultado de aplicar la fórmula de los trapecios

    '''
    h = x[1] - x[0] #Asumiendo que están equiespaciados

    #Creo los términos que voy a sumar
    f_xi = y[1:] #ndarray de términos f(x_i)
    f_xii = y[0:-1] #ndarray de términos f(x_{i-1})
    
    return h/2*np.sum(f_xi + f_xii)

def trapzf(f,a,b,npts = 100):
    '''
    Calcula el valor de la integral por trapecios

    Parameters
    ----------
    f: función. No es necesario que sea capaz de tomar ndarrays como entrada
    a, b: límites inferior y superior
    npts: número de puntos a utilizar
    
    '''
    #Creo el array x
    x_array = np.linspace(a,b,npts)

    #Creo el array y. Asumo por defecto que f no es capaz de tomar ndarrays como entrada
    y_array = np.zeros(npts)
    for i in range(npts):
        y_array[i] = f(x_array[i])

    return trapz(x_array,y_array)

def f_Euler(x):
    '''
    Integrando de la integral logarítmica de Euler
    '''
    return 1/np.log(x)

def Li(t,npts):
    '''
    Calcula la integral logarítmica de Euler mediante el método de trapecios

    Parameters
    ----------
    t: límite superior de la integral
    npts: número de puntos a utilizar en el método de trapecios

    '''

    return trapzf(f_Euler, 2, t, npts)


#Calculo la integral (test)

t = 10
print(f"Test: para t = {t}")
print("npts", "valor", sep = "\t")
npts_array = np.arange(10,61,10)
integral = np.zeros(len(npts_array))
for i in range(len(npts_array)):
    print(npts_array[i], Li(t, npts_array[i]), sep = "\t")
