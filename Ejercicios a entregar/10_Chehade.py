#Resolución del ejercicio 10 (b) de la clase 20 del curso  Introducción al lenguaje Python orientado a ingenierías y física - Instituto Balseiro
#Pablo Chehade
#Creación: 13/03/2023
#Última modificación: 13/03/2023
#Comentarios:
#(1) en mi pc corre con python archivo.py, no con python3 archivo.py

#Importo librerías
import numpy as np
import matplotlib.pyplot as plt


#Ejercicio 1
def metodo_cociente_de_areas(N_puntos):
    '''
    Aproxima el valor de pi mediante el método de cociente de áreas
    
    Parameters
    ----------
    N_puntos (int): nro de puntos aleatorios generados

    Returns
    -------
    pi_aprox (float): aproximación de pi

    '''
    #Genero puntos en el plano de forma aleatoria con una distribución uniforme
    puntos = np.random.rand(N_puntos, 2)

    #Calculo la distancia al origen
    distancias = puntos[:,0]**2 + puntos[:,1]**2

    area = 4*np.sum(distancias <= 1)/N_puntos

    pi_aprox = area

    return pi_aprox

#Test:
# N_puntos =  100000
# print(metodo_cociente_de_areas(N_puntos))


#Ejercicio 2

def f(x):
    return np.sqrt(1-x**2)

def metodo_valor_medio(N):
    '''
    Aproxima el valor de pi mediante el método del valor medio

    Parameters
    ----------
    N (int): nro de valores aleatorios generados
    
    Returns
    -------
    pi_aprox (float): aproximación de pi
    '''

    #Generamos valores aleatoriamente 
    valores = np.random.rand(N)


    pi_aprox = 4*np.mean(f(valores))

    return pi_aprox

#Test:
# N = 1000000
# print(metodo_valor_medio(N))




#Ejercicio 3

N_array = np.logspace(2,4,20, dtype = int)

metodo_cociente_de_areas_vec = np.vectorize(metodo_cociente_de_areas)
metodo_valor_medio_vec = np.vectorize(metodo_valor_medio)

pi_metodo1 = metodo_cociente_de_areas_vec(N_array)
pi_metodo2 = metodo_valor_medio_vec(N_array)

plt.plot(N_array, pi_metodo1, label = "Método de cociente de áreas")
plt.plot(N_array, pi_metodo2, label = "Método del valor medio")
plt.legend()
plt.xlabel("N")
plt.ylabel("Aproximación de $\pi$")
plt.xscale("log")
plt.show()

#Ejercicio 4
N_exp = 1000 #Nro de veces que se repite el experimento
N = 15000
N_array = N*np.ones(N_exp, dtype = int)

pi_metodo1 = metodo_cociente_de_areas_vec(N_array)
pi_metodo2 = metodo_valor_medio_vec(N_array)


#Grafico
alpha = 0.3
plt.hist(pi_metodo1, alpha = alpha, label = "Método de cociente de áreas", density = True, color = "tab:blue")
plt.hist(pi_metodo2, alpha = alpha, label = "Método del valor medio", density = True, color = "tab:red")
plt.xlim([2.9,3.4])
plt.legend()

#Calculo la desviación estándar y la media
mean1, std1 = np.mean(pi_metodo1), np.std(pi_metodo1)
mean2, std2 = np.mean(pi_metodo2), np.std(pi_metodo2)

#Grafico una gaussiana para cada método
def f_gauss(x, mu, sigma):
    '''
    Evalúa la función Gaussiana con desviación estándar sigma y media mu
    '''
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))

x_array = np.linspace(2.9,3.4, 10000)
plt.plot(x_array, f_gauss(x_array, mean1, std1), color = "tab:blue")
plt.plot(x_array, f_gauss(x_array, mean2, std2), color = "tab:red")
plt.xlabel("Estimación de Monte Carlo")
plt.ylabel("Densidad de probabilidad")
plt.show()

#Ejercicio 5
def metodo_aguja(N):
    t = 2
    l = 1

    #La distancia entre rayas es t

    #Genero un punto random entre [0,1]x[0,t]. Este será el punto inicial de la aguja
    puntos_inicio = np.random.rand(N,2)

    #Genero un ángulo al azar
    angulos = np.random.rand(N)*(2*np.pi) #Es trampa usar pi acá?

    #Calculo el punto final de la aguja
    puntos_final = l*np.array([np.cos(angulos), np.sin(angulos)]).T + puntos_inicio

    #Verificación: el resultado del print debe dar l para cada punto
    # print(np.sqrt((puntos_final[:,0] - puntos_inicio[:,0])**2 + (puntos_final[:,1] - puntos_inicio[:,1])**2))

    #Si la coordenada y es > t o < 0, entonces cruzó una raya
    P = (np.sum(puntos_final[:,1]>t) + np.sum(puntos_final[:,1]<0))/N

    pi_aprox = 2*l/t/P

    return pi_aprox

#Test:
# N = 1000000
# print(metodo_aguja(N))

#Vectorizo
metodo_aguja_vec = np.vectorize(metodo_aguja)

#Ejercicio 5 parte 2
N_exp = 1000 #Nro de veces que se repite el experimento
N = 15000
N_array = N*np.ones(N_exp, dtype = int)


metodos = [metodo_cociente_de_areas_vec, metodo_valor_medio_vec, metodo_aguja_vec]
colores = ["tab:blue", "tab:red", "tab:green"]
legendas = ["Método de cociente de áreas", "Método del valor medio","Método de las agujas"]

#Parámetros de graficación
alpha = 0.3 #opacidad del gistograma

for (i,metodo) in enumerate(metodos):
    #Calculo pi con cada método
    pi_aprox = metodo(N_array)
    #Histograma
    plt.hist(pi_aprox, alpha = alpha, label = legendas[i], density = True, color = colores[i])
    #Calculo la desviación estándar y la media
    mean, std = np.mean(pi_aprox), np.std(pi_aprox)
    #Grafico gaussiana
    x_array = np.linspace(2.9,3.4, 10000)
    plt.plot(x_array, f_gauss(x_array, mean, std), color = colores[i])

plt.xlabel("Estimación de Monte Carlo")
plt.ylabel("Densidad de probabilidad")
plt.xlim([2.9,3.4])
plt.legend()
plt.show()
