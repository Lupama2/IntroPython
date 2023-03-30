#Resolución del ejercicio 2 (b) de la clase 14 del curso  Introducción al lenguaje Python orientado a ingenierías y física - Instituto Balseiro
#Pablo Chehade
#Creación: 30/03/2023
#Última modificación: 30/03/2023
#Comentarios:
#(1) en mi pc corre con python archivo.py, no con python3 archivo.py


#Importo librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# %matplotlib tk #Descomentar en un notebook
plt.ioff()


#Caída Libre clase 08

#a = -g
#v = -g*t + v0
#z = -g/2*t**2 + v0*t + h0

def dinamica(h0=0,v0=0,g=10, N=1):
    #Desde t = 0 hasta t tal que h = 0
    
    #t array
    t0 = 0
    tmax = (-v0 - np.sqrt(v0**2 - 4*(-g/2)*h0))/(-g)
    t_array = np.linspace(t0,tmax, N)

    #h_array
    h_array = -g/2*t_array**2 + v0*t_array + h0

    #v_array
    v_array = -g*t_array + v0

    #Guardo los resultados
    archivo = f"caida_{v0}_{h0}.dat"

    np.savetxt(archivo, np.array([t_array, v_array, h_array]).transpose())


#Animación

#Creo datos
h0 = 1
v0 = 5
g = 10
N_puntos = 100

dinamica(h0,v0, g, N_puntos)

#Calculo la altura máxima, de modo de fijar el límite superior del gráfico
#hmax está dado por la condición v = 0, equivalente a t = v0/g, de modo que
tmax = v0/g
hmax = -g/2*tmax**2 + v0*tmax + h0

#Cargo datos
datos = np.loadtxt(f"caida_{v0}_{h0}.dat")
t_array, v_array, h_array = datos[:,0], datos[:,1], datos[:,2]
x_array = np.zeros(len(h_array))
data = np.vstack([x_array, h_array])


def update_point(num, data, elementos):
  #Separo los elementos
  point = elementos[0]
  point_ghost1 = elementos[1]
  point_ghost2 = elementos[2]
  texto = elementos[3]

  #Actualizo cada elemento
  point.set_data(data[:, num])
  if num == 0:
    point_ghost1.set_data(data[:, num])
    point_ghost2.set_data(data[:, num])
  if num == 1:
    point_ghost1.set_data(data[:, num-1])
    point_ghost2.set_data(data[:, num-1])
  if num>1:
    point_ghost1.set_data(data[:, num-1])
    point_ghost2.set_data(data[:, num-2])
  texto.set_text(f"t = {round(t_array[num],2)} \nv = {round(v_array[num],2)} \nh = {round(h_array[num],2)}")

  return elementos



# Creo la figura e inicializo
fig1, ax = plt.subplots(figsize=(12,8))

#Creo los elementos
P, = plt.plot([], [], 'o', color = "tab:red") # equivalente a la siguiente
P_ghost1, = plt.plot([], [], 'o', color = "tab:red", alpha = 0.5)
P_ghost2, = plt.plot([], [], 'o', color = "tab:red", alpha = 0.25)
font = {'family': 'serif',
        'color':  'tab:blue',
        'weight': 'normal',
        'size': 16,
        }
texto = plt.text(x = -0.35,y = hmax/2, s = "\t\tt\tt\t", fontdict = font)

elementos = [P, P_ghost1, P_ghost2, texto]

#Fijo límites
ax.set_xlim(-.5, .5)
ax.set_ylim(0, 5/4*hmax)
ax.set_xlabel('x')
ax.set_ylabel('h')
ax.set_title('Animación de una caída libre')

line_ani = animation.FuncAnimation(fig1, update_point, N_puntos, fargs=(data, elementos), interval=100, blit=True)

plt.show()