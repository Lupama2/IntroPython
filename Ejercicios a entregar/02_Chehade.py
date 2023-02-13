#Resolución del ejercicio 02 (a) de la clase 02 del curso  Introducción al lenguaje Python orientado a ingenierías y física - Instituto Balseiro
#Pablo Chehade
#Creación: 06/02/2023
#Última modificación: 09/02/2023
#Comentarios:
#(1) usé lazos for para facilitar el código pero no es necesario usarlos
#(2) en mi pc corre con python archivo.py, no con python3 archivo.py


print("Pablo Chehade\nClase 2")

s = '''Aquí me pongo a cantar
Al compás de la vigüela,
Que el hombre que lo desvela
Una pena estraordinaria
Como la ave solitaria
Con el cantar se consuela.'''

#Ítem 1
#Cuento el nro de veces que aparecen los substrings
substrings = ["es", "la", "que", "co"]

#Distinguiendo:
print("Distinguiendo: ", end = " ")
for substring in substrings:
    print(s.count(substring), end = " ") #naturalmente distingue entre mayúscular y minúsculas porque son caracteres distintos

#Sin distinguir: hago mayus a todo
print("\nSin distinguir: ", end = " ")
s_upper = s.upper()
for substring in substrings:
    print(s_upper.count(substring.upper()), end = " ")

#ítem 2
#Separo el string con split
s_lista = s.split(sep = "\n")

#Busco el máximo
print("\n",max(s_lista, key = len), " : longitud=",len(max(s_lista, key = len)))

#ítem 3
nuevo_string = s[:5] + s[-5:]
print(nuevo_string)

#ítem 4
n_centro = 10 #nro de caracteres centrales
#Imprimo desde la mitad, n_centro/2 a izquierda y lo mismo a derecha
print(s[int(len(s)/2 - n_centro/2): int(len(s)/2 + n_centro/2)])

#ítem 5
#Necesito reemplazar a un 3er elemento
s_new = s.replace("m", "6")
s_new = s_new.replace("n","m")
s_new = s_new.replace("6", "n")
print(s_new)