#Resolución del ejercicio 04 (d) de la clase 0}4 del curso  Introducción al lenguaje Python orientado a ingenierías y física - Instituto Balseiro
#Pablo Chehade
#Creación: 19/02/2023
#Última modificación: 20/02/2023
#Comentarios:
#(1) en mi pc corre con python archivo.py, no con python3 archivo.py



def string_to_list(sudoku_str):
    '''
    Convierte un string con formato sudoku a una lista bidimensional de la forma [[...],[...],...]

    Parameters
    ----------
    sudoku_str: string de 9 filas, cada una formada por 9 números

    Returns
    -------
    sudoku_list: lista de números enteros tamaño 9x9 donde cada elemento hace referencia al correspondiente en sudoku_str
    '''
    size = 9 #tamaño vertical y horizontal del sudoku
    #Creo la lista
    sudoku_list = []

    sudoku_split = sudoku_str.split()

    linea = [0]*9
    for i in  range(size):
        for j in range(size):
            linea[j] = int(sudoku_split[i][j])
        sudoku_list.append(linea.copy())

    return sudoku_list

def check_repetidos(lista):
    '''
    Verifica si una lista tiene elementos repetidos

    Parameters
    ----------
    list: lista

    Returns
    -------
    bool: True si tiene elementos repetidos y False en caso contrario
    
    '''
    conj = set(lista)
    if len(conj) != len(lista):
        return True #Hay elementos repetidos
    else:
        return False

def check_sudoku(grilla):
    '''
    Dada una grilla bidimensional de nros, verifica si es solución correcta del Sudoku. Para que sea una solución correcta debe cumplirse que
    - Los números estén entre 1 y 9
    - En cada fila no deben repetirse
    - En cada columna no deben repetirse
    - En todas las regiones de 3x3 que no se solapan, empezando de cualquier esquina, no deben repetirse

    Parameters
    ----------
    grilla: lista bidimensional de números enteros de tamaño 9x9

    Returns
    -------
    bool: True si corresponde a la resolución correcta del Sudoku y False en caso contrario.
    
    '''
    size = 9

    #Verifico que los nros estén entre el 1 y el 9
    numeros = [1,2,3,4,5,6,7,8,9]
    for i in range(size):
        for j in range(size):
            if (grilla[i][j] in numeros) == False: #Si el nro no está en la lista de nros
                # print("Los nros no están entre 1 y 9")
                return False
            
    #Verifico que los nros de cada fila no se repitan
    for i in range(size):
        if check_repetidos(grilla[i]):
            # print("Los nros se repiten en una fila")
            return False
    
    #Verifico que los nros en cada columna no se repitan
    for i in range(size):
        columna = [0]*9
        for j in range(size):
            columna[j] = grilla[j][i]
        if check_repetidos(columna):
            # print("Los nros se repiten en una columna")
            return False
        
    #Verifico que en todas las regiones de 3x3 que no se solapan, los nros no se repitan
    for i in range(size):
        j, k = (i // 3) * 3, (i % 3) * 3 #Según esto, cuando i = 0,1,2 j=0 y k = 0,3,6. De este modo, se recorren los 4 cuadrados 3x3 de arriba de izquierda a derecha. De forma análoga cuando i = 3,4,5,6,7,8
        r = [grilla[a][b] for a in range(j, j+3) for b in range(k, k+3)] #Se genera una lista con los elementos de cada cuadrado
        if check_repetidos(r):
            # print("Los nros se repiten en un cuadrado")
            return False
    
    #Si la ejecución llegó hasta acá, el sudoku es resolución correcta
    return True


#Test:

# #Dado el sudoku
# sudoku = """145327698
#         839654127
#         672918543
#         496185372
#         218473956
#         753296481
#         367542819   
#         984761235
#         521839764"""

# #Lo convierto a lista y checkeo si está resuelto
# sudoku_list = string_to_list(sudoku)
# print("Sudoku resuelto?", check_sudoku(sudoku_list))

# #Agrego un 0
# import copy
# sudoku_list1 = copy.deepcopy(sudoku_list) #Si no hago una deepcopy, los elementos de sudoku_list1 (que son listas) siguen siendo los mismos que de sudoku_list. Por lo tanto, al modificar uno también estaría modificando el otro.
# sudoku_list1[0][0] = 0
# print("Agrego un 0. Sudoku resuelto?", check_sudoku(sudoku_list1))

# #Agrego un 1 para que se repita en una fila
# sudoku_list1 = copy.deepcopy(sudoku_list)
# sudoku_list1[0][1] = 1
# print("Repetición en fila. Sudoku resuelto?", check_sudoku(sudoku_list1))