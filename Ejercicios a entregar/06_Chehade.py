#Resolución del ejercicio 06 (b) de la clase 04 del curso  Introducción al lenguaje Python orientado a ingenierías y física - Instituto Balseiro
#Pablo Chehade
#Creación: 28/02/2023
#Última modificación: 02/03/2023
#Comentarios:
#(1) en mi pc corre con python archivo.py, no con python3 archivo.py


class Polinomio():
  #Antes decía "Objeto padre". Se lo saqué porque si no no corría

  def __init__(self, coefs=[]):
    """Crea el objeto. Si los coeficientes son proporcionados lo inicializa
    Keyword Arguments:
    coefs -- (default [])
    
    Polinomio tiene como datos una lista de coeficientes que puede estar vacía
    """
    self.set_coeficientes(coefs)
    return None

  def set_coeficientes(self, coefs = []):
    """Si los coeficientes son proporcionados lo inicializa
    Keyword Arguments:
    coefs -- Una lista de coeficientes
    """
    self.coeficientes = coefs #Los coeficientes se guardan directamente como una lista
    return None

  def grado(self):
    "Devuelve el grado del polinomio (un entero)"
    grado = len(self.get_coeficientes())-1
    for coef in self.get_coeficientes()[::-1]: #Esto está en caso de que el polinomio tenga coef nulo acompañando a la mayor potencia
      if coef == 0:
        grado -= 1
      else:
        break
    return grado


  def get_coeficientes(self):
    """Devuelve los coeficientes del polinomio """
    return self.coeficientes
  

  def suma_pol(self, p1):
    """Al polinomio le suma el polinomio `p1` y devuelve un nuevo polinomio
    Keyword Arguments:
    p1 -- Polinomio a sumar
    """
    lista_de_polinomios = [self,p1]
    lista_de_grados = [self.grado(), p1.grado()]
    mayor_orden = max(lista_de_grados)
    menor_orden = min(lista_de_grados)

    p_suma_coefs = [0]*(mayor_orden + 1)
    
    pol_mayor = lista_de_grados.index(mayor_orden)
    
    for i in range(menor_orden + 1):
      p_suma_coefs[i] = self.get_coeficientes()[i] + p1.get_coeficientes()[i]

    for i in range(menor_orden + 1, mayor_orden + 1):
      p_suma_coefs[i] = lista_de_polinomios[pol_mayor].get_coeficientes()[i]

    p_suma = Polinomio(p_suma_coefs)
    return p_suma



  def mul(self, k):
    """Al polinomio lo multiplica por `k` y devuelve un nuevo polinomio
    Keyword Arguments:
    Agregar Documentación
    """
    mult_coefs = self.coeficientes.copy()
    for i in range(len(mult_coefs)):
      mult_coefs[i] = mult_coefs[i]*k
    poli = Polinomio(mult_coefs)
    return poli


  def derivada(self, n=1):
    """Devuelve la derivada (n-ésima) del polinomio (un nuevo polinomio)
    Keyword Arguments:
    n -- (default 1) Orden de derivación

    Modo de uso:
    >>> P = Polinomio([0.1,2,3,0,1)
    >>> P1 = P.derivada()
    >>> P2 = P.derivada(n=2)
    """
    #Control:
    if n <= 0:
     raise Exception(f"n = {n}, debería ser positivo")
    if int(n)-n != 0:
      raise Exception(f"n = {n}, debería ser entero")

    if n == 1: #Primera derivada
      orden_deriv = self.grado()
      pol_deriv_coefs = [0]*orden_deriv
      for i in range(orden_deriv):
          pol_deriv_coefs[i] = self.coeficientes[i+1]*(i+1)
      return Polinomio(pol_deriv_coefs)
    else:
      return self.derivada(n-1).derivada()
  

  def integrada(self, cte = 0, n = 1):
    """Devuelve la antiderivada (n-ésima) del polinomio (un nuevo polinomio)

    Keyword Arguments:
    n -- (default 1) Orden de integración
    cte -- (default 0) Constante de integración

    Modo de uso:
    >>> P = Polinomio([0.1,2,3,0,1)
    >>> P1 = P.integrada()
    >>> P2 = P.integrada(cte=1.2, n=2)

    Nota: para n>1 se usa la misma cte de integración en todas las integrales
    """

    #Control:
    if n <= 0:
     raise Exception(f"n = {n}, debería ser positivo")
    if int(n)-n != 0:
      raise Exception(f"n = {n}, debería ser entero")

    if n == 1:
      orden_int = self.grado()
      pol_int_coefs = [0]*(orden_int+2)
      for i in range(orden_int+1):
          pol_int_coefs[i+1] = self.coeficientes[i]/(i+1)
      #Cte de integración
      pol_int_coefs[0] = cte

      return Polinomio(pol_int_coefs)
    else:
      # p = 
      # print(p, type(p))
      return self.integrada(cte, n-1).integrada(cte)


  def __str__(self):
    "Devuelve un string con la representación del polinomio"
    poli_str = []
    for i, coef in enumerate(self.coeficientes,0):
      #Corrijo las potencias
      if i == 0:
        poli_str.append(f"{coef}")
      elif i == 1:
        poli_str.append(f"{coef} x")
      else:
        poli_str.append(f"{coef} x^{i}")

      #Corrijo los símbolos +/-
      if coef < 0:
        poli_str.append(f"({coef})*x^{i}")
    return " + ".join(poli_str)

  def from_string(self, s, var='x'):
    """
    Keyword Arguments:
    s   -- Representación del polinomio como string
    var -- (default 'x') Variable del polinomio: P(var)

    Modo de uso:
    >>> P = Polinomio()
    >>> P.from_string('x + 2 x^2 + 3x + 1 + x^4', var='x')

    No devuelve nada.
    Nota: Si una potencia aparece más de una vez, sus coeficientes se suman
    """

    #Tenemos que identificar los términos en el string dado. Esto puede no ser sencillo dado que varios términos se pueden escribir de forma distinta. Por ej: x = 1 x = 1 x^1 = + x. Entonces, "normalicemos" el string y llevemos cada término a una expresión estándar

    #Se podrían usar expresiones regulares

    #Eliminamos todos los espacios en blanco y los símbolos ^ porque ya sabemos que están acompañados siempre por var
    new_string = ""
    for character in s:
      if character != " " and character != "^": 
        new_string += character


    #Agregamos signos + al inicio y antes de cada signo -
    new_string2 = ""
    for character in new_string:
      if character == "-": 
        new_string2 += "+"
      new_string2 += character


    #Spliteamos por signo +
    s_split = new_string2.split("+")

    #Recorremos cada elemento y vamos sumando
    poli_suma = Polinomio()
    for termino in s_split:
      #Spliteo por var
      termino_split = termino.split(var)
      
      if len(termino_split) == 1:
        #Si la nueva lista tiene un solo elemento, es una cte
        coef = float(termino_split[0])
        potencia = 0

      else:
        #Si la lista tiene más de un elemento, no es una cte. El primer elemento es el coef y el segundo, la potencia.
        #Si el coef es "" significa que es 1
        if termino_split[0] == "":
          coef = 1
        elif termino_split[0] == "-":
        #Si el coef es "-" significa que es -1
          coef = -1
        else:
          coef = float(termino_split[0])
        
        #Si la potencia es "" signicia que es 1
        if termino_split[1] == "":
          potencia = 1
        else:
          potencia = int(termino_split[1])

      #Creo un polinomio representado por el término y se lo sumo a self
      coeficientes = [0]*(potencia+1)
      coeficientes[-1] = coef
      poli_term = Polinomio(coeficientes)

      poli_suma = poli_suma.suma_pol(poli_term)

    self.set_coeficientes(poli_suma.get_coeficientes())

    return None

  def __call__(self, x):
    """Evalúa el polinomio en el valor `x` y devuelve el resultado
    permite simplemente llamar p(x) para evaluar el polinomio
    Keyword Arguments:
    x -- valor en el que se evalúa el polinomio
    """
    suma = 0
    for i in range(self.grado() + 1):
        suma+=self.get_coeficientes()[i]*x**i
    return suma

  def __add__(self, p):
    """
    Evalúa la suma de polinomios self + p y devuelve el resultado como un polinomio
    Keyword Arguments:
    p -- polinomio a sumar
    """
    return self.suma_pol(p)

  def __mul__(self,value):
    """
    Evalúa la multiplicación de un polinomio por un escalar y devuelve el resultado como un polinomio
    Keyword Arguments:
    value -- escalar a multiplicar
    """
    return self.mul(value)
  
  def __rmul__(self,value):
    """
    Evalúa la multiplicación (a derecha) de un escalar por un polinomio y devuelve el resultado como un polinomio
    Keyword Arguments:
    value -- escalar a multiplicar
    """
    return self.mul(value)

if __name__ == '__main__':

  print('Nombre Apellido')

  P1 = Polinomio([1, 2.1, 3, 1.])   # 1 + 2.1x + 3 x^2 + x^3
  P2 = Polinomio([0, 1, 0, -1, -2])  # x - x^3 - 2x^4

  print(P1.get_coeficientes())
  print(P1.suma_pol(P2).get_coeficientes())
  P11 = P1.derivada()
  print(P11.get_coeficientes())
  P21 = P2.derivada()
  print(P21.get_coeficientes())
  P23 = P2.derivada(n=3)
  print(P23.get_coeficientes())
  print(P1)

  P3 = Polinomio()
  P3.from_string('x + 1 - x^2 - 3x^3')
  print(P3.get_coeficientes())

  print(P3(3.3))
