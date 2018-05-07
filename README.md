# Graph-Regularized-Non-Negative-Matrix-Factorization

La Factorización No Negativa de Matrices es un grupo de algoritmos cuyo objetivo es descomponer una matriz con entradas no negativas en el producto de dos matrices con la misma propiedad. La clásica NMF aproxima la factorización en el espacio euclideo sin considerar la geometría intrínseca de los datos. El Graph Regularized Non Negative Matrix Factorization supone que los datos están sobre una variedad e incorpora esta información en la función de costo.

# Resultados y usos

Función de costo para distintos valores del parámetro de  regularizacion.
![descarga](https://user-images.githubusercontent.com/30848298/39685807-9e167542-518a-11e8-878a-1ee0e905d808.png)


Se utlizo el algoritmo NMF para una base de datos de imagenes de rostros, se forma una matriz donde cada columna es un rostro. Al utilizar la NMF lo que se consigue es encontrar una base de m vectores del espacio de la matriz original y al mismo tiempo aprender una matriz de pesos con la cual se consigue reconstruir los vectores columna de la matriz original. Si cada columna es la imagen de un rostro, se consigue formar una base del espacio de los rostros y por la naturalza aditiva de la factorización (no existen restas al ser solo numeros positivos), cada elemento de esa base se puede ver como algo que contribuye a formar un rostro (la nariz, los ojos, los labios, etc).
Ejemplo de la base de datos

![00](https://user-images.githubusercontent.com/30848298/39685779-75d5ff58-518a-11e8-99bc-916697443346.png)

Base encontrada para distintos valores del parámetro de  regularizacion.

![10](https://user-images.githubusercontent.com/30848298/39685817-ab42d6a2-518a-11e8-8a91-179eb3485f78.png)
	

## Referencias 
1- Non-Negative Matrix Factorization on Manifold D Cai, X He, X Wu, J Han - Data Mining, 2008. ICDM'08. Eighth IEEE International …, 2008

