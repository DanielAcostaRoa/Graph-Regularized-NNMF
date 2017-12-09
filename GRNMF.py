from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re

def objetiveFunction(X, U, V, L, lambd):
    Z = X - np.dot(U,V.T)
    Y = np.dot(V.T,np.dot(L,V))
    return np.linalg.norm(Z)**2 + lambd*np.trace(Y)

def optimizaGRNMF(X,U,V,lamb,W,L,D,eps,maxITE):
    flag=True
    it=1
    valAc=objetiveFunction(X,U,V,L,lamb)
    l=[]
    while(flag):
        l.append(valAc)
        print("Iteracion= ",it , " FunObj= ", valAc)
        XV=np.dot(X,V)
        UVV=np.dot(U,np.dot(V.T,V))
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U[i][j]=U[i][j]*(XV[i][j]/UVV[i][j])
        Fac1= np.dot(X.T,U) + lamb*np.dot(W,V)
        Fac2= np.dot(V,np.dot(U.T,U)) + lamb*np.dot(D,V)
        for i in range(V.shape[0]):
            for j in range(U.shape[1]):
                V[i][j]=V[i][j]*(Fac1[i][j]/Fac2[i][j])
        valNew=objetiveFunction(X,U,V,L,lamb)
        if np.abs(valAc-valNew)< eps or it> maxITE:
            flag=0
        valAc=valNew
        it=it+1
    return U,V, np.array(l)

 def GRNMF(X,lambd,vecinos,k,eps,maxIte):
    nbrs = NearestNeighbors(n_neighbors=vecinos, algorithm='ball_tree').fit(X.T)
    W=nbrs.kneighbors_graph(X.T).toarray()
    D=np.zeros(W.shape)
    for i in range(W.shape[0]):
        suma=0
        for j in range(W.shape[1]):
            suma=suma+W[i][j]
        D[i][i]=suma
    L=D-W
    U=np.random.random((X.shape[0],k))
    V=np.random.random((X.shape[1],k))
    U1,V1,valOb=optimizaGRNMF(X,U,V,lambd,W,L,D,eps,maxIte)
    return U1, V1, valOb

def read_pgm(filename, byteorder='>'):
    #Return image data from a raw PGM file as numpy array.
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def muestraBase(base,u): #Muestra un elemento de la base aprendida
    im=np.zeros(2000)
    for i in range(2000):
        im[i]=(u[i][base])*255
    im=np.reshape(im,(50,40))
    x = np.array(im, dtype = np.int32)
    plt.imshow(x, plt.cm.gray)
    plt.show()

    #Ejemplo de la base de datos que hay que leer
image = read_pgm("FaceDataBase/012.pgm", byteorder='<')
image1 = np.zeros((50,40))
for i in range(50):
    for j in range(40):
        image1[i][j]=image[2*i+6][2*j+6]
plt.imshow(image1, plt.cm.gray)
plt.show()

#Creamos la matriz con cada imagen en la columna, comprimimos imagen y normalizamos
M=np.zeros((2000,400))
for i in range(400):
    name=""
    if i+1<10:
        name=name+"0"
    if i+1<100:
        name=name+"0"
    name="FaceDataBase/"+name+str(i+1)+".pgm"
    image = read_pgm(name, byteorder='<')
    image1 = np.zeros((50,40))
    for j in range(50):
        for k in range(40):
            image1[j][k]=image[2*j+6][2*k+6]
    image1 = np.reshape(image1,2000)
    image1 = image1/255.
    for j in range(2000):
        M[j][i]=image1[j]


#Prueba lamb=0
U4, V4, l4 = GRNMF(M,0,1,18,0.01,500)