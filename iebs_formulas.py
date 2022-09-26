import numpy as np

def tipificacion(a:np.array) -> np.array:
    """
    Normalizar valores de una matriz en base a la fórmula z = (x - mu) / sigma
    :param np.array a: matriz de entrada
    :return: copia de la matriz inicial con valores tipificados.
    """
    a_copy = a.copy().astype("float64")
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            col_mean = a[:][:,col].mean()
            col_stdev = np.sqrt(sum([(x-col_mean)**2/len(a[:][:,col]) for x in a[:][:,col]]))
            a_copy[row][col] = (a[row][col] - col_mean) /col_stdev
    return a_copy

def calc_covariance(x,y):
    return sum((x - x.mean())*(y - y.mean()))/(len(x)-1)

def det(a:np.array,valor_i_fijado = 0)-> np.int64:
    """
    Calcula la determinante de una matriz
    :param a np.array: Matriz de entrada
    :param valor_i_fijado int: Fijación de fila.
    :return np.int64: determinante de la matriz.
    """

    total = 0
    
    i= valor_i_fijado #Fijamos i(row).

    # Acabar recursión si el array es 2x2 ; retornamos su determinante (a_{11}*a_{22}-a_{12}*a_{21})
    if len(a) == 2 and len(a[0]) == 2:
        val = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        return val

    for j in range(len(a)):
        a_copy = a.copy()
        # Eliminamos la fila i y columna j para conseguir a sobrelínea.
        a_overline = a_copy[[x for x in range(len(a_copy))if x != i],:][:,[y for y in range(len(a_copy)) if y !=j]]
        #Aplicamos la fórmula de la determinante
        total += ((-1)**(i+j)) * a[i][j] * det(a_overline)
        
    return total

class PCA:
    def __init__(self,array,n_componentes):
        self.array = array
        self.n_components = n_componentes

    def fit_transform(self):
        #Centramos observaciones a la media
        mean_centered = self.array - np.mean(self.array,axis=0)
        #Matriz de covarianza
        covariance_matrix = np.cov(mean_centered.T)
        #Cálculo de eigenvalues y eigenvectors.
        egvals,egvecs = np.linalg.eig(covariance_matrix)
        sorted_egval_indices = egvals.argsort()[::-1]
        selected_egvecs = egvecs[:,sorted_egval_indices][:,:self.n_componentes]
        reducted_array = np.dot(selected_egvecs.T,mean_centered.T).T
        reducted_covar_matrix = np.cov(reducted_array.T)
        self.explained_variance = np.array(reducted_covar_matrix[0][0],reducted_covar_matrix[1][1])
        self.explained_variance_ratio = np.array(reducted_covar_matrix[0][0],reducted_covar_matrix[1][1]) / sum(reducted_covar_matrix[0][0],reducted_covar_matrix[1][1])
        return np.dot(selected_egvecs.T,mean_centered.T).T

def average_distance(a,b):
    if len(a) != len(b):
        print(a,"\n",b)
        raise ValueError("Uneven vectors inputted, please select symmetrical vectors.")

    return ((1/len(a[0])) * sum([(x - y)**2 for x,y in zip(a[0],b[0])]))**0.5