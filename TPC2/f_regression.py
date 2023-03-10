from scipy.stats import linregress
import numpy as np
import sys
sys.path.append('../TPC1')
from tpc1 import Dataset

# https://vitalflux.com/interpreting-f-statistics-in-linear-regression-formula-examples/?utm_content=cmp-true

def f_regression(dataset):
    x = dataset.getX()
    y = dataset.getY()
    
    # add linha de 1's ao conjunto de dados
    x = np.hstack((np.ones((x.shape[0], 1)), x))

    # calcula os coeficientes da regressão linear
    b = np.linalg.inv(x.T @ x) @ x.T @ y

    # calcula o erro quadrático
    y_hat = x @ b
    e = y - y_hat

    # calcula a soma dos quadrados dos resíduos e dos quadrados dos coeficientes
    ssr = np.sum((y_hat - np.mean(y)) ** 2)
    sse = np.sum(e ** 2)
    # sst = np.sum((y - np.mean(y)) ** 2)

    # calcula os graus de liberdade
    df_sse = x.shape[0] - x.shape[1]
    df_ssr = x.shape[1] - 1
    # df_sst = x.shape[0] - 1

    # calcula o valor de F e o p-value
    f = (ssr / df_ssr) / (sse / df_sse)
    p = 1 - linregress(x[:, 1], y)[3]

    return f, p



if __name__ == "__main__":
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([10, 11, 12])
    print(f_regression(Dataset(x, y)))