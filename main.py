
import math
import numpy as np
from matplotlib import pyplot as plt


def loadingData(x): #path
    f = open(x, "r") #path
    file = f.read()
    #print(len(file))
    file = file.splitlines()
    #print(file)
    for i in range(len(file)):
        file[i] = file[i].split(",\t")
    return file

def matrix(file):
    matrix = np.matrix(file, float)

    #print(matrix)
    return matrix


def translation(matrix, x, y, z):
    num_rows, num_cols = matrix.shape
    #print(num_rows, num_cols)
    for i in range(num_rows):
        matrix[i] += (x, y, z)
    return matrix


def scaling(matrix, x, y, z):
    num_rows, num_cols = matrix.shape
    #print(num_rows, num_cols)
    S = np.matrix([[x, 0, 0], [0, y, 0], [0, 0, z]])
    matrix = np.dot(matrix, S)
    return matrix

def rotateNeutral(matrix, alfa, axis):
    tmp1 = []
    tmp2 = []
    if axis=="X":
        for i in matrix:
            tmp1.append(i[0,1])
            tmp2.append(i[0,2])
        offset = [0, (max(tmp1)+min(tmp1))/2, (max(tmp2)+min(tmp2))/2]
    elif axis=="Y":
        for i in matrix:
            tmp1.append(i[0,0])
            tmp2.append(i[0,2])
        offset = [(max(tmp1)+min(tmp1))/2, 0, (max(tmp2)+min(tmp2))/2]
        #offset = [(matrix[120, 0] + matrix[0, 0]) / 2, (matrix[120, 1] + matrix[0, 1]) / 2, 0]
    elif axis=="Z":
        for i in matrix:
            tmp1.append(i[0,0])
            tmp2.append(i[0,1])
        offset = [(max(tmp1)+min(tmp1))/2, (max(tmp2)+min(tmp2))/2, 0]
        #offset = [(matrix[120, 0] + matrix[0, 0]) / 2, (matrix[120, 1] + matrix[0, 1]) / 2, 0]
    else:
        offset=0
    #print(offset)
    return rotate(matrix, alfa, offset, axis)

def chooseRotationMatrix(axis, alpha):
    if axis=="X":
        return np.matrix([[1, 0, 0], [0, math.cos(alpha), math.sin(alpha)], [0, -math.sin(alpha), math.cos(alpha)]])
    elif axis=="Y":
        return np.matrix([[math.cos(alpha), 0, math.sin(alpha)], [0, 1, 0], [-math.sin(alpha), 0, math.cos(alpha)]])
    elif axis=="Z":
        return np.matrix([[math.cos(alpha), -math.sin(alpha), 0], [math.sin(alpha), math.cos(alpha), 0], [0, 0, 1]])
    else:
        return 0

def rotate(matrix, alfa, offset, axis):
    #num_rows, num_cols = matrix.shape
    #print(num_rows, num_cols)
    matrix=translation(matrix, 0-offset[0], 0-offset[1], 0-offset[2])
    r = chooseRotationMatrix(axis, alfa)
    matrix = np.dot(matrix, r)
    matrix=translation(matrix, offset[0], offset[1], offset[2])
    return matrix

def reflection(matrix):
    #num_rows, num_cols = matrix.shape
    #print(num_rows, num_cols)
    R = np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return np.dot(matrix, R)

def shear(matrix, cx, cy):
    num_rows, num_cols = matrix.shape
    print(num_rows, num_cols)
    Sh = np.matrix([[1, cx, 0], [cy, 1, 0], [0, 0, 1]])
    return np.dot(matrix, Sh)

def plotDots(matrix):
    num_rows, num_cols = matrix.shape
    ax = plt.axes(projection="3d")
    X = np.zeros(shape=(num_rows, 1))
    Y = np.zeros(shape=(num_rows, 1))
    Z = np.zeros(shape=(num_rows, 1))
    for i in range(num_rows):
        X[i] = matrix[i, 0]
        Y[i] = matrix[i, 1]
        Z[i] = matrix[i, 2]
    #PlotPoints
    ax.scatter(X, Y, Z, marker=".", alpha=1)
    plt.show()


def plotLine(matrix):
    num_rows, num_cols = matrix.shape
    ax = plt.axes(projection="3d")
    X = np.zeros(shape=(num_rows, 1))
    Y = np.zeros(shape=(num_rows, 1))
    Z = np.zeros(shape=(num_rows, 1))
    for i in range(num_rows):
        X[i] = matrix[i, 0]
        Y[i] = matrix[i, 1]
        Z[i] = matrix[i, 2]
    ax.plot(X, Y, Z)
    plt.show()

def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def create_matrix():
    global noise
    widmo = generate_fractal_noise_2d([50, 50], [1, 1], 2)
    print(widmo.shape)
    noise = []
    for i in range(len(widmo)):
        for j in range(len(widmo)):
            noise.append([i, j, widmo[i][j]*100])
    return matrix(noise)

def plotSzescian(m):

    C = np.array([[0, 255, 0], [0, 255, 0], [0, 255, 0], [255, 255, 0], [255, 0, 0], [255, 0, 255], [0, 255, 0],
                  [128, 0, 0]])
    X = np.zeros(shape=(len(m), 1))
    Y = np.zeros(shape=(len(m), 1))
    Z = np.zeros(shape=(len(m), 1))
    ax = plt.axes(projection="3d")
    for i in range(len(m)):
        X[i] = m[i, 0]
        Y[i] = m[i, 1]
        Z[i] = m[i, 2]
    ax.scatter(X, Y, Z, c=C / 255.0, s=120)
    plt.show()

if __name__ == '__main__':

    #x = np.linspace(-6, 6, 30)
    #y = np.linspace(-6, 6, 30)
    #X, Y = np.meshgrid(x, y)
    #Z = f(X, Y)

    # plot surface 3d
    # plot3d(X,Y,Z)

    cube = matrix(loadingData("szescian.txt"))

    cube2 = cube.copy()
    plt.title("cube")
    plotSzescian(cube2)

    cube2 = cube.copy()
    plt.title("translacja")
    plotSzescian(translation(cube2, 250, 250, 250))

    cube2 = cube.copy()
    plt.title("odbicie")
    plotSzescian(reflection(cube2))

    cube2 = cube.copy()
    plt.title("skalowanie")
    plotSzescian(scaling(cube2, 50, 50, 50))

    cube2 = cube.copy()
    plt.title("rotacja 1")
    plotSzescian(rotateNeutral(rotateNeutral(cube2, math.pi/4, "X"), math.pi/4, "Y"))  # obracanie wokol srodka

    cube2 = cube.copy()
    plt.title("rotacja 2")
    plotSzescian(rotateNeutral(cube2, math.pi / 2, "X"))

    cube2 = cube.copy()
    plt.title("rotacja 3")
    plotSzescian(rotate(cube2, 1 * math.pi, [100, 0, 0], "X"))

    cube2 = cube.copy()
    plt.title("przechylanie")
    plotSzescian(shear(cube2, 0.5, 0))




    noise = create_matrix()
    noise2 = noise.copy()
    plt.title("noise")
    plotDots(noise2)

    noise2 = noise.copy()
    plt.title("translacja")
    plotDots(translation(noise2, 250, 250, 250))

    noise2 = noise.copy()
    plt.title("odbicie")
    plotDots(reflection(noise2))

    noise2 = noise.copy()
    plt.title("skalowanie")
    plotDots(scaling(noise2, 50, 50, 50))

    noise2 = noise.copy()
    plt.title("rotacja 1")
    plotDots(rotateNeutral(noise2, math.pi/6, "X"))  # obracanie wokol srodka

    noise2 = noise.copy()
    plt.title("rotacja 2")
    plotDots(rotateNeutral(noise2, math.pi / 2, "X"))

    noise2 = noise.copy()
    plt.title("rotacja 3")
    plotDots(rotate(noise2, 1 * math.pi, [0, 100, 0], "X"))

    noise2 = noise.copy()
    plt.title("przechylanie")
    plotDots(shear(noise2, 0.5, 0))



