import numpy as np
import matplotlib.pyplot as plt
import GaussianProcess as GP
from KRR import Ridge

"""
    Just some exeriments, disregard this file pls
    ALSO: the code is chaos, I know ... I'm sorry
"""


#np.random.seed(10)
NOISE_STRENGTH = 5


def random_polynomial():
    # Generate a random polynomial in 3D and plot it

    # Generate random coefficients
    a = np.random.rand(4)
    b = np.random.rand(4)
    c = np.random.rand(1)

    # Generate x and y values
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Generate Z values
    func =  lambda X, Y: (a[1]*X**3 + b[1]*Y**2 + a[3]*X + b[3]*Y) * np.sin(X)
    Z = func(X, Y)

    # generate 10 random points (initaial data)
    x_points = np.random.uniform(-3, 3, 10)
    y_points = np.random.uniform(-3, 3, 10)
    z_points = func(x_points, y_points)
    z_points = [point + NOISE_STRENGTH * (np.random.random() - 0.5) for point in z_points]        # add random noise, should be replaced by gaussian noise
    

    # Plot the polynomial
    fig = plt.figure()
    ax =  fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, Z)
    
    #ax.scatter(x_points, y_points, z_points, c='b', marker='o')

    # plot the point with the highest uncertainty (loop)
    mse_krr, mse_gp = [], []
    for i in range(40):
        max_uncert_point = find_high_uncertainty_points(x_points, y_points, z_points, ax)    #implements GP to do this
        x_points = np.append(x_points, max_uncert_point[0])
        y_points = np.append(y_points, max_uncert_point[1])
        z_points = np.append(z_points, func(max_uncert_point[0], max_uncert_point[1]))
            
        # plot max uncertainty point
        max_uncert_point[2] =  func(max_uncert_point[0], max_uncert_point[1]) + NOISE_STRENGTH * (np.random.random() - 0.5)      # add random noise, should be replaced by gaussian noise
        ax.scatter(max_uncert_point[0], max_uncert_point[1], max_uncert_point[2], c='r', marker='o')

        if i%5 == 0:
            print("Iteration: ", i)
            # Fit a Kernel Ridge Regression model to the full data set
            mse_krr.append(fit_KRR(x_points, y_points, z_points, ax, func))
            mse_gp.append(fit_GP(x_points, y_points, z_points, ax, func))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.legend()
    plt.show()

    plt.plot(mse_krr, label = 'KRR')
    plt.plot(mse_gp, label = 'GP')
    plt.xlabel('steps')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def find_high_uncertainty_points(x_points, y_points, z_points, ax):
    # Find the points with the highest uncertainty using a Gaussian Process

    # refctor the data and feed them to a Gaussian Process
    data = [[x_points[i], y_points[i]] for i in range(len(x_points))]
    data = np.array(data)
    targets = np.array([[z_points[i]] for i in range(len(z_points))])
    gp = GP.GaussianProcess(data, targets = targets, kernel_type = 'RBF')
    gp.train()

    # Find the points with the highest uncertainty
    max_uncertainty = 0
    max_point = None
    x_vec = np.linspace(-3, 3, 50)
    y_vec = np.linspace(-3, 3, 50)
    for x in x_vec:
        for y in y_vec:
            sample = [x, y]            
            z_pred, uncertainty = gp.predict(sample)
            if uncertainty > max_uncertainty:          # uncertainty and z_pred is a measure of a good point
                max_uncertainty = uncertainty
                max_point = [x, y, z_pred]
    
    return max_point



def fit_KRR(x_points, y_points, z_points, ax, func):
    mse = []

    # data
    data = [[x_points[i], y_points[i]] for i in range(len(x_points))]
    targets = np.array([[z_points[i]] for i in range(len(z_points))])

    # Fit a Kernel Ridge Regression model to the data
    krr = Ridge(data, targets, None, kernel_type= "polynomial", alpha= 1e-8, gamma= 0.001)  # kernel ridge regression
    print("finding hyperparameters ... ")
    krr.optimize_hyperparameters()
    krr.fit()

    # Plot the model
    x_vec = np.linspace(-3, 3, 50)
    y_vec = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_vec, y_vec)
    Z = np.zeros((len(x_vec), len(y_vec)))
    for i in range(len(x_vec)):
        for j in range(len(y_vec)):
            Z[j, i] = krr.model.predict([[x_vec[i], y_vec[j]]])
            mse.append((Z[j, i] - func(x_vec[i], y_vec[j]))**2 )

    ax.plot_surface(X, Y, Z, alpha=0.5, color = 'blue', label = 'KRR')
    #ax.scatter(x_points, y_points, z_points, c='grey', marker='o')

    return np.mean(mse)


def fit_GP(x_points, y_points, z_points, ax, func):
    mse = []

    # data
    data = [[x_points[i], y_points[i]] for i in range(len(x_points))]
    data = np.array(data)
    targets = np.array([[z_points[i]] for i in range(len(z_points))])
    targets = np.array(targets)
    
    # fit GP
    gp = GP.GaussianProcess(data, targets = targets, kernel_type = 'RBF')
    gp.train()
    
    # plot the model
    x_vec = np.linspace(-3, 3, 30)
    y_vec = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x_vec, y_vec)
    Zgp = np.zeros((len(x_vec), len(y_vec)))
    for i in range(len(x_vec)):
        for j in range(len(y_vec)):
            Zgp[j, i] = gp.predict([x_vec[i], y_vec[j]])[0]
            mse.append((Zgp[j, i] - func(x_vec[i], y_vec[j]))**2 )

    ax.plot_surface(X, Y, Zgp, alpha=0.5, color = 'red', label = 'GP')
    #ax.scatter(x_points, y_points, z_points, c='grey', marker='o')

    return np.mean(mse)


random_polynomial()