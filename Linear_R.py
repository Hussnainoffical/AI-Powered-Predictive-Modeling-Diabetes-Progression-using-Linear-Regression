import matplotlib.pylab as plt
from  sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error


diabties = datasets.load_diabetes()

X= diabties.data
Y=diabties.target

X_train=X[:-100]
X_test=X[-30:]
Y_train=Y[:-100]
Y_test=Y[-30:]

modell=linear_model.LinearRegression()
modell.fit(X_train,Y_train)
modelper=modell.predict(X_test)
print("Mean square error is ",mean_squared_error(Y_test,modelper))
print("Model coefficients:", modell.coef_)
print("Model intercept:", modell.intercept_)
import matplotlib.pyplot as plt

plt.scatter(Y_test, modelper, color="blue")
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.title("Actual vs Predicted")
plt.show()


