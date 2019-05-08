def grad(x):
    return 2*x
def update_x(lr,x):
    return x - lr*grad(x)
def func_x(x):
    return x**2


x = -2
for i in range(1,5):
    x = update_x(0.001,x)
    print(x)
    print(func_x(x))