import numpy as np
import math
from numpy import linalg
import matplotlib.pyplot as plt

a1 = int(input('Введіть а1 від 1 до 10 '))
a2 = int(input('Введіть а2 від 1 до 10 '))
a = np.array([[0, 1, 0],
              [0, 0, 1],
              [-1, -a1, -a2]])
b_i = int(input('Введіть b від 1 до 10 '))
b = np.array([[0],
              [0],
              [b_i]])
c = np.array([[1, 0, 0]])
t0 = float(input("Період квантування t0 [ 0.001, 0.05] "))
q0 = int(input("Введіть точність q "))
x = np.array([[1], [2], [3]])
x_d = np.array([[1], [8], [7]])


def f():
    tmp_a = a.dot(t0)
    i = 0
    res = (1 / math.factorial(i)) * linalg.matrix_power(tmp_a, i)
    while i != q0:
        i += 1
        res += (1 / math.factorial(i)) * linalg.matrix_power(tmp_a, i)

    return res


f = f()
i = 2
q = 0
while i >= 2:
    q = np.array([[i*t0], [i*t0], [(i-1)*t0]])
    fi = f - q.dot(c)
    e = linalg.eigvals(fi)
    flag = True
    for i in e:
        if i.real >= 1:
            flag = False
            print('There is an eigenvalue that is greater than 1!')
        print('Real part of eigenvalue', i.real)
    if flag is True:
        break
    else:
        i += 1
print(q)
t = np.arange(0, 20+t0, t0)
E = np.eye(3)
g = (f - E).dot(linalg.inv(a)).dot(b)
u_k = []
for i in range(len(t)):
    u_k.append(1)
x_k = [x]


def calculate_equation_1(x, u_k):
    res = f.dot(x) + g.dot(u_k)
    return res


for k in range(len(t)):
    x_k.append(calculate_equation_1(x_k[k], u_k[k]))
y_k = []
for m in x_k:
    y_k.append(c.dot(m))


x_k_d = [x_d]


def find_x_d(x_d_i, y_i, u_i):
    return f.dot(x_d_i) + q*(y_i - c.dot(x_d_i)) + g.dot(u_i)


for l in range(len(t)):
    x_k_d.append(find_x_d(x_k_d[l], y_k[i], u_k[i]))
r = []
for n in range(len(t)):
    tmp = x_k_d[n] - x_k[n]
    ri = tmp[0]**2 + tmp[1]**2 + tmp[2]**2
    r.append(math.sqrt(ri))

plt.xlabel('t')
plt.ylabel('похибка')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, r, label='u_k = 1', color='purple')
plt.legend()
plt.show()

for j in range(len(t)):
    print(j, x_k[j][0], x_k[j][1], x_k[j][2], u_k[j], x_k_d[j][0], x_k_d[j][1], x_k_d[j][2], r[j], sep='   |   ')
lol1 = []
lol2 = []
lol3 = []
lol1_d = []
lol2_d = []
lol3_d = []
for i in range(len(x_k)-1):
    tmp = x_k[i][0]
    a = tmp.tolist()
    lol1.append(a[0])
    lol2.append(x_k[i][1].tolist()[0])
    lol3.append(x_k[i][2].tolist()[0])
    lol1_d.append(x_k_d[i][0].tolist()[0])
    lol2_d.append(x_k_d[i][1].tolist()[0])
    lol3_d.append(x_k_d[i][2].tolist()[0])

plt.xlabel('t')
plt.ylabel('x1')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol1,label='x', color='purple')
plt.plot(t, lol1_d,label='x with ^', color='green')
plt.legend()
plt.show()
plt.xlabel('t')
plt.ylabel('x2')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol2, label='x', color='purple')
plt.plot(t, lol2_d, label='x with ^', color='green')
plt.legend()
plt.show()
plt.xlabel('t')
plt.ylabel('x3')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol3, label='x', color='purple')
plt.plot(t, lol3_d,  label='x with ^',color='green')
plt.legend()
plt.show()



u_k.clear()
x_k_d.clear()
x_k.clear()
y_k.clear()
r.clear()
lol1.clear()
lol2.clear()
lol3.clear()
lol1_d.clear()
lol2_d.clear()
lol3_d.clear()
for i in range(len(t)):
    if i % 2 == 0:
        u_k.append(10)
    else:
        u_k.append(-10)
x_k = [x]
for k in range(len(t)):
    x_k.append(calculate_equation_1(x_k[k], u_k[k]))
y_k = []
for m in x_k:
    y_k.append(c.dot(m))
x_k_d = [x_d]
for l in range(len(t)):
    x_k_d.append(find_x_d(x_k_d[l], y_k[i], u_k[i]))
r = []
for n in range(len(t)):
    tmp = x_k_d[n] - x_k[n]
    ri = tmp[0]**2 + tmp[1]**2 + tmp[2]**2
    r.append(math.sqrt(ri))
plt.xlabel('t')
plt.ylabel('похибка')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, r,label='Змінне керування на протилежне', color='purple')
plt.legend()
plt.show()
lol1 = []
lol2 = []
lol3 = []
lol1_d = []
lol2_d = []
lol3_d = []
for i in range(len(x_k)-1):
    tmp = x_k[i][0]
    a = tmp.tolist()
    lol1.append(a[0])
    lol2.append(x_k[i][1].tolist()[0])
    lol3.append(x_k[i][2].tolist()[0])
    lol1_d.append(x_k_d[i][0].tolist()[0])
    lol2_d.append(x_k_d[i][1].tolist()[0])
    lol3_d.append(x_k_d[i][2].tolist()[0])

plt.xlabel('t')
plt.ylabel('x1')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol1, label='x', color='purple')
plt.plot(t, lol1_d,label='x with ^', color='green')
plt.legend()
plt.show()
plt.xlabel('t')
plt.ylabel('x2')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol2,label='x', color='purple')
plt.plot(t, lol2_d,label='x with ^', color='green')
plt.legend()
plt.show()
plt.xlabel('t')
plt.ylabel('x3')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol3,label='x', color='purple')
plt.plot(t, lol3_d,label='x with ^', color='green')
plt.legend()
plt.show()




u_k.clear()
x_k_d.clear()
x_k.clear()
y_k.clear()
r.clear()
lol1.clear()
lol2.clear()
lol3.clear()
lol1_d.clear()
lol2_d.clear()
lol3_d.clear()
for i in range(len(t)):
    u_k.append(3*math.sin(2*t0*i*math.pi/5))
x_k = [x]
for k in range(len(t)):
    x_k.append(calculate_equation_1(x_k[k], u_k[k]))
y_k = []
for m in x_k:
    y_k.append(c.dot(m))
x_k_d = [x_d]
for l in range(len(t)):
    x_k_d.append(find_x_d(x_k_d[l], y_k[i], u_k[i]))
r = []
for n in range(len(t)):
    tmp = x_k_d[n] - x_k[n]
    ri = tmp[0]**2 + tmp[1]**2 + tmp[2]**2
    r.append(math.sqrt(ri))
plt.xlabel('t')
plt.ylabel('похибка')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, r,label='sin', color='purple')
plt.legend()
plt.show()
lol1 = []
lol2 = []
lol3 = []
lol1_d = []
lol2_d = []
lol3_d = []
for i in range(len(x_k)-1):
    tmp = x_k[i][0]
    a = tmp.tolist()
    lol1.append(a[0])
    lol2.append(x_k[i][1].tolist()[0])
    lol3.append(x_k[i][2].tolist()[0])
    lol1_d.append(x_k_d[i][0].tolist()[0])
    lol2_d.append(x_k_d[i][1].tolist()[0])
    lol3_d.append(x_k_d[i][2].tolist()[0])

plt.xlabel('t')
plt.ylabel('x1')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol1,label='x', color='purple')
plt.plot(t, lol1_d, label='x with ^',color='green')
plt.legend()
plt.show()
plt.xlabel('t')
plt.ylabel('x2')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol2,label='x', color='purple')
plt.plot(t, lol2_d,label='x with ^', color='green')
plt.legend()
plt.show()
plt.xlabel('t')
plt.ylabel('x3')
plt.grid()
plt.xticks(np.arange(t[0], t[-1] + 1, 2))
plt.plot(t, lol3,label='x', color='purple')
plt.plot(t, lol3_d, label='x with ^',color='green')
plt.legend()
plt.show()
