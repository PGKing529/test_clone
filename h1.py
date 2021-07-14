import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import Ridge,Lasso


df = pd.read_csv("data.csv")
df = pd.DataFrame(df)
y = df.iloc[:, -1:]
df = df.iloc[:, 0:-1]

# mean = pd.mean(df)
# print(mean)
# print(df)
mean=df.mean()

# print(mean)
new = (df-mean)
# print(new)
p=df**2
p = pd.DataFrame(p)
p_mean = (p.mean())**(1/2)
new=new/p_mean

# print(new)
# print(y)



sns.pairplot(df)




lamda = [0.01,0.1,0.5,1,1.5,2,5,10,20,30,50,100,200,300]
all_lists = []
for i in lamda:
    Reg = Ridge(i)
    Reg.fit(new,y)
    all_lists.append(Reg.coef_)


matrix = np.array(all_lists)
# print(matrix[0][0][0])


line0 = []
for i in range(0,14):
    line0.append(matrix[i][0][0])

line1 = []
for i in range(0,14):
    line1.append(matrix[i][0][1])

line2 = []
for i in range(0,14):
    line2.append(matrix[i][0][2])

line3 = []
for i in range(0,14):
    line3.append(matrix[i][0][3])

line4 = []
for i in range(0,14):
    line4.append(matrix[i][0][4])

line5 = []
for i in range(0,14):
    line5.append(matrix[i][0][5])

line6 = []
for i in range(0,14):
    line6.append(matrix[i][0][6])

line7 = []
for i in range(0,14):
    line7.append(matrix[i][0][7])




plt.figure(figsize=(20,8),dpi=80)

plt.plot(range(0,14),line0, color="red")
plt.plot(range(0,14),line1, color="brown")
plt.plot(range(0,14),line2, color="green")
plt.plot(range(0,14),line3, color="blue")
plt.plot(range(0,14),line4, color="orange")
plt.plot(range(0,14),line5, color="pink")
plt.plot(range(0,14),line6, color="purple")
plt.plot(range(0,14),line7, color="grey")





df = pd.read_csv("data.csv")
df = pd.DataFrame(df)
train_x=df.iloc[:-1, 0:-1]
train_y=df.iloc[:-1, -1:]
test_x=df.iloc[-1:, 0:-1]
test_y=df.iloc[-1:, -1:]

p=-0.1
E_list=[]
for i in range(0,501):
    p = p + 0.1
    alpha = round(p,1)
    E = 0
    for n in range(0,38):
        Reg = Lasso(alpha)
        Reg.fit(train_x,train_y)
        pre = Reg.predict(test_x)
        err = (pre - test_y)**2
        E = E + err
    E = E / 38
    E = E.values
    E_list.append(E[0][0])

print(E_list)
plt.figure(figsize=(20,8),dpi=80)
plt.plot(range(0,501),E_list, color="red")



plt.figure(figsize=(20,8),dpi=80)
plt.plot(range(0,501),E_list, color="red")


lamda = [0.01,0.1,0.5,1,1.5,2,5,10,20,30,50,100,200,300]
all_lists = []
for i in lamda:
    Reg = Lasso(i)
    Reg.fit(new,y)
    all_lists.append(Reg.coef_)

matrix = np.array(all_lists)
print(matrix)
# print(matrix[0][0][0])


line0 = []
for i in range(0,14):
    line0.append(matrix[i][0])

line1 = []
for i in range(0,14):
    line1.append(matrix[i][1])

line2 = []
for i in range(0,14):
    line2.append(matrix[i][2])

line3 = []
for i in range(0,14):
    line3.append(matrix[i][3])

line4 = []
for i in range(0,14):
    line4.append(matrix[i][4])

line5 = []
for i in range(0,14):
    line5.append(matrix[i][5])

line6 = []
for i in range(0,14):
    line6.append(matrix[i][6])

line7 = []
for i in range(0,14):
    line7.append(matrix[i][7])




plt.figure(figsize=(20,8),dpi=80)

plt.plot(range(0,14),line0, color="red")
plt.plot(range(0,14),line1, color="brown")
plt.plot(range(0,14),line2, color="green")
plt.plot(range(0,14),line3, color="blue")
plt.plot(range(0,14),line4, color="orange")
plt.plot(range(0,14),line5, color="pink")
plt.plot(range(0,14),line6, color="purple")
plt.plot(range(0,14),line7, color="grey")




df = pd.read_csv("data.csv")
df = pd.DataFrame(df)
train_x=df.iloc[:-1, 0:-1]
train_y=df.iloc[:-1, -1:]
test_x=df.iloc[-1:, 0:-1]
test_y=df.iloc[-1:, -1:]

p=-0.1
E_list=[]
for i in range(0,501):
    p = p + 0.1
    alpha = round(p,1)
    E = 0
    for n in range(0,38):
        Reg = Ridge(alpha)
        Reg.fit(train_x,train_y)
        pre = Reg.predict(test_x)
        err = (pre - test_y)**2
        E = E + err
    E = E / 38
    E = E.values
    E_list.append(E[0][0])

print(E_list)
plt.figure(figsize=(20,8),dpi=80)
plt.plot(range(0,501),E_list, color="red")
plt.show()