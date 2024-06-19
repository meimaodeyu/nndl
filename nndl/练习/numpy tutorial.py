import numpy as np
'建立一个一维数组 a 初始化为[4,5,6], (1)输出a 的类型（type）(2)输出a的各维度的大小（shape）(3)输出 a的第一个元素（值为4）'
#创建一维组数a并初始化
a = np.arry([4,5,6])

#输出a的类型(type)
print("Type of a: ", type(a))

#输出a的各维度的大小（shape）
print("shape of a: ", a.shape)

#输出a的第一个元素（4)
print("first element of a: ", a[0])

'建立一个二维数组 b,初始化为 [ [4, 5, 6],[1, 2, 3]] (1)输出各维度的大小（shape）(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）'
b = np.array([[4, 5, 6],[1, 2, 3]])

#输出各维度大小（shape）
print("Type of b: ", type(b))

#输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）
print("b(0,0), b(0,1),b(1,1) is", b[0,0], b[0,1], b[1,1])

'(1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2)建立一个全1矩阵b,大小为4x5; (3)建立一个单位矩阵c ,大小为4x4; (4)生成一个随机数矩阵d,大小为 3x2.'
#建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）
a = np.zeros((3,3), dtype=int)
print("Matrix a:\n", a)
#2)建立一个全1矩阵b,大小为4x5;
b = np.ones(4,5)
print("\nMatrix b:\n", b)
#(3)建立一个单位矩阵c ,大小为4x4
c = np.eye(4)
print("\nMatrix c:\n", c)
#生成一个随机数矩阵d,大小为 3x2.
d = np.random.rand(3,2)
print("\nMatrix d:\n", d)

'建立一个数组 a,(值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] ) ,(1)打印a; (2)输出 下标为(2,3),(0,0) 这两个数组元素的值'
a = np.array([[[1, 2, 3,4], [5, 6, 7,8], [9,10,11,12]]])
print("Array a:\n", a)

#输出 下标为(2,3),(0,0) 这两个数组元素的值
print("a(2,3), a(0,0) is",a[2,3],a[0,0])

#a数组的 0到1行 2到3列，放到b里面去，（此处不需要从新建立a,直接调用即可）
a = np.array([[1, 2, 3,4], [5, 6, 7,8], [9,10,11,12]])
b = a[0:2, 2:4]

#输出b
print("Array b:\n", b)

#输出b的(0,0)这个元素的值
print("Value at b(0,0):", b[0, 0])

'把第5题中数组a的最后两行所有元素放到 c中，（提示： a[1:2, :]）(1)输出 c ; (2) 输出 c 中第一行的最后一个元素（提示，使用 -1 表示最后一个元素）'
a = np.array([[1, 2, 3,4], [5, 6, 7,8], [9,10,11,12]])
c = a[1:3,:]
#输出 c
print("Array c:\n",c)

#输出 c 中第一行的最后一个元素（提示，使用 -1 表示最后一个元素）
print("Value of the last element in the first row of c:", c[0, -1])

'建立数组a,初始化a为[[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0）这三个元素（提示： 使用 print(a[[0, 1, 2], [0, 1, 0]]) ）'
a = np.array([[1, 2], [3, 4], [5, 6]])
#a[[0, 1, 2], [0, 1, 0]] 使用的是高级索引（fancy indexing），其中第一个列表 [0, 1, 2] 表示行索引，第二个列表 [0, 1, 0] 表示列索引。
print(a[[0, 1, 2], [0, 1, 0]])

'建立矩阵a ,初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，输出(0,0),(1,2),(2,0),(3,1) (提示使用 b = np.array([0, 2, 0, 1]) print(a[np.arange(4), b]))'
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
'np.arange(4) 生成一个数组 [0, 1, 2, 3]，这是行索引,b = np.array([0, 2, 0, 1]) 是列索引，表示要取每行对应的列,a[np.arange(4), b] 使用了高级索引，它会同时指定行和列的索引，即 (0,0), (1,2), (2,0), (3,1) 这四个位置的元素'
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])

'对9 中输出的那四个元素，每个都加上10，然后重新输出矩阵a.(提示： a[np.arange(4), b] += 10 ）'
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
a[np.arange(4), b] += 10
print("Updated matrix a:\n", a)

'执行 x = np.array([1, 2])，然后输出 x 的数据类型'
x = np.array([1,2])
print("Data type of x:", x.dtype)

'执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型(type）'
x = np.array([1.0, 2.0])
print("Type of the Data Structure:", type(x))

'执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)'
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
z = x+y
print("The result",z)
print("np.add(x,y) is",np.add(x,y))

'利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)'
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

print("x-y \=n",x-y)
print("np.subtract(x,y) \=n",np.subtract(x,y))

' 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有 np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。'
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
# 逐元素相乘
elementwise_product = x * y
np_elementwise_product = np_multiply(x,y)

#矩阵乘法
dot_product = np.dot(x,y)

print("x * y =\n", elementwise_product)
print("np.multiply(x, y) =\n", np_elementwise_product)
print("np.dot(x,y) =\n", dot_product)

#尝试不是方阵
# 创建两个新的 NumPy 数组
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)

# 逐元素相乘（将会报错，因为形状不匹配）
try:
    elementwise_product_ab = a * b
except ValueError as e:
    elementwise_product_ab = str(e)

try:
    np_elementwise_product_ab = np.multiply(a, b)
except ValueError as e:
    np_elementwise_product_ab = str(e)

# 矩阵乘法
dot_product_ab = np.dot(a, b)

# 输出结果
print("a * b =\n", elementwise_product_ab)
print("np.multiply(a, b) =\n", np_elementwise_product_ab)
print("np.dot(a, b) =\n", dot_product_ab)