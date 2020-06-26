
'''
1、python没有自增运算符

可以用 x+=1 来代替
比如：
a = 3
a += 1
+=是相当于重新生成了一个值为4的对象，把操作后的对象重新赋予给变量a。
但是++的话代表改变了对象本身，注意是对象本身，而不是变量本身。
a++
这个3是整型，也是数值型，python中的数值型就是不可变的数据类型。
所以不可以++

2、python 赋值号左右两边对等位置赋值
a, b = 0, 1

3、python算术运算符中的注意点
**表示指数，比如7**3 表示7的3次方
//表示整除 7//3=2

4、python中的逻辑运算符：
and or not 注意是英文而不是符号

'''

# a = 10
# a += 1
# a++
# print(a)

# a,b=2,3
# for i in range(1,5):
#     a,b=i,i*i
# print(a)
# print(b)

# print(pow(7,3))
# print(7**3)

# print(7/3)
# print(7//3)

# a=True
# b=False
# if(a||b):
#     print('ok')

# if(a or b):
#     print('ok1')

# if(a and b):
#     print('ok2')

# if(not b):
#     print('ok3')



