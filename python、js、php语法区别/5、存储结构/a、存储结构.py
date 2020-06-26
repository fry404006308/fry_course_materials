"""
1、字符串
2、列表（就像js和php中的索引数组）
3、元组（元组可以看做不能修改的列表）
4、字典（就像js和php中的关联数组）
5、集合

"""
# 1、字符串

# a = "hello"
# b = "python"
 
# print("a + b 输出结果：", a + b)
# print("a * 2 输出结果：", a * 2)
# print("a[1] 输出结果：", a[1])
# print("a[1:4] 输出结果：", a[1:4])
 
# if( "h" in a) :
#     print("h 在变量 a 中")
# else :
#     print("h 不在变量 a 中")
 
# if( "m" not in a) :
#     print("m 不在变量 a 中")
# else :
#     print("m 在变量 a 中")


# c = a.capitalize()
# print(c)

# 2、列表
# list1 = ["a", "a", "c", "d", "d"]
# print(list1)

# # 增
# # list1[10] = "f" #IndexError: list assignment index out of range
# # #将对象插入列表
# list1.append("f")
# print(list1)


# # 删
# del list1[2]
# print(list1)

# # 改
# list1[1] = "b"
# print(list1)

# # 查
# print(list1[0])
# # 留头不留尾
# print(list1[1:3]) 

# # 循环
# for i in list1:
#     print(list1.index(i),i)

# # 方法
# list1.reverse()
# print(list1)



# 3、元组
# tuple1=(1,2,3,4,5)
# tuple1[1]=22 #TypeError: 'tuple' object does not support item assignment
# del tuple1[1] #TypeError: 'tuple' object doesn't support item deletion

# tuple2=("a","b")
# tuple3=tuple1+tuple2
# print(tuple3)
# print(len(tuple3))

# 增


# 删 整个元组
# del tuple1
# print(tuple1) #NameError: name 'tuple1' is not defined

# 改

# 查
# print(tuple1[0]) # 1
# print(tuple1[1:3]) # (2, 3)

# 循环
# for i in tuple1:
#     print(i)

# 函数
# print(max(tuple1))


# 4、字典
# dict1={"name":"孙悟空","age":11}
# print(dict1)

# # 增
# dict1["aa"]="bb";
# print(dict1)

# # 删
# del dict1["age"]
# print(dict1)

# # 改
# dict1["name"]="齐天大圣"
# print(dict1)

# # 查
# print(dict1["name"])

# # 循环
# print(dict1.items())
# # items() 方法以列表返回可遍历的(键, 值) 元组数组
# for key,val in dict1.items():
#     print(key,val)


# 5、集合
# 集合就是数学中的集合，元素不会重复
# 集合为啥可以和字典同时都用{}
# 因为字典是键值对，集合就是值，所以不冲突
# set1={1,2,3,1,24,52,2,3}
# print(set1) # {1, 2, 3, 52, 24}

# # 增
# set1.add(9);
# print(set1)

# # 删
# set1.remove(24)
# print(set1)

# # # 改

# # 查
# print(9 in set1) #True
# print(18 in set1) #false 

# # 循环
# for i in set1:
#     print(i)

