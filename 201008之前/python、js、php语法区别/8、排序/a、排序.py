"""

1、list的sort方法
2、全局sorted()方法来对可迭代的序列排序生成新的序列

"""

# 1、list的sort方法

# a、sort升序排列
# list1=[43,54,12,3,1,435]
# list1.sort()
# print(list1)

# b、sort降序排列
# list1=[43,54,12,3,1,435]
# list1.sort(reverse=True)
# print(list1)

# c、对元组列表进行排序，指定按照元组的第二个元素排序

# # 获取列表的第二个元素
# def takeSecond(elem):
#     return elem[1]
# random = [(2, 2), (3, 4), (4, 1), (1, 3)]
# # 指定第二个元素排序
# random.sort(key=takeSecond)
# print (random)


# 2、全局sorted()方法来对可迭代的序列排序生成新的序列

# a、正序
# list1=[43,54,12,3,1,435]
# ans=sorted(list1)
# print(list1)
# print(ans)

# b、倒序
# list1=[43,54,12,3,1,435]
# ans=sorted(list1,reverse=True)
# print(list1)
# print(ans)


# c、利用key，指定对哪个字段进行排序
# students = [
#     ('john', 'A', 15), 
#     ('jane', 'B', 12), 
#     ('dave', 'B', 18)]
# ans = sorted(students, key=lambda s: s[2], reverse=True) # 按年龄倒序排序
# print(ans)

# d、对字典排序
# dict1={'a':1,'c':3,'b':2}
# ans = sorted(dict1)
# print(ans) #['a', 'b', 'c']


# dict1={'a':1,'c':3,'b':2}    
# print(dict1.items())
# # dict1.items()返回的是： dict_items([('a', 1), ('c', 3), ('b', 2)])
# ans=sorted(dict1.items(),key=lambda x:x[1])  
# # 按字典集合中，每一个元组的第二个元素排列。                                                      # x相当于字典集合中遍历出来的一个元组。
# print(ans)
# # 得到:  [('a', 1), ('b', 2), ('c', 3)]
# print(dict(ans))
# # 得到:  {'a': 1, 'b': 2, 'c': 3}
