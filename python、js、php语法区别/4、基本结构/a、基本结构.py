"""
1、选择结构

2、三元运算符

3、循环结构
    a、while循环
    b、for循环
    c、循环列表
    d、循环字符串

"""
# 1、选择结构
# score=12;
# if(score>=80):
#     print("优秀")
#     pass
# elif(score>=60):
#     print("及格")
#     pass
# else:
#     print("不及格")

# 2、三元运算符
# a=155
# b=20
# max_num=a if a>=b else b
# print(max_num)

# 3、循环结构
# 3.1、while循环
# a=10
# while(a>=0):
#     print(a);
#     a-=2;  
# else:
#     print("a小于0")

# 3.2、for循环
# for i in range(1,5):
#     print(i)
#     pass
# else:
#     print("循环做完了")

# 3.3、遍历列表
# list1=[7,432,12,32,12,2];
# for i in list1:
#     print(list1.index(i),i)

# for i in range(len(list1)):
#     print(i,list1[i])

# enumerate() 函数用于将一个可遍历的数据对象
# (如列表、元组或字符串)组合为一个索引序列，
# 同时列出数据和数据下标，一般用在 for 循环当中。
# seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
# list2=list(enumerate(seasons))
# print(list2)

# for i,val in enumerate(list1): 
#     print(i,val)



# 3.4、遍历字符串
# str1="abcdbaba"
# for i in str1:
#     print(str1.index(i),i,id(i))

