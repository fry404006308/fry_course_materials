"""

字典根据键从小到大排序
dict1={"name":"zs","age":18,"city":"深圳","tel":"1366666666"}

"""
dict1={"name":"zs","age":18,"city":"深圳","tel":"1366666666"}
# 将字典转换成元组列表
list1=dict1.items();
print(list1)
# 将元组列表按照元组的第0号元素排序（也就是按照字典的键排序）
list2=sorted(list1,key=lambda i:i[0],reverse=False)
print(list2)
# 将元组列表转化成字典
dict2=dict(list2)
print(dict2)
