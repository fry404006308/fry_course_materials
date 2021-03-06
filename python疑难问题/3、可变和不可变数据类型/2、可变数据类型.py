"""
可变数据类型：列表list和字典dict；

1、改变值，变量的地址不会改变
允许变量的值发生变化，即如果对变量进行append、+=等这种操作后，
只是改变了变量的值，而不会新建一个对象，变量引用的对象的地址也不会变化，

2、值相同的两个变量，地址是不一样的
不过对于相同的值的不同对象，在内存中则会存在不同的对象，
即每个对象都有自己的地址，相当于内存中对于同值的对象保存了多份，
这里不存在引用计数，是实实在在的对象。

"""
list1=[1,2,3,4,5]
print(id(list1)) #57361872
list1.append(6)
print(list1)
print(id(list1)) #57361872
print("---------------------------")
list2=[1,2,3,4,5]
print(id(list2)) #59656792
list2.append(6)
print(list2)
print(id(list2)) #59656792

