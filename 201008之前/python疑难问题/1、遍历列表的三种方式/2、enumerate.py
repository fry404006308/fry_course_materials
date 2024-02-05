
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
print(enumerate(seasons))
# <enumerate object at 0x017C28A0>
list2=list(enumerate(seasons))
print(list2)
# [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]


"""
存在一个sequence，对其使用enumerate将会得到如下结果：
    start     sequence[0]
    start+1　 sequence[1]
    start+2   sequence[2]
    ......
"""

