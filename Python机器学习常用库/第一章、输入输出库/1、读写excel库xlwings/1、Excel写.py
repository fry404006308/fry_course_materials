"""

xlwings介绍
xlwings 是 Python 中操作Excel 的一个第三方库，
支持.xls读写，.xlsx读写
操作非常简单，功能也很强大

1、安装库
pip3 install xlwings

2、引入库
import xlwings as xw

3、
应用->工作簿->工作表->范围

应用：一个应用（一个xlwings程序）：
app = xw.App(visible=True, add_book=False)

工作簿（book）：
excel文件（excel程序）：wb = app.books.add()

工作表（sheet）：
sheet：sht = wb.sheets['sheet1']

范围：行列：
sht.range('a6').expand('table').value = [['a','b'],['d','e']]

xlwings.App(visible=True,add_book=False)
其中参数visible（表示处理过程是否可视，也就是处理Excel的过程会不会显示出来），add_book（是否打开新的Excel程序，也就是是不是打开一个新的excel窗口）

"""
import xlwings as xw

# 写到Excel中去
# add_book也就是是否增加excel 的book
# visible=True 表示操作过程是否可显示
app = xw.App(visible=True, add_book=False)
# 工作簿
wb = app.books.add()

# 页sheet1
sht = wb.sheets['sheet1']
# 单个值插入
# sht.range('A1').value = '产品名称'
# sht.range('B1').value = '编号'
# sht.range('C1').value = '价格'
# sht.range('A2').value = '不告诉你'
# sht.range('B2').value = 'n110110'
# sht.range('C2').value = '688.26'
# sht.range('A3').value = '不告诉你1'
# sht.range('B3').value = 'n1101101'
# sht.range('C3').value = '688.261'

# 插入一行
# sht.range('a1').value = [1,2,3,4]
# 等同于
# sht.range('a1:d4').value = [1,2,3,4]

# 插入一列
# sht.range('a2').options(transpose=True).value = [5,6,7,8]

# 同时插入行列
# sht.range('a6').expand('table').value = [['a','b','c'],['d','e','f'],['g','h','i']]

# 在当前目录下生成文件
wb.save('demo1.xlsx')
wb.close()
app.quit()

# import os
# path1=os.path.abspath('.')   # 表示当前所处的文件夹的绝对路径
# print(path1)
# path2=os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
# print(path2)

# 关于路径问题，切换到指定目录即可





