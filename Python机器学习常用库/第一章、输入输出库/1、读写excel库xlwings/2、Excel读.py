
import xlwings as xw

app = xw.App(visible=True, add_book=False)
# 显示警报（）
app.display_alerts = True
# 屏幕更新（应用中）
app.screen_updating = True
# 打开文件

wb = app.books.open('demo1.xlsx')
sht = wb.sheets['sheet1']

# 遍历读取单元格
# column_name = ['A','B',"C"]
# data_list = [] #将数据存到list中去
# for i in range(3): # 遍历行
#     row_list = []
#     for j in range(3): #遍历列
#         str1 = column_name[j]+str(i+1)
#         a = sht.range(str1).value
#         row_list.append(a)
#         print(a)
#         pass
#     data_list.append(row_list)
#     pass
# print(data_list)

# 读取行列：读取A1:C7（直接填入单元格范围就行了）,得到一个二维列表
print(sht.range('a1:c7').value)

# 读取行：得一维列表 
# print(sht.range('a1:c1').value)

# 读取列：得一维列表
# print(sht.range('a1:a7').value)


wb.save()
wb.close()
app.quit()






