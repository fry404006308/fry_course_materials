import xlwings as xw

app = xw.App(visible=True, add_book=False)
app.display_alerts = True
app.screen_updating = True
# 打开文件

wb = app.books.open('第一章、输入输出库\\2、表格操作实例\\283地级市的欧氏距离.xlsx')
sht = wb.sheets['283地级市欧式直线距离（千米）']

# 读取行列
list_data = sht.range('a2:i80090').value
print(len(list_data))
print(list_data[0]) 
print(list_data[0][1]) 
print(list_data[0][5]) 
# ['北京市', 1.0, 39.9361, 116.412, '北京市', 1.0, 39.9361, 116.412, 0.0]
print(list_data[1]) 
# ['天津市', 2.0, 39.1127, 117.195, '北京市', 1.0, 39.9361, 116.412, 113.55159]

# 构建一个283*283的表，也就是 各个城市之间的 欧氏距离
distance = [[0]*283 for _ in range(283) ]
print(len(distance))
print(len(distance[0]))
# print(distance)

# 遍历list_data，将各个城市之间的 欧式距离填到 distance数组之中
for i in list_data:
    distance[int(i[1])-1][int(i[5]-1)] = i[8]
print(distance[0][0])
print(distance[0])

wb.save()
wb.close()
app.quit()


# 将距离数据写到excel中去
# 写到Excel中去
app = xw.App(visible=True, add_book=False)
# 工作簿
wb = app.books.add()

# 页sheet1
sht = wb.sheets['sheet1']

# 同时插入行列
# 将两个城市的欧式距离写到excel表中
sht.range('a1').expand('table').value = distance

# 在当前目录下生成文件
wb.save('第一章、输入输出库\\2、表格操作实例\\格式化后的欧氏距离.xlsx')
wb.close()
app.quit()




