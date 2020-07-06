
"""
# 第一步：获取每个城市的GDP
# 第二步：获取城市之间的欧氏距离
# 第三步：获取城市行政区域面积
# 第四步：根据公式计算城市潜能
# 第五步：将计算的潜能存入excel表
"""

import xlwings as xw

# 第一步：获取每个城市的GDP
def get_gdp():
    app = xw.App(visible=True, add_book=False)
    app.display_alerts = True
    app.screen_updating = True
    # 打开文件

    wb = app.books.open('第一章、输入输出库\\2、表格操作实例\\283地级市GDP.xlsx')
    sht = wb.sheets['地区生产总值（万元）']

    # 读取行列
    gdp_list = sht.range('a2:d3680').value
    # print(len(gdp_list)) #3679
    # print(gdp_list[0]) #['北京市', 1.0, 2003.0, 35572637.0]
    # print(gdp_list[3678]) #['克拉玛依市', 283.0, 2015.0, 4330085.579449926]

    wb.save()
    wb.close()
    app.quit()
    return gdp_list
gdp_list = get_gdp()

# 第二步：获取城市之间的欧氏距离
def get_distance():
    app = xw.App(visible=True, add_book=False)
    app.display_alerts = True
    app.screen_updating = True
    # 打开文件

    wb = app.books.open('第一章、输入输出库\\2、表格操作实例\\格式化后的欧氏距离.xlsx')
    sht = wb.sheets['Sheet1']

    # 读取行列
    distance_list = sht.range('a1:jw283').value
    # print(len(distance_list)) #283
    # print(distance_list[0]) 
    # print(distance_list[282])

    wb.save()
    wb.close()
    app.quit()
    return distance_list
distance_list = get_distance()

# 第三步：获取城市行政区域面积
def get_area():
    app = xw.App(visible=True, add_book=False)
    app.display_alerts = True
    app.screen_updating = True
    # 打开文件

    wb = app.books.open('第一章、输入输出库\\2、表格操作实例\\283地级市的欧氏距离.xlsx')
    sht = wb.sheets['283地级市行政区域面积（平方公里）']

    # 读取行列
    city_area_list = sht.range('a2:d3680').value
    # print(len(city_area_list)) #3679
    # print(city_area_list[0]) #['北京市', 1.0, 2003.0, 12484.0]
    # print(city_area_list[3678]) #['克拉玛依市', 283.0, 2015.0, 7735.0]

    wb.save()
    wb.close()
    app.quit()
    return city_area_list
city_area_list = get_area()

# 第四步：根据公式计算城市潜能
"""
输出数据格式：
城市名 城市id 年份 城市潜能

"""
import copy
import math
pi= math.pi

city_potential_list = copy.deepcopy(gdp_list)
# print(len(city_potential_list))
# print(city_potential_list[0])
for i in range(283): # 0-282 对城市进行遍历，计算每个城市的潜能
    for y in range(13): # 0-12 对年份进行遍历，计算每个城市不同年份的潜能
        potential = 0
        yi = gdp_list[(i+1)*(y+1)-1][3]
        area = city_area_list[(i+1)*(y+1)-1][3]
        dii = (2/3)*math.sqrt(area/pi)
        potential += yi/dii
        for j in range(283): # 0-282 对公式中的j进行遍历
            yj = gdp_list[(j+1)*(y+1)-1][3]
            dij = distance_list[i][j]
            if i!=j :
                potential += yj/dij
        city_potential_list[(i+1)*(y+1)-1][3] = potential
        pass
    pass

# 第五步：将计算的潜能存入excel表
def save_city_potentail(city_potential_list):
    # 写到Excel中去
    app = xw.App(visible=True, add_book=False)
    # 工作簿
    wb = app.books.add()

    # 页sheet1
    sht = wb.sheets['sheet1']

    # 同时插入行列
    # 将两个城市的欧式距离写到excel表中
    sht.range('A1').value = 'city'
    sht.range('B1').value = 'city_id'
    sht.range('C1').value = 'year'
    sht.range('D1').value = 'potentail'
    sht.range('a2').expand('table').value = city_potential_list

    # 在当前目录下生成文件
    wb.save('第一章、输入输出库\\2、表格操作实例\\城市潜能.xlsx')
    wb.close()
    app.quit()
save_city_potentail(city_potential_list)





