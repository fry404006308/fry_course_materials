"""
需求：
创建Animal类（name属性，say方法）
创建Animal类的子类Bird类（age属性，say方法）
"""
class Animal:
    def __init__(self,name):
        self.name = name
        pass
    def say(self):
        print("我是{}".format(self.name))

animal1 = Animal("大动物")
animal1.say()

class Bird(Animal):
    def __init__(self,name,age):
        # Animal.__init__(self,name)
        # super(Bird,self).__init__(name)
        super().__init__(name) 
        self.age = age
        pass
    def say(self):
        print("我是{}，我今年{}岁，我在自由自在的飞翔".format(self.name,self.age))

monkey=Bird('大飞猴',15);
monkey.say();


