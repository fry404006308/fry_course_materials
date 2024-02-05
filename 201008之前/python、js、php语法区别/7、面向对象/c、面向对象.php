<?php
/*
php面向对象
*/
class Animal{
    function __construct($name) {
        $this->name = $name;
    }
    function say(){
        echo "我是".$this->name."\n";
    }
}
$animal1 = new Animal("大动物");
$animal1->say();

class Bird extends Animal{
    function __construct($name,$age) {
        parent::__construct($name);
        $this->age = $age;
    }
    function say(){
        echo "我是{$this->name}，我今年{$this->age}岁，我在自由自在的飞翔"."\n";
    }
}
$monkey = new Bird("大飞猴",13);
$monkey->say();

?>






