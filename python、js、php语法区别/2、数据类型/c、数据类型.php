<?php
/*
1、数据类型
Integer（整型）, Float（浮点型）,
String（字符串）, Boolean（布尔型）, 
Array（数组）, Object（对象）, 
NULL（空值）


2、查看变量的类型
gettype(传入一个变量) 能够获得变量的类型
var_dump(传入一个变量) 输出变类型和值

判断变量的数据类型    
is_int 是否为整型
is_bool 是否为布尔
is_float 是否是浮点
is_string 是否是字符串
is_array 是否是数组
is_object 是否是对象
is_null 是否为空
is_resource 是否为资源
is_scalar 是否为标量
is_numeric 是否为数值类型
is_callable 是否为函数

3、类型转换
    a、隐式转换
        $a=true;
        $b=$a+10;

    b、显示转换
        $a=true;
        $d=(int)$a;

    c、函数转换
        $a=true;
        $f=intval($a);


*/

//1、数据类型
//2、查看变量的类型
// $a = 10;
// echo gettype($a); //integer
// echo "\n";
// var_dump($a); //int(10)

// $a = 10.1;
// echo gettype($a); //double
// echo "\n";
// var_dump($a); //float(10.1)

// $a = true;
// echo gettype($a); //boolean
// echo "\n";

// $a = "10";
// echo gettype($a); //string
// echo "\n";

// $a = [1, 2, 3];
// echo gettype($a); //array
// echo "\n";

// $a = null;
// echo gettype($a); //NULL
// echo "\n";

// class Person
// {
//     private $name;
//     public function __construct($name)
//     {
//         $this->name = $name;
//     }
//     public function get_name()
//     {
//         return $this->name;
//     }
// }

// $a = new Person('aaaa');
// //调用方法用->
// echo $a->get_name()."\n"; //aaaa
// echo gettype($a); //object
// echo "\n";

// echo is_int(10)."_\n";
// echo is_int(11.1)."_\n";
// echo is_int(true)."_\n";


//3、类型转换
//a、隐式转换
// $a=true;
// echo gettype($a)."\n"; //boolean

// $b=$a+10;
// $c=$a+10.1;
// echo $b."\n";
// echo gettype($b)."\n"; //integer

// echo $c."\n";
// echo gettype($c)."\n"; //double

//b、显示转换
// $a=true;
// $d=(int)$a;
// echo $d."\n";
// echo gettype($d)."\n"; //integer
/*
(int)(integer)
(bool)(boolean)
(float)(real)
(string)
(array)
(object)
*/

//c、函数转换
// $a=true;
// $f=intval($a);
// echo $f."\n";
// echo gettype($f)."\n"; //integer
/*
intval() 转换为整型
floatval() 转换为浮点数
strval() 转换为字符串
*/    

// settype($a,'string');
// echo $a."\n";
// echo gettype($a)."\n"; //string

?>