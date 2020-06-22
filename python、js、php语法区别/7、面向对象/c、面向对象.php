<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <?php
    class Animal{
        function __construct($name) {
            $this->name = $name;
        }
        function say(){
            echo "我是".$this->name."<br>";
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
            echo "我是{$this->name}，我今年{$this->age}岁，我在自由自在的飞翔"."<br>";
        }
    }
    $monkey = new Bird("飞猴",13);
    $monkey->say();
    
    ?>
</body>
</html>






