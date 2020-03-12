function fn1() {
    var m='我的模块1';
    function a() {
        return m+'：a函数';
    }
    function b() {
        return m+'：b函数';
    }
    return {
        a:a,
        b:b
    };
}