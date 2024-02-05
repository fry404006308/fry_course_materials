// (function () {
//     var m='我的模块2';
//     function a() {
//         return m+'：a函数';
//     }
//     function b() {
//         return m+'：b函数';
//     }
//     window.myModule2={
//         a:a,
//         b:b
//     };
// })();

(function (w) {
    var m='我的模块2';
    function a() {
        return m+'：a函数';
    }
    function b() {
        return m+'：b函数';
    }
    w.myModule2={
        a:a,
        b:b
    };
})(window);