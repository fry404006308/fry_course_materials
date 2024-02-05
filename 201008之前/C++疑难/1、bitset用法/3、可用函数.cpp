#include <iostream>
#include <bitset>
using namespace std;
int main(){

    // bitset<8> foo ("10011011");
    // cout << foo.count() << endl;//5　　（count函数用来求bitset中1的位数，foo中共有５个１
    // cout << foo.size() << endl;//8　　（size函数用来求bitset的大小，一共有８位

    // cout << foo.test(0) << endl;//true　　（test函数用来查下标处的元素是０还是１，并返回false或true，此处foo[0]为１，返回true
    // cout << foo.test(2) << endl;//false　　（同理，foo[2]为０，返回false

    // cout << foo.any() << endl;//true　　（any函数检查bitset中是否有１
    // cout << foo.none() << endl;//false　　（none函数检查bitset中是否没有１
    // cout << foo.all() << endl;//false　　（all函数检查bitset中是全部为１

    // //补充说明一下：test函数会对下标越界作出检查，而通过 [ ] 访问元素却不会经过下标检查，所以，在两种方式通用的情况下，选择test函数更安全一些


    // bitset<8> foo ("10011011");

    // cout << foo.flip(2) << endl;//10011111　　（flip函数传参数时，用于将参数位取反，本行代码将foo下标２处"反转"，即０变１，１变０
    // cout << foo.flip() << endl;//01100000　　（flip函数不指定参数时，将bitset每一位全部取反

    // cout << foo.set() << endl;//11111111　　（set函数不指定参数时，将bitset的每一位全部置为１
    // cout << foo.set(3,0) << endl;//11110111　　（set函数指定两位参数时，将第一参数位的元素置为第二参数的值，本行对foo的操作相当于foo[3]=0
    // cout << foo.set(3) << endl;//11111111　　（set函数只有一个参数时，将参数下标处置为１

    // cout << foo.reset(4) << endl;//11101111　　（reset函数传一个参数时将参数下标处置为０
    // cout << foo.reset() << endl;//00000000　　（reset函数不传参数时将bitset的每一位全部置为０


    //一些类型转换的函数
    bitset<8> foo ("10011011");

    string s = foo.to_string();//将bitset转换成string类型
    unsigned long a = foo.to_ulong();//将bitset转换成unsigned long类型
    unsigned long long b = foo.to_ullong();//将bitset转换成unsigned long long类型

    cout << s << endl;//10011011
    cout << a << endl;//155
    cout << b << endl;//155


    return 0;
}


