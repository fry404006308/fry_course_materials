#include <iostream>
#include <bitset>
using namespace std;
int main(){

    // bitset<4> foo (string("1001"));
    // bitset<4> bar (string("0011"));

    // cout<<foo<<endl; //1001
    // cout<<bar<<endl; //0011
    // cout<<"================================================"<<endl;
    // cout<<"================================================"<<endl;


    // //异或：不同为1，相同为0
    // cout << (foo^=bar) << endl;       //1001异或0011 得 1010 (foo对bar按位异或后赋值给foo)
    // cout << (foo&=bar) << endl;       //1010与0011 得 0010 (按位与后赋值给foo)
    // cout << (foo|=bar) << endl;       //0010或0011 得 0011 (按位或后赋值给foo)

    // cout << (foo<<=2) << endl;        //0011左移两位得 1100 (左移２位，低位补０，有自身赋值)
    // cout << (foo>>=1) << endl;        //1100右移一位得 0110 (右移１位，高位补０，有自身赋值)

    // cout << (~bar) << endl;           //0011取反 1100 (按位取反)
    // cout << (bar<<1) << endl;         // 0110 (左移，不赋值)
    // cout << (bar>>1) << endl;         // 0001 (右移，不赋值)

    // cout << (foo==bar) << endl;       // false (0110==0011为false)
    // cout << (foo!=bar) << endl;       // true  (0110!=0011为true)

    // //foo  0110
    // //bar  0101
    // cout << (foo&bar) << endl;        //0110与0011 得 0010 (按位与，不赋值)
    // cout << (foo|bar) << endl;        // 0111 (按位或，不赋值)
    // cout << (foo^bar) << endl;        // 0101 (按位异或，不赋值)


    // 可以通过 [ ] 访问元素(类似数组)，注意最低位下标为０
    bitset<4> foo ("1011");
    cout << foo[0] << endl;//1
    cout << foo[1] << endl;//1
    cout << foo[2] << endl;//0


    return 0;
}


