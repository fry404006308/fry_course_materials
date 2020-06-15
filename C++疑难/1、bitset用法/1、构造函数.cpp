#include <iostream>
#include <bitset>
using namespace std;
int main(){
    // bitset<4> bitset1;//无参构造，长度为４，默认每一位为０
    // bitset<8> bitset2(12);//长度为８，二进制保存，前面用０补充

    // string s = "100101";
    // bitset<10> bitset3(s);//长度为10，前面用０补充
    
    // char s2[] = "10101";
    // bitset<13> bitset4(s2);//长度为13，前面用０补充

    // cout << bitset1 << endl;//0000
    // cout << bitset2 << endl;//00001100
    // cout << bitset3 << endl;//0000100101
    // cout << bitset4 << endl;//0000000010101


    cout<<"==================================="<<endl;
    cout<<"==================================="<<endl;

    bitset<2> bitset1(12);//12的二进制为1100（长度为４），但bitset1的size=2，只取后面部分，即00
    
    string s = "100101";
    bitset<4> bitset2(s);//s的size=6，而bitset的size=4，只取前面部分，即1001
    
    char s2[] = "11101";
    bitset<4> bitset3(s2);//与bitset2同理，只取前面部分，即1110

    cout << bitset1 << endl;//00
    cout << bitset2 << endl;//1001
    cout << bitset3 << endl;//1110



    return 0;
}


