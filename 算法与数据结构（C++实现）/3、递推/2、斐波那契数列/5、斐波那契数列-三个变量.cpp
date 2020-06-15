/*
f[3] 可以直接用3个变量a、b、c来代替
这个时候就不能通过取模来自动变换位置了


*/
#include <iostream>
using namespace std;
const int mod=1000000007;
int main(){
    int n;
    int a,b,c;
    cin>>n;
    //1、确定初始值
    //这里对a也赋值为1，是为了保证n=1和n=2的时候也有正确结果输出
    c=a=b=1;
    //2、循环做递推，3-n
    for(int i=3;i<=n;i++){
        //F(n)=F(n-1)+F(n-2)
        c=(b+a)%mod;
        //保留f(n)和f(n-1)做下一轮的f(n-1)和f(n-2)
        a=b;
        b=c;
    }
    cout<<c<<endl;
    return 0;
}


