/*

递推关系式：
题目中已经非常明显的给出了，就是
F(n)=F(n-1)+F(n-2)

解决递推问题的一般步骤
1、建立递推关系式：F(n)=F(n-1)+F(n-2)
2、确定边界条件：
f(1)=f(2)=1，
所以我们的循环可以从3开始，到n结束，
也就是3-n

算法步骤：
1、确定初始值
2、循环做递推，3-n

*/
#include <iostream>
using namespace std;
const int mod=1000000007;
int f[200000];
int main(){
    int n;
    cin>>n;
    //1、确定初始值
    f[1]=f[2]=1;
    //2、循环做递推，3-n
    for(int i=3;i<=n;i++){
        //F(n)=F(n-1)+F(n-2)
        f[i]=(f[i-1]+f[i-2])%mod;
    }
    cout<<f[n]<<endl;
    return 0;
}


