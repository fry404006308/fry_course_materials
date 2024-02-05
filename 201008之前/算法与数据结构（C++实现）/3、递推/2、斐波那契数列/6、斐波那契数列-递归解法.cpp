/*

本题递推法的递推的关系式非常明确，就是f[i]=f[i-1]+f[i-2]
递推法的递推关系式，对应到递归，就是递归的各个元素之间的关系
明确这个，递归的代码就特别好敲

递归注意点
递归的终止条件：n=2和n=1
递归的递推表达式：f[i]=f[i-1]+f[i-2] (3<=i<=n)
递归的返回值：所求值（斐波那契数列第n项 mod 10^9+7的值）

*/
#include <iostream>
using namespace std;
const int mod=1000000007;

int find(int n){
   if(n==1||n==2) return 1;
   else{
       return (find(n-1)+find(n-2))%mod;
   }
}

int main(){
    int n;
    cin>>n;
    cout<<find(n)<<endl;
    return 0;
}


