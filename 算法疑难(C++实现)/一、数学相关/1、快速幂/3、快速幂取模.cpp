/*
因为求幂运算得到的结果一般很大
所以一般需要对求幂的结果进行取模
所以下面就来说说快速幂取模的操作

根据同余定理，我们知道
(a*b)%m = ((a%m)*(b%m))%m；
其实快速幂取模也是用到这个


*/

#include <iostream>
using namespace std;

const int mod=10007;

int pow(int a,int n){
   int ans=1,base=a%mod;
   while(n){
       if(n%2==1) ans=(ans*base)%mod;
       base=(base*base)%mod;
       n=n/2;
   }
   return ans;
}

int main(){
    cout<<pow(2,110)<<endl;
    return 0;
}


