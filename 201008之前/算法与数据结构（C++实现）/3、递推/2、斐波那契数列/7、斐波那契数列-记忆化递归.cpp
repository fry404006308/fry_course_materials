/*

比如我们要求f[6]
f[6]=f[5]+f[4]
f[5]=f[4]+f[3]
f[4]=f[3]+f[2]
...

我们可以看到，在上述过程中，f[4]、f[3]等都出现了很多次，都被重复计算了很多次
这就是递归效率为什么不高的原因

解决这个问题就用记忆化递归，就是把已经计算的中间状态保存下来，
下次需要的时候就直接拿这个结果，就不用重复计算了


记忆化递归的思想和动态规划的思想是一样的，
都是保存中间计算结果，避免重复计算，拿空间换时间


*/

#include <iostream>
#include <cstring>
using namespace std;
const int mod=1000000007;

int cache[200000];

int find(int n){
    //就是如果缓存中有，就直接拿缓存
    //否则计算，然后将计算的结果保存到缓存
   if(cache[n]!=-1) return cache[n];
   else{
       return cache[n]=(find(n-1)+find(n-2))%mod;
   }
}

int main(){
    int n;
    cin>>n;
    memset(cache,-1,sizeof(cache));
    cache[2]=cache[1]=1;
    cout<<find(n)<<endl;
    return 0;
}


