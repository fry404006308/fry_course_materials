/*
最朴素的求幂方法
也就是平常使用pow函数，最简单的实现就是一直累乘

比如求a^n

*/
#include <iostream>
using namespace std;

int pow(int a,int n){
   int ans=1;
   for(int i=1;i<=n;i++){
       ans*=a;
   } 
   return ans;
}

int main(){
    cout<<pow(2,22)<<endl;
    return 0;
}

/*

可以看到，算法的时间复杂度是O(n)。为了降低时间复杂度，
我们可以使用快速幂算法，
将时间复杂度降低到O(logn)。

*/
