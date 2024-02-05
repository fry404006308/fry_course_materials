/*

快速幂：

首先，快速幂的目的就是做到快速求幂，

假设我们要求a^n，那么其实n是可以拆成二进制的，
例如当n==11时
11的二进制是1011，
11 =2º×1+2¹×1+2²×0+2³×1=1+2+8，
所以
a^11=a^1*a^2*a^8
原来算11次，现在只需要算三次

具体怎么算呢：
我们可以用一个变量base来在每次循环的时候记录a^i，
最开始base是a
然后每次循环让base=base*base
那么base的值
a-->a^2-->a^4-->a^8-->a^16-->a^32.......
然后根据11的二进制，1011，
取位为1时候的base值即可，
也就是取a，a^2，a^8

由此可以得到代码：


*/

#include <iostream>
using namespace std;

int pow(int a,int n){
   int ans=1,base=a;
   while(n){
       if(n%2==1) ans=ans*base;
       base=base*base;
       n=n/2;
   }
   return ans;
}

int main(){
    cout<<pow(2,22)<<endl;
    return 0;
}



/*

n 11
ans a
base a^2

n 5
ans a*a^2
base a^2*a^2=a^4

n 2
ans a*a^2
base a^4*a^4=a^8

n 1
ans a*a^2*a^8
base a^8*a^8=a^16

n 0

*/
