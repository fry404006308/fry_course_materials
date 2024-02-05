/*

我们来看一眼代码：
代码中用到a数组位置除了a[1]这个固定的，剩下的就是a[i]
所以a数组可以被一个变量代替。


然后再来看一眼f数组
f数组：全程我们只用到了f[i]元素和f[i-1]元素
是不是闻到了滚动数组的气息
最终我们可以得出空间优化版

*/
#include <iostream>
#include <algorithm>
using namespace std;
int f[2]={0};
int main(){
    int n;
    cin>>n;
    cin>>f[1];
    //1、确定动态规划初始条件：f[1]=a[1]
    int maxx=f[1];
    for(int i=2;i<=n;i++){
        int x;
        cin>>x;
        //2、动态规划操作：f[i]=max(f[i-1]+a[i],a[i]) (2<=i<=n)
        f[i%2]=max(f[!(i%2)]+x,x);
        //3、求ans：Answer=max{f[i]|1<=i<=n}
        maxx=max(f[i%2],maxx);
    }
    cout<<maxx<<endl;
    return 0;
}









