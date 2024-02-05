/*

上面贪心代码从代码层面上来看是可以优化的
1、时间方面优化：循环可以合并（循环方向一致，循环最大值也是，并且两个循环之间没有什么逻辑操作代码）
2、空间方面优化：代码中只用到了a[i]，所以a[]数组可以用一个变量来代替

*/
#include <iostream>
#include <algorithm>
using namespace std;
int main(){
    int n;
    cin>>n;
    int sum;
    cin>>sum;
    int maxx=sum;
    for(int i=2;i<=n;i++){
        int x;
        cin>>x;
        if(sum<=0) sum=0;
        sum+=x;
        maxx=max(sum,maxx);
    }
    cout<<maxx<<endl;
    return 0;
}



