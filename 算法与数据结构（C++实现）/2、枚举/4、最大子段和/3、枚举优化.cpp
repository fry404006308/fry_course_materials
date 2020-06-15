/*
枚举优化

可以把求和的那层循环去掉，我们可以对数据做预处理
用s[i]表示第一个数到第i个数这个序列的和

那么求s[i-j]（第i个数到第j个数这个序列的和）的时候，
可以直接用s[j]-s[i]+a[i]即可
s[j]-s[i]表示的是i+1到j这个序列的和，所以需要加上a[i]

现在的时间复杂度：
O(n)+O(n^2)=O(n^2)

优化方法：
减少重复计算


*/
#include <iostream>
using namespace std;
int a[200005];
int s[200005]={0};
int main(){
    int n;
    cin>>n;
    int maxx=-0x7fffffff;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        s[i]=s[i-1]+a[i];
    }
    //1、枚举每一段的起点和终点
    for(int i=1;i<=n;i++){
        for(int j=i;j<=n;j++){
            //2、对每一段进行求和，在这些和里面选出最大的
            int sum=s[j]-s[i]+a[i];
            if(sum>maxx) maxx=sum;
        }
    }
    cout<<maxx<<endl;
    return 0;
}

