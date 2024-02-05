/*
枚举法

分析：
我们可以直接按照题目的要求来枚举就好了

题目的要求是要 求a[1]-a[n]中连续非空的一段的和最大
那么我们把每个连续的一段都枚举出来，然后来算出里面的和，找出最大值即可

所以在这个需求下：
我们需要枚举每一段的起点、每一段的终点
然后对这一段进行求和

枚举变量：每一段的起点、终点
枚举范围：起点：1-n，终点：起点-n
枚举判断条件：
求和得到每一段的和，在这些和里面选出最大的

时间复杂度：
O(n^3)

算法思路：
1、枚举每一段的起点和终点
2、对每一段进行求和，在这些和里面选出最大的

*/
#include <iostream>
using namespace std;
int a[200005];
int main(){
    int n;
    cin>>n;
    int maxx=-0x7fffffff;
    for(int i=1;i<=n;i++){
        cin>>a[i];
    }
    //1、枚举每一段的起点和终点
    for(int i=1;i<=n;i++){
        for(int j=i;j<=n;j++){
            //2、对每一段进行求和，在这些和里面选出最大的
            int sum=0;
            for(int k=i;k<=j;k++){
                sum+=a[k];
            }
            if(sum>maxx) maxx=sum;
        }
    }
    cout<<maxx<<endl;
    return 0;
}

