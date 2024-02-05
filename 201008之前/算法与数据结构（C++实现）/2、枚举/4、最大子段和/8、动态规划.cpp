
/*
动态规划解法

这道题目，也可以用动态规划来做


以第i个数结尾的最大连续子序列的和，只存在两种选择：
情形1：只包含a[i]
情形2：包含a[i]和以a[i-1]结尾的最大连续和序列

所以f[i]表示以第i个元素结尾的子序列的最大值 

f[i]=max(f[i-1]+a[i],a[i]) (2<=i<=n)

所以要求所有序列的最大值，就是
Answer=max{f[i]|1<=i<=n}

初始状态：f[1]=a[1]

算法步骤：
1、确定动态规划初始条件：f[1]=a[1]
2、动态规划操作：f[i]=max(f[i-1]+a[i],a[i]) (2<=i<=n)
3、求ans：Answer=max{f[i]|1<=i<=n}

动态规划能够优化枚举的原理
动态规划能够优化，是因为找准了状态之间的转移关系（找到了题目的规律），并且存储了中间的状态，
减少了大量重复求状态的计算，所以动态规划一般效率非常高

*/

//代码层面优化的代码
#include <iostream>
#include <algorithm>
using namespace std;
int a[200005];
int f[200005]={0};
int main(){
    int n;
    cin>>n;
    cin>>a[1];
    f[1]=a[1];
    //1、确定动态规划初始条件：f[1]=a[1]
    int maxx=f[1];
    for(int i=2;i<=n;i++){
        cin>>a[i];
        //2、动态规划操作：f[i]=max(f[i-1]+a[i],a[i]) (2<=i<=n)
        f[i]=max(f[i-1]+a[i],a[i]);
        //3、求ans：Answer=max{f[i]|1<=i<=n}
        maxx=max(f[i],maxx);
    }
    cout<<maxx<<endl;
    return 0;
}

//未从代码层面优化的代码
// #include <iostream>
// #include <algorithm>
// using namespace std;
// int a[200005];
// int f[200005]={0};
// int main(){
//     int n;
//     cin>>n;
//     for(int i=1;i<=n;i++){
//         cin>>a[i];
//     }
//     //1、确定动态规划初始条件：f[1]=a[1]
//     f[1]=a[1];
//     //2、动态规划操作：f[i]=max(f[i-1]+a[i],a[i]) (2<=i<=n)
//     for(int i=2;i<=n;i++){
//         f[i]=max(f[i-1]+a[i],a[i]);
//     }
//     //3、求ans：Answer=max{f[i]|1<=i<=n}
//     int maxx=-0x7fffffff;
//     for(int i=1;i<=n;i++){
//         maxx=max(f[i],maxx);
//     }
//     cout<<maxx<<endl;
//     return 0;
// }

