/*
有n级的台阶，你一开始在底部，
每次可以向上迈最多n级台阶，
问到达第n级台阶有多少种不同方式。


题目位置：
变态跳台阶_牛客网
https://www.nowcoder.com/practice/22243d016f6b47f2a6928b4313c85387?tpId=13&&tqId=11162&rp=1&ru=/activity/oj&qru=/ta/coding-interviews/question-ranking

*/

/*
分析：

f(1)=1
f(2)=2
f(3)=f(2)+f(1)+1=4
f(4)=f(3)+f(2)+f(1)+1=8
f(5)=f(4)+f(3)+f(2)+f(1)+1=16

如果发现数字规律，可以直接
f(n)=2^(n-1)

如果没发现数字规律，可以用递推公式
f(n)=f(n-1)+f(n-2)+...+f(1)+1

*/
#include <iostream>
using namespace std;
int f[10000]={0};
int main(){
    int n;
    cin>>n;
    f[1]=1;
    f[2]=2;
    //这层循环用来求f[i]
    for(int i=3;i<=n;i++){
        f[i]=1;
        for(int j=1;j<=i-1;j++){
            f[i]+=f[j];
        }
    }
    cout<<f[n]<<endl;
    return 0;
}


