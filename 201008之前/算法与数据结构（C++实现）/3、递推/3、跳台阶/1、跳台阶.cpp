/*
题目描述
有N级的台阶，你一开始在底部，
每次可以向上跳一步或者跳两步，
问到达第N级台阶总共有多少种不同方式。

输入格式
一个数字，楼梯数。

输出格式
走的方式几种。

输入输出样例
输入
4
输出
5

数据范围
60%，N<=50
100%，N<=5000


题目位置：
P1255 数楼梯 - 洛谷 | 计算机科学教育新生态
https://www.luogu.com.cn/problem/P1255


*/

/*

分析：
第1级台阶有一种方式
第2级台阶有两种方式

第3级台阶可以由第1阶走两步或第2阶走一步得出。1+2=3
第4级台阶由第2阶走两步或第3阶走一步得出。2+3=5
第5级台阶由第3阶走两步或第4阶走一步得出。3+5=8
...
所以
第n级台阶由第n-2阶走两步或第n-1阶走一步得出。

所以如果用 f(n)表示跳到第n级台阶的总方式数
所以 f(n)=f(n-1)+f(n-2)


*/

#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

//高精度数对应的结构体
struct BigNum{
    int length=0;
    int v[5000];
    BigNum(){
        memset(v,0,sizeof(v));
    }
};

//高精度加法
BigNum add(BigNum a,BigNum b){
    BigNum ans;
    ans.length=max(a.length,b.length);
    //逐位相加
    for(int i=0;i<=ans.length-1;i++){
        ans.v[i]=a.v[i]+b.v[i];
    }
    //处理进位
    for(int i=0;i<=ans.length-1;i++){
        if(ans.v[i]/10){
            ans.v[i+1]+=ans.v[i]/10;
            ans.v[i]%=10;
        }
    }
    //判断最高位的数
    if(ans.v[ans.length]) ans.length++;
    return ans;
}


int main(){
    int n;
    BigNum a,b,c;
    cin>>n;
    //1、确定初始值
    a.length=1;b.length=1;
    a.v[0]=1;
    b.v[0]=2;
    //2、循环做递推，3-n
    for(int i=3;i<=n;i++){
        //F(n)=F(n-1)+F(n-2)
        c=add(b,a);
        //保留f(n)和f(n-1)做下一轮的f(n-1)和f(n-2)
        a=b;
        b=c;
    }
    if(n==0) cout<<0<<endl;
    else if(n==1) cout<<1<<endl;
    else if(n==2) cout<<2<<endl;
    else{
        for(int i=c.length-1;i>=0;i--){
            cout<<c.v[i];
        }
        cout<<endl;
    }
    return 0;
}

