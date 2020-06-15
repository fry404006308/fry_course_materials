/*

分析：

01背包本身是可以进行空间优化的
因为动态规划本质上是填表
f[i][j]表示前i件物品中总重量为j方案总数
f[i][j]=f[i-1][j] + f[i-1][j-w[i]]; 
填表的顺序为：
i是从1-num
j是从1-1000
是用的2维的表格
01背包问题用一维表格也可以实现保存中间状态
具体实现就是去掉i这一维，

f[j]=f[j] + [j-w[i]]; 
只不过这个时候，填表的顺序就是
i是从1-num
j是从1000-1

*/
#include <iostream>
using namespace std;
int f[1005]={0};
int main(){
    //1、统计砝码总数，准备好砝码序列
    int num=0;//砝码总数
    int w[1005];//砝码序列
    int a[7]={0,1,2,3,5,10,20};
    for(int i=1;i<=6;i++){
        int x;
        cin>>x;
        for(int j=1;j<=x;j++) w[++num]=a[i];
    }
    //2、初始化动态规划数组，做动态规划
    f[0]=1;
    for(int i=1;i<=num;i++){
        for(int j=1000;j>=1;j--){
            if(j-w[i]>=0)
            f[j]=f[j] + f[j-w[i]];
        }
    }
    //3、统计方案总数
    int count=0;
	for(int i=1;i<=1000;i++){
		if(f[i]) count++;
	}
	cout<<"Total="<<count<<endl;
    return 0;
}



