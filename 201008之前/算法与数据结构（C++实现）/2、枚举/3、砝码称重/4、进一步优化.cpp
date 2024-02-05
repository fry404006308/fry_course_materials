/*
砝码称重(NOIP1996)

【问题描述】
设有1g、2g、3g、5g、10g、20g的砝码各若干枚（其总重<=1000），
求用这些砝码能称出不同的重量的个数。
【输入文件】
        输入1g、2g、3g、5g、10g、20g的砝码个数
【输出文件】
        能称出不同的重量的个数
【输入样例】
        1 1 0 0 0 0 
【输出样例】

分析：

f[j]=f[j] + [j-w[i]]; 
只不过这个时候，填表的顺序就是
i是从1-num
j是从1000-1

题中说总重<=1000，所以我们的动态规划根据这个1000做循环，
实际上，我们可以根据给的输入数据里面的砝码重量做循环，
因为砝码重量总是小于等于1000的，所以可以进行一定程度的优化


*/

#include <iostream>
using namespace std;
int f[1005]={0};
int main(){
    
    int n[7];
    int weight=0;
    int a[7]={0,1,2,3,5,10,20};
    for(int i=1;i<=6;i++){
        cin>>n[i];
        weight+=n[i]*a[i];
    }
    //2、初始化动态规划数组，做动态规划
    f[0]=1;
    for(int i=1;i<=6;i++){ //对不同型号的砝码进行循环
        for(int k=1;k<=n[i];k++){ //对同一个型号的多个砝码进行循环
            for(int j=weight;j>=a[i];j--){
                f[j]=f[j] + f[j-a[i]];
            }
        }
    }
    //3、统计方案总数
    int count=0;
	for(int i=1;i<=weight;i++){
		if(f[i]) count++;
	}
	cout<<"Total="<<count<<endl;
    return 0;
}




