
/*


注意点：
1、
读入n的时候不能用int，因为1<=n<2^63

2、
注意结构体中的数组要用 long long类型，因为涉及到矩阵的乘法，
涉及到两个数相乘，所以int mod 1000000007之后，两个数相乘，还是会超int

3、
因为读入的n是long long类型，所以函数传递参数的时候，也要记得别用成int了


*/
#include <iostream>
#include <cstring>
using namespace std;
const int mod=1000000007;

//定义矩阵对应的结构体
struct Matrix{
    int row,column;
    long long v[3][3];
    Matrix(){
        memset(v,0,sizeof(v));
    }
};

//矩阵乘法
Matrix multiply(Matrix a,Matrix b){
    Matrix ans;
    ans.row=a.row;
    ans.column=b.column;
    //具体来做矩阵乘法
    for(int i=1;i<=a.row;i++){
        for(int j=1;j<=b.column;j++){
            for(int k=1;k<=a.column;k++){
                ans.v[i][j]+=(a.v[i][k]*b.v[k][j])%mod;
                ans.v[i][j]%=mod;
            }
        }
    }
    return ans;
}


//矩阵的快速幂
Matrix pow(Matrix a,long long n){
    Matrix ans,base=a;
    ans.row=2;ans.column=2;
    ans.v[1][1]=ans.v[2][2]=1;
    while(n){
        if(n%2==1) ans=multiply(ans,base);
        base=multiply(base,base);
        n/=2;
    }
    return ans;
}


int main(){
    long long n;
    cin>>n;
    Matrix ans,base,last;
    //初始化base矩阵
    base.row=2;base.column=2;
    base.v[1][1]=base.v[1][2]=base.v[2][1]=1;
    //初始化last矩阵
    last.row=2;last.column=1;
    last.v[1][1]=last.v[2][1]=1;
    if(n==1||n==2){
        cout<<1<<endl;
    }else{
        ans=pow(base,n-2);
        ans=multiply(ans,last);
        cout<<ans.v[1][1]<<endl;
    }

    return 0;
}









