/*

矩阵的乘法在算法中有很多应用，
比如直接考矩阵的乘法，比如用矩阵优化递推表达式等等


矩阵a*矩阵b 要满足矩阵a的列等于矩阵b的行
最后乘出来的矩阵的行为矩阵a的行
列为矩阵b的列

总结：
矩阵阵法就是按照矩阵相乘的规律，一步步来做的
也就是拿矩阵a的每一行乘以矩阵b的每一列，
并且把矩阵a的每一行里面的每一个元素都和矩阵b里面每一列的每一个元素都一一相乘


矩阵a
1 2 3 
4 5 6

矩阵b
1 2
3 4
5 6


1*1+2*3+3*5

*/

#include <iostream>
#include <cstring>
using namespace std;

struct Matrix{
    int row,column;
    int v[5][5];
    Matrix(){
        memset(v,0,sizeof(v));
    }
};

Matrix multiply(Matrix a,Matrix b){
    Matrix ans;
    ans.row=a.row;
    ans.column=b.column;
    //遍历矩阵a的每一行
    for(int i=1;i<=a.row;i++){
        //遍历矩阵b的每一列
        for(int j=1;j<=b.column;j++){
            //把矩阵a的每一行里面的每一个元素都和矩阵b里面每一列的每一个元素都一一相乘
            for(int k=1;k<=a.column;k++){
                ans.v[i][j]+=a.v[i][k]*b.v[k][j];
            }
        }
    }
    return ans;
}

int main(){
    Matrix a,b,ans;
    a.row=2;a.column=3;
    b.row=3;b.column=2;

    a.v[1][1]=1;a.v[1][2]=2;a.v[1][3]=3;
    a.v[2][1]=4;a.v[2][2]=5;a.v[2][3]=6;

    b.v[1][1]=1;b.v[1][2]=2;
    b.v[2][1]=3;b.v[2][2]=4;
    b.v[3][1]=5;b.v[3][2]=6;

    ans=multiply(a,b);

    cout<<ans.v[1][1]<<endl;

    return 0;
}



