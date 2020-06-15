/*

状态：
如果设f[i][j]表示走到(i,j)点的路径总数

递推表达式（状态转移方程）：
f[i][j]=f[i-1][j]+f[i][j-1]

初始状态：
f[0][0]=1
假设马控制的点的坐标为(mx,my)，那么f[mx][my]始终为0

*/
#include <iostream>
#include <cstring>
using namespace std;
int hx[9]={0,-2,-1,1,2,2,1,-1,-2};
int hy[9]={0,1,2,2,1,-1,-2,-2,-1};
long long f[25][25];

int main(){
    int bx,by,mx,my;
    cin>>bx>>by>>mx>>my;
    memset(f,-1,sizeof(f));
    f[0][0]=1;

    //将马控制的点加入到f数组
    for(int i=0;i<=8;i++){
        int now_x=mx+hx[i];
        int now_y=my+hy[i];
        if(now_x>=0&&now_y>=0){
            f[now_x][now_y]=0;
        }
    }

    //做动态规划
    for(int i=0;i<=bx;i++){
        for(int j=0;j<=by;j++){
            //if(i||j)
            //动态规划填表的时候，对于能够填表的点才填表
            //不能填表的点比如初始状态就不能动
            if(f[i][j]==-1)
            {
                //f[i][j]=f[i-1][j]+f[i][j-1]
                if(i-1>=0&&j-1>=0) f[i][j]=f[i-1][j]+f[i][j-1];
                else if(i-1>=0) f[i][j]=f[i-1][j];
                else if(j-1>=0) f[i][j]=f[i][j-1];
                else f[i][j]=0;   
            }
        }
    }

    cout<<f[bx][by]<<endl;
    return 0;
}




