/*

把整个图向右下移动2，
右下移2也就是把每个点的横纵坐标都加2

A是A点 , B是B点, M是马的位置, X是被马拦着不能走的点
A 0 0 0 0 0 0
0 0 X 0 X 0 0
0 X 0 0 0 X 0
0 0 0 M 0 0 0
0 X 0 0 0 X 0
0 0 X 0 X 0 0
0 0 0 0 0 0 B

取正后，也就是每个点都右下移2后
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 A 0 0 0 0 0 0
0 0 0 0 X 0 X 0 0
0 0 0 X 0 0 0 X 0
0 0 0 0 0 M 0 0 0
0 0 0 X 0 0 0 X 0
0 0 0 0 X 0 X 0 0
0 0 0 0 0 0 0 0 B

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
    bx+=2;by+=2;mx+=2;my+=2;
    memset(f,-1,sizeof(f));
    f[2][2]=1;
    for(int i=1;i<=bx;i++) f[i][1]=0; 
    for(int i=1;i<=by;i++) f[1][i]=0; 
    //将马控制的点加入到f数组
    for(int i=0;i<=8;i++){
        f[mx+hx[i]][my+hy[i]]=0;
    }

    //做动态规划
    for(int i=2;i<=bx;i++){
        for(int j=2;j<=by;j++){
            //if(i||j)
            //动态规划填表的时候，对于能够填表的点才填表
            //不能填表的点比如初始状态就不能动
            if(f[i][j]==-1)
            {
                //f[i][j]=f[i-1][j]+f[i][j-1]
                f[i][j]=f[i-1][j]+f[i][j-1];
            }
        }
    }

    cout<<f[bx][by]<<endl;
    return 0;
}

