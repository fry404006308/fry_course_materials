/*

递推表达式f[i][j]=f[i-1][j]+f[i][j-1]
需要 i-1 和 j-1, 

int hx[9]={0,-2,-1,1,2,2,1,-1,-2};
int hy[9]={0,1,2,2,1,-1,-2,-2,-1};

而初始化马控制的点的时候会有 i-2, j-2 ,

所以我们可以 把整个图向右下移动2，
右下移2也就是把每个点的横纵坐标都加2
这样i-1、j-1、i-2、j-2都不会为负数了
这样就省了很多if判断，
而且这样对问题不会有任何影响


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
long long find(int x,int y){
    //如果缓存里面有，就从缓存里面拿
    //否则计算结果存入缓存
    if(f[x][y]!=-1) return f[x][y];
    else{
        //f[i][j]=f[i-1][j]+f[i][j-1]
        return f[x][y]=find(x-1,y)+find(x,y-1);
    }
}

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

    cout<<find(bx,by)<<endl;
    return 0;
}

