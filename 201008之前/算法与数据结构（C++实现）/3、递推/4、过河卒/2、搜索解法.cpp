/*

递推表达式：
卒子可以向下走和向右走
如果设f[i][j]表示走到(i,j)点的路径总数
对应的走到f[i][j]只能从上边来或者从左边来
f[i][j]=f[i-1][j]+f[i][j-1]


简单的思考：
如果没有这个马，搜索应该怎么做

递归：
递归的终止条件：起点
递归的递推表达式：f[i][j]=f[i-1][j]+f[i][j-1]
递归的返回值：路径条数

初始值：f[0][0]=1

如果有马的情况
递归的终止条件：起点或者马控制的区域


注意：
1、本题的路径条数是超过int的，所以要用long long
2、使用递推表达式f[i][j]=f[i-1][j]+f[i][j-1]时，
因为有i-1、j-1，所以要考虑i、j是否大于1的情况
3、初始化的时候，不能直接初始化i=0和j=0对应的两条线，
因为当马的控制点在这两条线上时，控制点后的点是达不到的


思考：
1 1 1 1 1 1 1
1 2 X 1 X 1 2
1 X 0 1 1 X 2
1 1 1 M 1 1 3
1 X 1 1 0 X 3
1 1 X 1 X 0 3
1 2 2 3 3 3 6
这里初始化的时候能直接初始化i=0和j=0对应的两条线么
不能，因为如果这样初始化后，当马的位置如果是(4,0)，
那么(5,0)的位置本来是去不了的，
但是这样初始化却会初始化为1


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
        if(x-1>=0&&y-1>=0) return f[x][y]=find(x-1,y)+find(x,y-1);
        else if(x-1>=0) return f[x][y]=find(x-1,y);
        else if(y-1>=0) return f[x][y]=find(x,y-1);
        else return f[x][y]=0;
    }
}

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

    cout<<find(bx,by)<<endl;
    return 0;
}

