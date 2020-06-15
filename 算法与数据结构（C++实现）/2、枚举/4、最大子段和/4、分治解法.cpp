/*

样例
7
2 -4 3 -1 2 -4 3

分治解法
假定a[1]-a[n]的序列对应的区间[l...r]，其中间位置为mid，其最大和的子序列为[i...j]。
那么显然，最大连续子序列的位置只有三种可能：
①完全处于序列的左半：l<=i<=j<=mid
②跨越序列中间：i<=mid<=j<=r
③完全处于序列的右半：mid<i<=j<=r


只需要分别求出三种情况下的值，取他们最大的即可。
其中，很容易求出第二种情况，第二种情况也就是包含mid的子序列，
也就是[i...mid...j]，而求[i...mid...j]的最大值，
即求出区间[i..mid]的最大值maxx1与区间[mid..j]的最大值maxx2，将其合并即可。
合并之后就变成了[i...mid mid...j]，mid出现了两次，要减掉一次
所以[i...mid...j]的最大值就是maxx1+maxx2-mid

复杂度O(n)
如何处理第一种和第三种情况呢？
也不难发现，
第一种情况，其实就是求区间[l..mid]中的最大值，
第三种情况就是求区间[mid+1..r]中的最大值。那么，只需递归求出即可。
显然，该解法的复杂度为O(nlogn)通过此题是没问题的。


算法时间复杂度
O(nlogn)：二分是logn，处理第二种情况是n，所以合起来就是O(nlogn)


如何求区间[i..mid]的最大值与区间[mid..j]的最大值，
换句话说，也就是如何求以mid为尾的子序列的最大值 和 以mid为头的子序列的最大值
先说以mid为头的子序列的最大和
也就是[mid]，[mid...mid+1]，[mid...mid+2]......[mid...mid+j]这些序列里面的最大值
int maxx2=-0x7fffffff;
int sum2=0;
for(int k=mid;k<=j;k++){
    sum2+=a[k];
    maxx2=max(sum2,maxx2);
}

求以mid为尾的子序列的最大和
int maxx1=-0x7fffffff;
int sum1=0;
for(int k=mid;k>=i;k--){
    sum1+=a[k];
    maxx1=max(sum1,maxx1);
}

maxx1+maxx2-a[mid]


递归做分治：
a、递归的终止条件：
因为我们的递归是为了求l到r序列的子序列的最大值，
所以当区间只有一个元素时，就是终止条件，那个元素就是子序列的最大值
b、递归的递推表达式：比较方式1、2、3的最大值。第2种跨越mid值的需要我们去计算，1,3种情况又转化成了子问题
c、递归的返回值：子序列的最大和


算法步骤：
1、计算第二种跨越mid情况的序列的最大和
2、比较方式1、2、3的最大值



样例：
4
-1 3 -1 -2
结果是3 

mid=(1+4)/2 2
①完全处于序列的左半：l...mid：-1 3  对应的是3
②跨越序列中间：3+3-3=3
③完全处于序列的右半：mid+1...r：-1 -2 对应的结果是-1

-1 3
mid=1
①完全处于序列的左半：l...mid：-1
②跨越序列中间：-1+2-(-1)=2
③完全处于序列的右半：mid+1...r：3


为什么这道题目可以用分治来做
因为因为连续子序列只能是如下三种情况的一种：
取这三种情况里面的最大值，即可得到本题的解。
①完全处于序列的左半：l<=i<=j<=mid
②跨越序列中间：i<=mid<=j<=r
③完全处于序列的右半：mid<=i<=j<=r

*/
#include <iostream>
#include <algorithm>
using namespace std;
int a[200005];
//分治（二分）求最大连续子序列的和
int find(int l,int r){
    if(l==r) return a[l];
    int mid=(l+r)/2;
    //1、计算第二种跨越mid情况的序列的最大和
    //a、求以mid为尾的子序列的最大和
    int maxx1=-0x7fffffff;
    int sum1=0;
    for(int k=mid;k>=l;k--){
        sum1+=a[k];
        maxx1=max(sum1,maxx1);
    }

    //b、求以mid为头的子序列的最大和
    int maxx2=-0x7fffffff;
    int sum2=0;
    for(int k=mid;k<=r;k++){
        sum2+=a[k];
        maxx2=max(sum2,maxx2);
    }

    //2、比较方式1、2、3的最大值
    return max(max(find(l,mid),find(mid+1,r)),maxx1+maxx2-a[mid]);
}

int main(){
    int n;
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>a[i];
    }
    cout<<find(1,n)<<endl;
    return 0;
}
