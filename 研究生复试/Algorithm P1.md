### Part 1

#### （1）排序 + 查找

1.快排 + 第K个数

```c++
#include <bits/stdc++.h>
using namespace std;

int Quick_Select(int arr[], int left, int right, int index) {
    if (left == right) return arr[left];
    int i = left - 1, j = right + 1;
    int pivot = arr[(left + right) / 2];
    while (i < j) {
        while (arr[++i] < pivot);
        while (arr[--j] > pivot);
        if (i < j) swap(arr[i], arr[j]);

    }
    int temp = j - left + 1;
    if(index <= temp) return Quick_Select(arr, left, j, index);
    return Quick_Select(arr, j + 1, right, index - temp);
    //Quick_Sort
    //if (j > left) Quick_Sort(arr, left, j);
    //if (i < right) Quick_Sort(arr, i, right);
}    
  	int res = Quick_Select(arr, 0, N - 1, index);
```

2.归并 + 逆序对

```c++
# include <bits/stdc++.h>
using namespace std;

#define INF 100000 
int tempArr[INF]; long long int res = 0;

void mergeSort(int arr[], int left, int right){
    if(left >= right) return;
    int mid = (left + right) >> 1;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    int index = 0, i = left, j = mid + 1;
    while(i <= mid && j <= right){
        if(arr[i] <= arr[j]) tempArr[index++] = arr[i++];
        else {
            tempArr[index++] = arr[j++];
            // res += mid - i + 1;
        } // i、j都递增，则此i后全部为逆序对;以每个j判断所有i
    }
    while(i <= mid) tempArr[index++] = arr[i++];
    while(j <= right) tempArr[index++] = arr[j++];
    for(int i = left, j = 0;i <= right;i++, j++) arr[i] = tempArr[j];
}
    mergeSort(arr, 0, N-1);

```

3.二分 + 数的范围 + 三次方根；`特判查找失败`

```c++
// 二分最左侧的待定值	左0mid大
int bianrySearchlLeft(int arr[], int left, int right, int num){
    while(left < right){
        int mid = (left + right) >> 1;
        if(arr[mid] >= num) right = mid;
        else left = mid + 1;
    }
    return left;
}
// 二分最右侧的待定值	右1mid小
int bianrySearchRight(int arr[], int left, int right, int num){
    while(left < right){
        int mid = (left + right + 1) >> 1;
        if(arr[mid] <= num) left = mid;
        else right = mid - 1;
    }
    return left;
}
// 给定区间内求某数的三次方根
int main(){
    double num; cin >> num;
    double left = -10000, right = 10000;
    while(abs(left - right) > 1e-8){
        double mid = 1.0 * (left + right) / 2 ;
        if(mid * mid * mid >= num) right = mid;
        else left = mid;
    }
    printf("%6f",left);
}
```

#### （2）高精度

1.高精度加

```c++
// 逆向存储、逆向思维，便于最高位进位、低位对齐
vector<int> Add(vector<int> &A, vector<int> &B){
    int temp = 0, fp = 0;
    vector<int> res;
    for(int i = 0; i < A.size(); i++){
        temp = A[i] + fp;
        if(i < B.size()) temp += B[i];
        if(temp >= 10){
            res.push_back(temp % 10);
            fp = 1;
        } else {
            res.push_back(temp);
            fp = 0;
        }
    }
    // 最高位进位
    if(fp != 0) res.push_back(fp);
    while(res.size() > 1 && res.back() == 0) res.pop_back();
    return res;
}
```

2。高精度减

```c++
// 直接使用字符串比较会有奇怪的bug
// 一定是计算大的减小的，避免最高位借位麻烦
bool Cmp(vector<int> &A, vector<int> &B){
    if(A.size() != B.size()) return A.size() > B.size();
    for(int i = A.size() - 1;i >= 0; i--){
        if(A[i] != B[i]) return A[i] > B[i];
    }
    return true;
}

vector<int> Sub(vector<int> &A, vector<int> &B){
    vector<int> res;
    int temp = 0, fp = 0;
    for(int i = 0;i < A.size();i++){
        temp = A[i] + fp;
        // 无需再填满，默认让进入的长的减小的
        if(i < B.size()) temp -= B[i];
        if(temp < 0){
            res.push_back(temp + 10);
            fp = -1;
        } else {
            res.push_back(temp);
            fp = 0;
        }
    }
    while(res.size() > 1 && res.back() == 0) res.pop_back();
    return res;
}
// vector<int> A, B;
// for(int i = num1.size() - 1;i >= 0;i--) A.push_back(num1[i] - '0');
// for(int i = num2.size() - 1;i >= 0;i--) B.push_back(num2[i] - '0');
```

3.高精度乘、除

```c++
// B较小、每次取A的一位×B，再算进位，默认全正
vector<int> Mul(vector<int> A, int B){
    vector<int> res;
    int temp = 0;
    for(int i = 0; i < A.size() || temp > 0; i++){
        if(i < A.size()) temp += A[i] * B;
        res.push_back(temp % 10);
        temp /= 10;
    }
    while(res.size() > 1 && res.back() == 0) res.pop_back();
    return res;
}

// A/B 余R、A是逆置的
vector<int> Div(vector<int> &A, int B, int &R){
    vector<int> res;
    R = 0;
    for(int i = A.size() - 1; i >= 0; i--){
        R = R * 10 + A[i];
        res.push_back(R / B);
        R %= B;
    }
    reverse(res.begin(), res.end());
    while(res.size() > 1 && res.back() == 0) res.pop_back();
    return res;
}
```

#### （3）前缀和、差分

1. 前缀和、子矩阵的和

```c++
// 前缀和，使用逆向思维,求哪一区间就用不同前缀和减 
// 包含左右区间，因此左边要向左一位，则添加0位为0
	for(int i = 1; i <= N; i++){
        cin >> PSum[i];
        PSum[i] += PSum[i - 1];
    }
    for(int i = 0; i < M; i++){
        int t1, t2;
        cin >> t1 >> t2;
        res.push_back(PSum[t2] - PSum[t1 - 1]);
    }

// 子矩阵的和，前缀和的建议推广，要有前缀和的思维，看能否使用推广
int main(){
    int N, M, Q, Mix[INF][INF] = {0}, PSum[INF][INF] = {0}, temp = 0;
    vector<int> res;
    cin >> N >> M >> Q;
    for(int i = 1; i <= N; i++){
        for(int j = 1; j <= M; j++){
            cin >> PSum[i][j];
            PSum[i][j] += 
                PSum[i][j - 1] + PSum[i - 1][j] - PSum[i - 1][j - 1];
        }
    } 
    while(Q--){
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        res.push_back(PSum[x2][y2] - PSum[x2][y1 - 1] - 
                      PSum[x1 - 1][y2] + PSum[x1 - 1][y1 - 1]);
    }
    for(int i = 0; i < res.size(); i++) cout << res[i] << endl;
}
```

2.差分 + 差分矩阵

```c++
// 差分、前缀和的逆，A -> B的前缀和，B为A的差分 B[n] = A[n] - A[n - 1]
// B的前缀和是A， B[r] + C = A[r, N] + C
int main(){
    int N, M, A[INF], DSum[INF] = {0}, res[INF] = {0};
    cin >> N >> M;
    for(int i = 1; i <= N; i++) cin >> A[i];
    for(int i = 1; i <= N; i++) DSum[i] = A[i] - A[i - 1];
    while(M--){
        int left, right, num;
        cin >> left >> right >> num;
        DSum[left] += num;
        DSum[right + 1] -= num;
    }
    for(int i = 1; i <= N; i++) res[i] = res[i - 1] + DSum [i];
    for(int i = 1; i <= N; i++) cout << res[i] << " ";
}


// 可以不用想构造，0矩阵的差分矩阵也是0矩阵
// 矩阵A与其差分矩阵B是一一对应的
// 每次对A矩阵的每一个点 + C, 使用差分矩阵的算法对B进行插入，则生成差分矩阵
// 差分矩阵是对右下角包括本身全部 + C
void Insert(int B[][INF], int x1, int y1, int x2, int y2, int num){
    B[x1][y1] += num; B[x2 + 1][y2 + 1] += num;
    B[x1][y2 + 1] -= num; B[x2 + 1][y1] -= num; 
}

int main(){
    int A[INF][INF] = {0}, B[INF][INF] = {0}, N, M, Q;
    cin >> N >> M >> Q;
    for(int i = 1; i <= N; i++){
        for(int j = 1; j <= M; j++){
            cin >> A[i][j];
            Insert(B, i, j, i, j, A[i][j]);
        }
    }
    while(Q--){
        int x1, y1, x2, y2, num;
        cin >> x1 >> y1 >> x2 >> y2 >> num;
        Insert(B, x1, y1, x2, y2, num);
    }
    ......
}
```

#### （4）双指针

1.最长连续不重复子序列

```c++
// i, j双指针，一般为就j落后于i，向后扫描而不是向前
// 最长不重复序列中，j能切仅能向右移动
// 双指针是动态的，每次检查移动的i是否符合条件即可
// 记得初始化
int main(){
    int N, A[INF], Tub[INF] = {0}, res = 0;
    cin >> N;
    for(int i = 0; i < N; i++) cin >> A[i];
    for(int i = 0, j = 0; i < N; i++){
        Tub[A[i]]++;
        while(j < i && Tub[A[i]] > 1) Tub[A[j]]--, j++;
        res = max(res, i - j + 1);
    }
    cout << res;
}
```

2.数组元素的目标和

```c++
// 双指针就找单调性、固定一个，找另一个的单调性
// 固定i,对j而言，A[i] + B[j] >= Num,此时i向后移动，则j向左
// 则，只用寻找左侧
int main(){
    int N, M, X, A[INF], B[INF];
    cin >> N >> M >> X;
    for(int i = 0; i < N; i++) cin >> A[i];
    for(int i = 0; i < M; i++) cin >> B[i];
    for(int i = 0, j = M - 1; i < N; i++){
        while(j >= 0 && A[i] + B[j] > X) j--;
        if(A[i] + B[j] == X){
            cout << i << " " << j;
            break;
        }
    }
}
```

3.子序列:不要求连续

```c++
// 如果子序列存在，双指针一定能找到
int main(){
    long long int N, M, A[INF], B[INF], K = 0;
    cin >> N >> M;
    for(int i = 0; i < N; i++) cin >> A[i];
    for(int i = 0; i < M; i++) cin >> B[i];
    for(int i = 0; i < M; i++) if(A[K] == B[i]) K++;
    if(K >= N) cout << "Yes";
    else cout << "No";
}
```

#### （5）位运算、离散化、区间合并

1.位运算：第K位二进制数、快速得出二进制、lowbit

```c++
// (1) 第K位(0.1.2...)二进制数:
// 右移K位将其放到最右侧再查看最右侧(Num & 1)
// (2) 这也是快速求二进制表示的方法
// for(int i = N; i >= 0; i--) (Num >> i) & 1;

// (3) lowbit 返回Num的最后一位1,树状数组基本操作
// 1010->10   11000->1000 返回二进制数字
// 原理为，Num & -Num(其补码)
int lowbit(int Num){ return Num & (- Num); }
int main(){
    long long int N, A[INF];
    cin >> N;
    for(int i = 0; i < N; i++) cin >> A[i];
    for(int i = 0; i < N; i++){
        int count = 0;
        while(A[i] != 0) A[i] -= lowbit(A[i]), count++;
        cout << count << " ";
    }
}
```

2.离散化：区间和:数轴加减

```c++
const int INF = 300010;

// 离散:将离散的数字(数轴上，但个数较少)映射到连续的数组中
// (1)去重    (2)用二分找映射关系

// 离散化 + 前缀和; 正负被二分映射为了正整数
// 传入参数过多会显著降低速度，能开全局就全局
vector<int> All;
int findIndexByBS(int num){
    int left = 0, right = All.size() - 1, mid;
    while(left < right){
        mid = (left + right) >> 1;
        if(All[mid] >= num) right = mid;
        else left = mid + 1;
    }
    return right + 1;
    // 为了前缀和,离散到1...N
}  

int main(){
    vector<pair<int, int>> Add, Query;
    int N, M, A[INF] = {0}, res[INF] = {0};
    cin >> N >> M;
    for(int i = 0; i < N; i++){
        int X, C;
        cin >> X >> C;
        All.push_back(X);
        Add.push_back({X, C});
    }
    for(int i = 0; i < M; i++){
        int left, right;
        cin >> left >> right;
        All.push_back(left);
        All.push_back(right);
        // 将待查询左右也放入全部中一起去重排序
        Query.push_back({left, right});
    }
    // 去重
    sort(All.begin(), All.end());
    All.erase(unique(All.begin(), All.end()), All.end());
    // 处理插入
    for(int i = 0; i < Add.size(); i++){
        int temp = findIndexByBS(Add[i].first);
        A[temp] += Add[i].second;
    }
    // 计算前缀和,数组上限不会超过All中元素个数
    for(int i = 1; i <= All.size(); i++) res[i] = res[i - 1] + A[i];
    // 处理询问
    for(int i = 0; i < Query.size(); i++){
        int left = Query[i].first, right = Query[i].second;
        left = findIndexByBS(left);
        right = findIndexByBS(right);
        cout << res[right] - res[left - 1] << endl;
    }
    return 0;
}
```

3.区间合并

```c++
void InterMerge(vector<pair<int, int>> &A){
    sort(A.begin(), A.end());
    int left = -1e9 - 1, right = -1e9 - 1;
    for(int i = 0; i < A.size(); i++){
        if(right < A[i].first){
            if(left != (-1e9 - 1)) res.push_back({left, right});
            left = A[i].first;
            right = A[i].second;
        }
        else right = max(right, A[i].second);
    }
    if(left != (-1e9 - 1)) res.push_back({left, right});
}
```























