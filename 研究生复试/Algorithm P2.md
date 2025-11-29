### Part 2

#### （1）链表、队列、栈

1.单链表（静态链表模拟）

```c++
// 使用静态链表模拟，则第K个插入的节点必在K - 1下标
// 使用两个数组模拟，快
// A节点数值、next指针、head头节点、idx当前索引可分配
int node[INF], nextNode[INF], head, idx;

void initLinkList(){
    // 理解为：-1为尾指针所在索引，所以开始head -> -1
    head = -1; idx = 0;
}
// 插头结点
void addToHead(int num){
    node[idx] = num;
    nextNode[idx] = head;
    head = idx;
    idx++;
}
// 插普通，下标是inode的节点后面
void add(int inode, int num){
    node[idx] = num;
    nextNode[idx] = nextNode[inode];
    nextNode[inode] = idx;
    idx++;
}
// 删除下标inode后面的
void remove(int inode){
    // 特判删除头节点
    if(inode == -1) head = nextNode[head];
    nextNode[inode] = nextNode[nextNode[inode]];
}
```

2.双链表

```c++
// 使用头节点和尾节点,定为0与1
void initDLinkList(){
    nodeLeft[1] = 0;
    nodeRight[0] = 1;
    idx = 2;
}

// 在目标索引inode的右侧插入,注意顺序
void insertInRight(int inode, int num){
    node[idx] = num;
    nodeLeft[idx] = inode;
    nodeRight[idx] = nodeRight[inode];
    nodeLeft[nodeRight[inode]] = idx;
    nodeRight[inode] = idx;
    idx++;
}
// 删除inode,从2开始，则调用时remove(k + 1)
void remove(int inode){
    nodeRight[nodeLeft[inode]] = nodeRight[inode];
    nodeLeft[nodeRight[inode]] = nodeLeft[inode];

}
```

3.栈的应用：表达式求值

```c++
stack<int> num;
stack<char> op;
void calculate(){
    // 注意弹出顺序
    int res;
    int num2 = num.top(); num.pop();
    int num1 = num.top(); num.pop();
    char oper = op.top(); op.pop();
    if(oper == '+') res = num1 + num2;
    else if(oper == '-') res = num1 - num2;
    else if(oper == '*') res = num1 * num2;
    else res = num1 / num2;
    num.push(res);
}

int main(){
    // 哈希表，快速查询优先级，可直接按左侧符号访问
    unordered_map<char, int> pr = {{'+', 1}, {'-', 1}, 
                                   {'*', 2}, {'/', 2}};
    string str;
    cin >> str;
    for(int i = 0; i < str.size(); i++){
        auto temp = str[i];
        if(isdigit(temp)){
            int j = i, allNum = 0;
            while(j < str.size() && isdigit(str[j]))
                allNum = allNum * 10 + (str[j++] - '0');
            i = j - 1;
            num.push(allNum);
        } 
        else if (temp == '(') op.push(temp);
        else if (temp == ')'){
            while(op.top() != '(') calculate();
            op.pop();
        } 
        else {
            // 优先级里没有括号，因此特判，栈中不会存在右括号
            while(op.size() && op.top() != '(' && 
                  (pr[op.top()] >= pr[temp])) calculate();
            op.push(temp);
        }
    }
    while(op.size()) calculate();
    cout << num.top();
}
```

3.单调栈：左侧第一个比其小的

```c++
// 单调栈，左侧第一个比其小的
// 一般化，if(Ax >= Ay && x < y)x在y左侧,则x永远不会被选中
// 则每次有新的元素进入，都判定是否可以出栈栈顶元素(当前更优)
// 单调的意思是，删去无用的，剩下的必然单调
int main(){
    int N;
    stack<int> res;
    cin >> N;
    while(N--){
        int num;
        cin >> num;
        while(res.size() > 0 && res.top() >= num) res.pop();
        if(res.size() > 0) cout << res.top() << " ";
        else cout << "-1 ";
        res.push(num);
    }
}
```

4.单调队列：滑动窗口找最大最小值

```c++
// 单调队列 + 双端队列，用数组模拟，以便判断是否在窗口内
// 尾进头出,每次判断尾部,最终队列单调
int main(){
    int N, K, A[INF] = {0}, res[INF] = {0};
    cin >> N >> K;
    for(int i = 0; i < N; i++) cin >> A[i];
    int head = 0, tail = -1;
    for(int i = 0; i < N; i++){
        if(head <= tail && res[head] < i - K + 1) head++;
        while(head <= tail && A[i] <= A[res[tail]]) tail--;
        // 元素已经存在A[]中了，因此队列中一定是位序
        res[++tail] = i;
        // 让前面的元素先填满窗口
        if(i >= K - 1) cout << A[res[head]] << " ";
    }
    cout << endl;
    head = 0, tail = -1;
    for(int i = 0; i < N; i++){
        if(head <= tail && res[head] < i - K + 1) head++;
        while(head <= tail && A[i] >= A[res[tail]]) tail--;
        res[++tail] = i;
        if(i >= K - 1) cout << A[res[head]] << " ";
    }
}
```

#### （2）KMP、Trie

1.KMP：模式匹配

```c++
// KMP next[4],默认1-4,则为前三个元素的最长重复前序序列
// next[i] = j -> p[1 ~ j] = p[(j - i+ 1) ~ i]
// 例如 abab -> 2,同向扫描,则可不用移至最前到序号3即可重新匹配
// 因此先对模式串进行处理
int main(){
    char S[INF], P[INF];
    int N, M, nextidx[INF] = {0};
    // 从1开始读入
    cin >> N >> P + 1 >> M >> S + 1;
    // nextidx[1] = 0;
    for(int i = 2, j = 0; i <= N; i++){
        while(j > 0 && P[i] != P[j + 1]) j = nextidx[j];
        if(P[i] == P[j + 1]) j++;
        nextidx[i] = j;
    }
    // 每次都是S[i]与P[i + 1]作比较
    for(int i = 1, j = 0; i <= M; i++){
        while(j > 0 && S[i] != P[j + 1]) j = nextidx[j];
        if(S[i] == P[j + 1]) j++;
        if(j == N){
            cout << i - N << " ";
            j = nextidx[j];
        }
    }
}
```

2.Trie:字典树

```c++
// Trie树:高效存储查找字符串集合(字典树)
// son[]存储Trie树的逻辑关系，为当前节点的儿子节点的序号
// 理解为，二叉树每一层都有A-Z种可能，各层分开考虑，但凡终点相同就向下走
// idx保证了每个节点都有作为终点的机会
int son[INF][26] = {0}, cnt[INF] = {0}, idx = 0;
void insertTrie(string str){
    int pos = 0;
    for(int i = 0; i < str.size(); i++){
        int ti = str[i] - 'a';
        if(!son[pos][ti]) son[pos][ti] = ++idx;
        pos = son[pos][ti];
    }
    cnt[pos]++;
}
int queryTrie(string str){
    int pos = 0;
    for(int i = 0; i < str.size(); i++){
        int ti = str[i] - 'a';
        if(!son[pos][ti]) return 0;
        pos = son[pos][ti];
    }
    return cnt[pos];
}

// 使用哈希表
unordered_map<string, int> res;
if(op == 'I'){
	if(res.find(str) != res.end()) res[str]++;
	else res[str] = 1;
} else {
	if(res.find(str) != res.end()) cout << res[str] << endl;
	else cout << "0" << endl;
}
```

3.最大异或对：

```c++
// 异或，指二进制进行异或 1 3 -> 01 11 -> 10 -> 2
// 用Trie树存取0、1
int son[M][2] = {0}, idx = 0;
void insert(int num){
    int pos = 0;
    for(int i = 30; i >=0; i--){
        int temp = (num >> i) & 1;
        if(!son[pos][temp]) son[pos][temp] = ++idx;
        pos = son[pos][temp];
    }
}

int query(int num){
    int pos = 0, res = 0;
    for(int i = 30; i >=0; i--){
        int temp = (num >> i) & 1;
        if(son[pos][!temp]){
            res += 1 << i;
            pos = son[pos][!temp];
        } else {
            pos = son[pos][temp];
        }
    }
    return res;
}
```

#### （3）并查集

1.区间合并:

```c++
// 返回集合所在根节点的编号
// 路径压缩 + 查询
int findRoot(int idx){
    if(father[idx] != idx) father[idx] = findRoot(father[idx]);
    return father[idx];
}

int unionSet(int x, int y){
    father[findRoot(x)] = findRoot(y); 
}
// 计算集合大小，使用size[] = {1};维护,每次合并则更改
```

2.食物链:(1/2 X Y)->同类、X吃Y

```c++
#include<bits/stdc++.h>
using namespace std;
const int INF = 100010;

int father[INF] = {0}, dis[INF] = {0};
// 思路，维护到根节点的距离，0 - 1 - 2
// 1吃0、2吃1、0吃2，下层吃上层，再利用模运算
// 没有Union函数，同类和不同类关系合并操作不同
int find(int idx){
    if(father[idx] != idx){
        int temp = find(father[idx]);
        // 不能先更新，先更新父节点的dis未必就到根，要先递归
        // 若不保存，原来的父节点直接变成根，数据丢失了
        dis[idx] += dis[father[idx]];
        father[idx] = temp;
    }
    return father[idx];
}
int main(){
    int N, M, res = 0;
    cin >> N >> M;
    for(int i = 0; i < N; i++) father[i] = i, dis[i] = 0;
    while(M--){
        int op, x, y;
        cin >> op >> x >> y;
        if(x > N || y > N) res++;
        else {
            int rx = find(x), ry = find(y);
            if(op == 1){
                if(rx == ry && (dis[x] - dis[y]) % 3 != 0) res++;
                else if (rx != ry){                     
                    father[rx] = ry;
                    // (x + rx - y) % 3 = 0 画抽象集合树
                    dis[rx] = dis[y] - dis[x];
                }
            } else {
                if(rx == ry && (dis[x] - dis[y] - 1) % 3 != 0) res++;
                else if (rx != ry){
                    father[rx] = ry;
                    // (x + rx - 1 - y) % 3 = 0
                    dis[rx] = dis[y] - dis[x] + 1;
                }
            }
        }
    }
    cout << res;
}
```

#### （4）堆与哈希表

1.堆排序：

```c++
// 大根堆 
priority_queue<int> pq;

// 小根堆 在一亿个数字中找出最大的10000个
// 每次输出一个最小的，最终剩下的就是一万个最大的
priority_queue<int, vector<int>, greater<int>> pq1;
```

2.散列表

```c++
unordered_map<int, int> res;
```

3.字符串哈希:快速找字串

```c
const int INF = 100010, P = 131;
// 字符串哈希,原理是假设字符串是P进制数字,假设没有冲突
// 提前存储P的次方,h[]存储最左侧->目标为序的字串的哈希
// 根据进制数运算，left、right字串，将left左侧的字串抬高(×P的次方倍)再减
typedef unsigned long long ULL;
ULL h[INF] = {0}, p[INF];

ULL getHash(int left, int right){
    return h[right] - h[left - 1] * p[right - left + 1];
}

int main(){
    int N, M; cin >> N >> M; string str; cin >> str;
    str = "0" + str; p[0] = 1;   
    for(int i = 1; i <= N; i++){
        h[i] = h[i - 1] * P + str[i];	// 生成1-index的字串的hash
        p[i] = p[i - 1] * P;	// 存储次方
    }
	......
}
```

