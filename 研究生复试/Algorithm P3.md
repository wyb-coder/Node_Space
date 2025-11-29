### Part 3

#### （1）DFS、BFS

1.DFS:深度优先、排列数字

```c++
// 排列数字,假设空位,递归的用DFS搜索,回溯现场
// path保存当前路径,use检查某个数字是否被用过
int path[INF], N;
bool use[INF];
void dfs(int idx){
    if(idx == N){
        for(int i = 0; i < N; i++) cout << path[i] << " ";
        cout << endl;
    } else {
        for(int i = 1; i <= N; i++){
            if(!use[i]){
                path[idx] = i;
                use[i] = true;
                dfs(idx + 1);
                use[i] = false;
}	}	}	}
```

2.N-皇后：DFS + 剪枝

```c++
#include<bits/stdc++.h>
using namespace std;
const int INF = 100;

// chess代表棋盘,dfs每次在每一行中选可行的一列
// 这个选择的过程也就包含了剪枝
int N;
char chess[INF][INF];
bool col[INF], dg[INF], udg[INF];
void dfs(int idx){
    if(idx == N){
        // 方便的输出二维方阵
        for(int i = 0; i < N; i++) puts(chess[i]);
        cout << endl;
    }
    for(int i = 0; i < N; i++){
        // 用dg、udg标明每一个对角线、反对角线、col标记本列
        if(!col[i] && !dg[N - idx + i] && !udg[idx + i]){
            chess[idx][i] = 'Q';
            col[i] = dg[N - idx + i] = udg[idx + i] = true;
            dfs(idx + 1);
            col[i] = dg[N - idx + i] = udg[idx + i] = false;
            chess[idx][i] = '.';
        }
    }
}
// 初始化
for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++) chess[i][j] = '.';


```

3.BFS：广度优先、走迷宫

```c++
// graph存储地图，dis存储到源点距离,初始值是-1
int N, M, graph[INF][INF], dis[INF][INF];
// 利用数组枚举上下左右四个点
int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
void bfs(){
    queue<pair<int, int>> q;
    q.push({0, 0});
    //到源点距离,初始值是-1
    memset(dis, -1, sizeof(dis));
    dis[0][0] = 0;
    while(!q.empty()){
        pair<int, int> temp = q.front();
        q.pop();
        for(int i = 0; i < 4; i++){
            int x = temp.first + dx[i];
            int y = temp.second + dy[i];
            // 判断dis，不能被走过，否侧不是最短
            if((x >= 0 && x < N) && (y >= 0 && y < M) && !graph[x][y] && dis[x][y] == -1){
                q.push({x, y});
                dis[x][y] = dis[temp.first][temp.second] + 1;
            }
        }
    }
}
```

4.八数码:状态转移

```c++
// 状态A -> B 做为图的一个分支，最终正确分支只有一个
// BFS求最短分支
int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
string endState = "12345678x";
int bfs(string start){
    // dis存储每个状态的路径长度，找到一个新的路径就 + 1(新状态);
    unordered_map<string, int> dis;
    queue<string> q;
    q.push(start);
    dis[start] = 0;
    while(!q.empty()){
        string state = q.front();
        q.pop();
        // state会一直被修改，保存;
        int distance = dis[state];
        if(state == endState) return distance;
        int idx = state.find('x');
        int x = idx / 3, y = idx % 3;
        for(int i = 0; i < 4; i++){
            int tx = x + dx[i], ty = y + dy[i];
            if(tx >= 0 && tx < 3 && ty >= 0 && ty < 3){
                // 修改状态的值，一会还要恢复
                swap(state[idx], state[tx * 3 + ty]);
                // BFS一定是第一次扫到的最近，是逐步向目标的，反复扫到必然回头了，不取
                if(!dis.count(state)){
                    dis[state] = distance + 1;
                    q.push(state);
                }
                // 具体后面选哪个状态由队列决定，因此要恢复，继续以原状态试探
                swap(state[idx], state[tx * 3 + ty]);
            }
        }
    }
    return -1;
}
```

#### （2）树与图

1.树的重心

```c++
// 请你找到树的重心，并输出将重心删除后，剩余各个连通块中点数的最大值
// 使用vector存储图
vector<int> gp[INF];
bool use[INF];
int ans = INF, N;
// 本质还是暴力,返回以idx节点为根的子树的节点数
int dfs(int idx){
    // sum是子树的节点数，size是用于判断点数的最大值
    int sum = 1, size = 0;
    use[idx] = true;
    for(int i : gp[idx]){
        if(!use[i]){
            // 儿子是节点i的,以i为根的子树的节点数要求一个最大
            int tsum = dfs(i);
            size = max(size, tsum);
            // sum用于最终减去，求上面那一部分的总数
            sum += tsum;
        }
    }
    size = max(size, N - sum);
    ans = min(ans, size);
    return sum;
}
```

2.图中点的层次:最短距离

```c++
vector<int> gp[INF];
int N, M, dis[INF];
bool use[INF];
void bfs(int start){
    memset(dis, -1, sizeof(dis));
    queue<int> q;
    q.push(start);
    use[start] = true;
    dis[start] = 0;
    while(!q.empty()){
        int temp = q.front();
        q.pop();
        for(int i : gp[temp]){
            if(!use[i]){
                dis[i] = dis[temp] + 1;
                q.push(i);
                use[i] = true;
	}	}	}	}
```

3.朴素dijkstra

```c++
// 贪心，带权有向图用邻接矩阵存，dis表示距离
// use表示是否已经加入了贪心集合
// 显然使用use会多循环几次，但再优化也是N^2，不会卡
// dijkstra仅试用于权值全正
int val[MX][MX], dis[MX], N, M;
bool use[MX];
int dijkstra(){
    dis[1] = 0;
    for(int i = 0;i < N - 1; i++){
        int temp = -1;
        // 找到当前距离集合最近的点
        for(int j = 1; j <= N; j++){
            // temp = -1一定过是因为，最终一定要加一个点进入集合
            if(!use[j] && (temp == -1 || dis[temp] > dis[j])){
                    temp = j;
            }
        }
        // 更新距距离，无论是否在集合，都要更新经过temp
        // 实际上集合内的无需再更新，但此算法时间复杂度没那么重要
        for(int j = 1; j <= N; j++){
            dis[j] = min(dis[j], dis[temp] + val[temp][j]);
        }
        use[temp] = true;
    }
    if(dis[N] == INF) return -1;
    else return dis[N];
}
```

4.堆优化dijkstra

```c++
// 观察点的编号的范围，较大则默认稀疏图，仅能使用邻接表存储
// 默认所有结点先存权值，再存编号,不用考虑重复边，全存了，算法自动
int dis[INF], N, M;
bool use[INF];
vector<PII> gp[INF];
priority_queue<PII, vector<PII>, greater<PII>> heap;
int dijkstra(){
    dis[1] = 0;
    heap.push({0, 1});
    while(!heap.empty()){
        // 取出的直接就是距离最短的
        PII temp = heap.top();
        heap.pop();
        int distance = temp.first, idx = temp.second;
        if(use[idx]) continue;
        use[idx] = true;
        // 更新遍历也直接在单一邻接表中遍历即可，因为下一个选的也必定是以当前为头的
        for(int i = 0; i < gp[idx].size(); i++){
            int j = gp[idx][i].second, val = gp[idx][i].first;
            if(dis[j] > dis[idx] + val){
                dis[j] = dis[idx] + val;
                heap.push({dis[j], j});
            }
        }
    }
    if(dis[N] == MX) return -1;
    else return dis[N];
}
```

