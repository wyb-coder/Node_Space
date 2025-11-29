### Part4

#### （1）质数与约数

1.试除法判断质数

```c++
// 质数: 大于1,且能且仅能被1与其本身整除
// 定理 if (d % N == 0) => (d / N % N == 0)
// 例: 3、4与2、6都是12的约数
// 则所有约数都是成对出现的，因此枚举较小的那个。因此d <= sqrt(N)
// sqrt较慢。d <= sqrt(N) == d^2 <= N == d <= N / d
bool isPrime(int num){
    if(num < 2) return false;
    for(int i = 2; i <= num / i; i++){
        if(num % i == 0) return false;
    }
    return true;
}
```

2.分解质因数

```c++
// 分解质因数，6 = (2^1) * (3^1)
// 8 = (2^3)
void div(int num){
    for(int i = 2; i <= num / i; i++){
        if(num % i == 0){
            int sum = 0;
            while(num % i == 0) num /= i, sum++;
            cout << i << " " << sum << endl;
        }
    }
    if(num > 1) cout << num  << " " << 1 << endl;
    cout << endl;
}
```

3.质数筛

```c++
// 求1 ~ N中质数的个数，基本原理，将i的所有倍数全部删除
// primes存质数，is存是否是质数，cnt维护primes上限
int primes[INF], cnt;
bool is[INF];
// 朴素质数筛:将i的所有倍数全部删除
void getPrimes1(int N){
    for(int i = 2; i <= N; i++){
        if(!is[i]){
            primes[cnt++] = i;
            for(int j = i * 2; j <= N; j += i) is[j] = true;
        }
    }
}
// 线性筛法
void getPrimes2(int N){
    for(int i = 2; i <= N; i++){
        if(!is[i]) primes[cnt++] = i;
        for(int j = 0; primes[j] <= N / i; j++){
            is[i * primes[j]] = true;
            if(i % primes[j] == 0) break;
        }
    }
}
```

4.所有约数：`if (d % N == 0) => (d / N % N == 0)`

```c++
// 1~N的所有约数
vector<int> res;
void getDivisors(int N){
    for(int i = 1; i <= N / i; i++){
        if(N % i == 0){
            res.push_back(i);
            if(N / i != i) res.push_back(N / i);
        }
    }
    sort(res.begin(), res.end());
}
```

5.约数之和：很多个数乘积的约数之和

```c++
// 约数个数、约数之和
// 分解质因数:N = (P1 ^ X1) * ... * (Pn ^ Xn)
// 用分解质因数后的公式相乘
// 乘积的约数个数 = (X1 + 1) * ... * (Xn + 1)
// 约数之和 = (P1 ^ 0 + ... + P1 ^ X1) * ... * (Pn ^ 0 + ... + Pn ^ Xn)
unordered_map<int, int> primes;
const int mod = 1e9 + 7;
int main(){
    int N;
    cin >> N;
    while(N--){
        int num;
        cin >> num;
        for(int i = 2; i <= num / i; i++){
            while(num % i == 0){
                num /= i;
                primes[i]++;
            }
        }
        if(num > 1) primes[num]++;
    }
    long long int res = 1;
    for(auto prime : primes) res = res * (prime.second + 1) % mod;
    cout << res;
}
```

6.约数之和

```c++
unordered_map<int, int> primes;
int main(){
    int N;
    cin >> N;
    while(N--){
        int num;
        cin >> num;
        for(int i = 2; i <= num / i; i++){
            while(num % i == 0){
                num /= i;
                primes[i]++;
            }
        }
        if(num > 1) primes[num]++;
    }
    long long int res = 1, temp = 1;;
    for(auto prime : primes){
        int p = prime.first, t = prime.second;
        while(t--) temp = (temp * p + 1) % mod;
        res = res * temp % mod;
        temp = 1;
    }
    cout << res;
}
```

7.最大公约数

```c++
// 最大公约数 (A, B) == (B, A mod B)
int gcd(int A, int B){
    if(B != 0) return gcd(B, A % B);
    else return A;
}
```

#### （2）

1.快速幂：

```c++
// 快速幂。快速求(A^K mod P),预处理 A^(2^0) ... A^(2^logK)
// 其中 D1 = A^(2^0) D2 = A^(2^1) D3 = A^(2^2) 则Dn = D(n-1)^2
// 将A^K 拆解成 A^(2^X1) ... A^(2^Xn)
// 则将K 拆解为 (2^X1) ... ^(2^Xn) 这显然可以用二进制表示
// 则logK直接表示为K的二进制位数
typedef long long int LL;
LL quickPow(LL A, LL B, LL P){
    LL res = 1;
    while(B){
        // A的次方隐藏在了A的变化中
        if(B & 1) res = res * A % P;
        A = A * A % P;
        B = B >> 1;
    }
    return res;
}
```

2.汉诺塔：

```c
// A -> C by B
// from -> to by temp
// 初始输入 A,C,B
// 分为两部分，最大的盘子和剩下n - 1个
// (1)将A的n - 1个由C放到B
// (2)将A剩下的放到C
// (3)将B的n - 1个由A放到C
// 特例:有且仅有一个盘子，直接将A->C
void hanoi(int N, char A, char C, char B){
    if(N == 1){
        printf("Move disk 1 from %c to %c\n", A, C);
        return;
    }
    hanoi(N - 1, A, B, C);
    printf("Move disk %d from %c to %c\n", N, A, B);
    hanoi(N - 1, B, C, A);
}
```

































