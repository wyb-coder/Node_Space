#### C语言程序设计

1.常用库函数、额外注意：

```c
memset(res, 0, sizeof(res));
isdigit(A); isalpha(A);
floor(A); ceil(A)
abs(A); fabs(A);

scanf("%s", &str) != gets(str)	// scanf遇到空格就不读取了，一次读取全部则使用gets
```

2.qsort:`sizeof中为其元素的长度，则应为类型名如sizeof(int)`

```c
// cmp必须自己实现，形参声明无需更改
struct Students{int score; };
int cmp(const void *A, const void *B){
    // A > B -> 1 -> 增序
    return *(int*)A - *(int*)B;
    // 结构体排序
    // return *(Students*)A.score - *(Students*)B.score;
}
int main(){
	int res[5] = {5, 4, 3, 2, 1};
	qsort(res, 5, sizeof(res), cmp);
}

// QuickSort记忆
int QuickSort(int arr[], int left, int right){
    if(left == right) return arr[left];
    int i = left, j = right;
    int pivot = arr[(left + right) / 2];
    while(i <= j){
        while(arr[i] < pivot) i++;
        while(arr[j] > pivot) j--;
        if(i <= j){
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++; j--;
        }
    }
    if(j > left) QuickSort(arr, left, j);
    if(i < right) QuickSort(arr, i , right);
}
```

3.最大公约数与最小公倍数：

```c
// 两个数的乘积等于其最小公倍数和最大公约数的乘积
typedef long long int LL;
LL gcd(LL A, LL B){
    if(B != 0) return gcd(B, A % B);
    else return A;
}

LL lcm(LL A, LL B){
    return A * B / gcd(A, B);
}
```

4.闰年 + 第几天

```c
int isLeapYear(int year){
    if(year % 400 == 0) return 1;
    if(year % 4 == 0 && year % 100 != 0) return 1;
    return 0;
}
// 1 3 5 7 8 10 12月有31天
int days[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
int main(){
	int year, month, day, sum = 0;
    scanf("%d %d %d", &year, &month, &day);
    days[0] = day;
    if(isLeapYear(year)) days[2] = 29;
    else days[2] = 28;
    for(int i = 0; i < month; i++) sum += days[i];
    printf("%d", sum);
}
```

5.链表基础

```c
typedef struct Student{
    int id;
    int score;
    struct Student* next;
}LinkList;
int main(){
    LinkList *head, *node, *idx, *temp;
    // 声明head时，和声明其他类型变量相同，其指针可可能指向其地址空间内遗留内容
    // malloc实际上就是初始化，开辟一片空间给指针初始指向
    head = (LinkList*)malloc(sizeof(LinkList));
    idx = (LinkList*)malloc(sizeof(LinkList));
    // 链表遍历
	idx = head->next;
    while(idx != NULL){
        printf("%d %d\n", idx->id, idx->score);
        idx = idx->next;
    }
}
```

6.宏定义：

```c
#define INF 1010
#define Swap(A, B) {int temp = A; A = B; B = temp;}
#define Max(A, B) {(A) > (B) ? (A) : (B)}
```

7.位运算之循环移位

```c
// 循环右移函数
unsigned int circularRightShift(unsigned int num, unsigned int shift) {
    // 整数的位数，sizeof返回的是字节数 1Byte == 8 bit
    unsigned int numBits = sizeof(unsigned int) * 8;
    shift = shift % numBits; // 确保移位不超过位数
    return (num >> shift) | (num << (numBits - shift));
    // return (num << shift) | (num >> (numBits - shift)); 循环左移
}
```

8.利用负数求补码，保存符号位

```c
int main() {
    int str[16], sum = 0, p = 0;
    for(int i = 0; i < 16; i++){
        char temp;
        scanf("%c", &temp);
        str[i] = temp - '0';
    }
    int flag = str[0];
    for(int i = 15; i >= 0; i--){
        sum += pow(2, p++) * str[i];
    }
    sum = -sum;
    for(int i = 15; i >= 0; i--){
        str[i] = sum & 1;
        sum = sum >> 1;
    }
    str[0] = flag;
    for(int i = 0; i < 16; i++) printf("%d", str[i]);
}
```

9.汉诺塔

```c
// from to by
void hanoi(int N, char A, char C, char B){
    if(N == 1){
        printf("Move disk 1 from %c to %c\n", A, C);
        return;
    }
    hanoi(N - 1, A, B, C);
    // 将第N个盘子，由from -> to
    printf("Move disk %d from %c to %c\n", N, A, C);
    hanoi(N - 1, B, C, A);
}
```

10.函数中的二维数组

```c
void max(int arr[][3]){}
// 必须声明列的长度
// 实际上只保存着首指针，其他的信息一概不知
// C默认行优先存储，必须知道什么时候转行 == 知道一行多少个 == 列的长度
```

11.随机数

```c
srand((unsigned)time(NULL));
int num = rand() % 10;
```

12.变量命名：只能使用字母、数字、下划线；其中开头不能用数字

13.文件操作：

```c
// 文件打开、循环读入
int main(){
    char temp, buffer[256];
    FILE *fp1, *fp2;
    // 打开文件，失败返回NULL，而EOF用于判断文件结尾
    if(NULL == (fp1 = fopen("test.txt", "r+"))){
        printf("Error in open file");
        return 0;
    }

    while(EOF != (temp = fgetc(fp1))){
        printf("%c", temp);
    }
    printf("\n--------------\n");
    // 重定位到文件头 rewind(fp1) 等价于 fseek(fp1, 0, 0);;    
    // fseek(fp1, X, Y); Y有三个取值，是内置宏 
    // SEEK_SET文件头；SEEK_CUR当前位置；SEEK_END文件结尾
    fseek(fp1, 0, SEEK_SET);
    // EOF 是文件结束符，一个符号。而fgets读入一行，失败是NULL
    while(NULL != (fgets(buffer, sizeof(buffer), fp1)))
        printf("%s", buffer);
    
    rewind(fp1);
    // 文件向buffer输入。实际是读入文件
    fscanf(fp1, "%s", &buffer); printf("%s", buffer);
}

// 文件的写入
	// 定位到末尾 == 追加
	// 特别的，这里的偏移量以字节为单位
    fseek(fp1, 0, SEEK_END);
    fprintf(fp1, "%s", "aaaaa");

	// size_t与unsigned几乎完全相同，也是整数类型
	// 唯一不同在于，64位机器中size_t有八个字节
	size_t length;
    length = fread(buffer, sizeof(char), sizeof(buffer) - 1, fp1);
    buffer[length] = '\0';
    printf("%s", buffer);
```

14.贪心 + 递归：最多喝多少瓶酒

```c
// 本轮一共numBottles个酒，numExchange能换一瓶酒，返回最多喝多少瓶
// 无需在意numBottles具体装没装水，应该是没装水的瓶子数，因为但凡空瓶子也是满的喝了后变空的
int numWaterBottles(int numBottles, int numExchange) {
	if(numBottles < numExchange) return numBottles;
	else if(numBottles == numExchange) return numBottles + 1;
	else{
        // 第一次能换多少瓶酒
		int first = numBottles / numExchange;
        // 换完剩下多少
		int left = numBottles % numExchange;
        // 把换的喝完，剩下的够不够继续换
		if(first + left < numExchange) return first + numBottles;
		else if(first + left == numExchange) return first + numBottles + 1;
        // 够继续换，返回的是最多的瓶子，一共numBottles，left是没换的部分，直接加入下次换酒
        // 因此相当与被拿去下次计算了，要排除
		else return numBottles - left + numWaterBottles(first + left, numExchange);
	}
}
```

15.专业问题

```
-                                                              
```





































































































