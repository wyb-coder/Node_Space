快速排序：分治，每次放一个到正确的位置

```python
def QuickSort(arr, left, right):
    if left > right:
        return 0
    pivot = arr[(left + right) >> 1]
    i, j = left, right
    while i <= j:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1
        if i <= j:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1
    if i < right:
        QuickSort(arr, i, right)
    if j > left:
        QuickSort(arr, left, j)
```

---

归并排序 + 逆序数：

```python
def MergeSort(arr, left, right):
    if left >= right:
        return 0
    mid = (left + right) >> 1
    MergeSort(arr, left, mid)
    MergeSort(arr, mid + 1, right)
    # 归并
    i, j, tempArr = left, mid + 1, []
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            tempArr.append(arr[i])
            i += 1
        else:
            tempArr.append(arr[j])
            j += 1
            # 全局变量
            global reverseNum
            reverseNum += mid - i + 1

    while i <= mid:
        tempArr.append(arr[i])
        i += 1
    while j <= right:
        tempArr.append(arr[j])
        j += 1
    # 切片赋值
    arr[left: right + 1] = tempArr
```

---

二分：考虑元素相同的情况，定位最左侧和最右侧

```python
def BinarySearchLeft(arr, left, right, num):
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] >= num:
            right = mid
        else:
            left = mid + 1
    return left

def BinarySearchRight(arr, left, right, num):
    while left < right:
        mid = (left + right + 1) >> 1
        if arr[mid] <= num:
            left = mid
        else:
            right = mid - 1
    return left
```

