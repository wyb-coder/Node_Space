#### 一、HTML

##### （1）HTML基础语法

1.HTML（超文本标记语言）:

```html
超文本 == 链接
标记 == 标签 => <h1> </h1>
```

2.标签语法：

```html
- 单标签、双标签
<!--注释-->
<strong>Hello World!</strong> 加粗
<br>换行 <hr>水平线
```

3.==! + Tab==自动生成基本骨架

4.标签关系：父子关系、并列关系

5.==Ctrl + /== 快速注释

6.标签

```html
- 标题标签h1~h6
<h1></h1> 一般只用一次
<h2></h2> 无限次

- <p></p> 段落标签
独占一行、段落之间有间隙

- <br> 换行
- <hr> 水平线

- 文本格式化标签
<strong></strong> 加粗
<em></em> 倾斜
<ins></ins> 下划线
<del></del> 删除线

- 图像标签
<img src="./preview.jpg" alt="替换文本" title="提示文本(鼠标)" width="" weight=""> 
<!--宽高一般通过CSS规定-->

- 音频标签
<audio src="#" controls></audio>
<!--属性名和属性值完全一样，可以简写为一个单词-->

- 视频标签
<video src="#" controls loop muted autoplay></video>
<!--一般默认静音下自动播放-->
```

7.路径：

```
- 相对路径
../
.表示当前所在文件夹
/表示进入该文件夹

- 绝对路径，应用于友情链接(超链接)
```

8.超链接：

```html
<a href="./Test.html" target="_blank">Test</a> 新窗口打开
<a href="#">空链接</a>
```

##### （2）列表、表格和表单

1.列表：布局内容排列整齐的区域

```html
- 无序列表
<ul>
    <li></li>
    <li></li>
    <li></li>
</ul>

- 有序列表
<ol>
    <li></li>
</ol>

- 定义列表
<dl>
    <dt></dt>
    <dd></dd>
</dl>
```

2.表格

```html
<tr></tr> 标记一行
<th></th> 表头
<td></td> 普通
<table border="1">
    <tr>
        <th>111</th>
        <th>222</th>
    </tr>
    <tr>
        <td>333</td>
        <td>444</td>
    </tr>
</table>

- 跨行合并
<th rowspan="2">111</th>

- 跨列合并
<th colspan="2">111</th>
```

3.表单

```html
<input type="#">
text 文本
password 密码框
radio 单选框
checkbox 多选框
file 上传文件

- 下拉菜单
<select>
    <option value="beijing">北京</option>
    <option value="shanghai" selected>上海</option>
</select>

- 文本域 多行输入文本
<textarea></textarea>

- lable说明文本
<lable></lable>

- 按钮
<button type=""></button>
submit
reset
button

- 布局标签：划分网页区域
<div></div> 独占一整行
<span></span> 小盒子
包裹住一片区域

- 占位符

```



#### 二、CSS

##### （1）基础选择器、文字控制属性

1.CSS:层叠样式表

```html
- 内部样式表
<title>Document</title>
<!-- CSS写法1:直接写入HTML -->
<style>
    /* 规定谁的风格就写谁 */
    h1 {
        color: aqua;
    }
</style>
```

```css
- 外部样式表 .css 使用<link>引入
<link rel="stylesheet" href="./CSS1.css">

- 行内样式 配合JS使用

```

2.选择器：查找标签，设置样式

```css
- 标签选择器：标签名，所有同类标签
    
- 类选择器：差异化设置标签的显示效果
定义：.类名
使用：class = "类名"
.red {
    color: red;
}
<div class="red size"><h1>Class Select</h1></div>

- ID选择器:同一个ID名在一个页面中只能使用一次
#red {
    color: red;
}   

- 通配符选择器：所有的标签
* {
    color: red;
}
作用优先级更高
```

3.文字效果属性

```css
font-size
font-weight 粗细
font-style
line-height
font-family
font
text-indent 文本缩进
text-align 对齐方式
text-decoration 修饰线

```

##### （2）复合选择器、CSS特性、背景属性、显示模式

1.复合选择器

```css
- 后代选择器：选择出全部的后代
div span {
    color: red;
}

- 子带选择器
div > span {}

- 并集选择器
div, p, span {}

- 交集选择器
p.red {} 标签选择器要写在最前面
```

2.伪类选择器：元素状态

```css
a.hover {} 鼠标需求悬停状态

:link	访问前
:visited 访问后
:hover
:visited
```

3.CSS的性质

```css
- 继承性：子集默认继承父级的文字控制属性
全局同一的，设置为<body>的属性

- 层叠性：同样的后面生效，不同的都生效

- 优先级：选中标签的范围越大，优先级越低
!important 指定最高优先级
```

4.Emmet写法：

```html
div.box
p.box
div+p
div>p
span*3
w500+h200+bgc
```

5.背景属性

```css
background-color
background-image
background-repeat 背景平铺方式
background-position
background-size
background-attachment
background
```

6.显示模式

```css
- 块级元素<div>
- 行内元素
- 行内块元素
```

7.特殊选择器

```css
E:first-child {}
E:last-child {}
E:nth-child(N) {}
N 2N+1 ...
```

8.伪元素选择器：创建虚拟标签，用于装饰

```css
E::before
E::after
```

9.PxCook

10.盒子模型



































