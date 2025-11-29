#### 基础知识

1.各类文件及其作用

| .js   |       | 数据处理、函数、变量、运行逻辑，前端的后端 |
| ----- | ----- | ------------------------------------------ |
| .json |       | 配置文件                                   |
| .wxml | .html | 页面布局                                   |
| .wxss | .css  | 页面风格，决定颜色等、样式表               |

```json
- JSON
	一种独立的数据格式，一般用于配置文件；
	App.json:全局配置
  		“Pages”:页面路径；“window”:窗口的背景色、颜色等风格
  		“style”:样式版本
	project.config.json:开发工具配置
	sitemap.json:是否允许小程序被索引
```



#### Django基本流程

##### （1）基本配置

1.创建Django基本框架

```pow
django-admin startproject django_backend
```

2.创建配对的MySQL数据库

```
mysql -u root -p
// 数据库名字，不是表名
mysql> CREATE DATABASE digitalproject_v1 CHARACTER SET utf8;

- 配置Django的项目配置文件，以连接数据库
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'digitalproject_v1',
        'USER': 'root',
        'PASSWORD': '123456',
        'HOST': '127.0.0.1',
    }
}

- cd在项目文件夹下，进行Django自带数据迁移，自动生成表
python manage.py migrate
```

3.在Django框架下，创建一个实际App

```
python manage.py startapp dataprocess
```

4、将生成的App加入到Django框架settings文件中的installed_apps列表里

```
INSTALLED_APPS = [
    ...
    'dataprocess',
]
```

##### （2）项目内容

1.在生成App的models.py中编写自己的模块

```
在Django中，models.py主要用于定义数据库模型，也就是你的数据结构。
在models.py里，你通常会定义类，这些类会映射到数据库中的表。每个类的属性就是数据库的字段。
```

2.在生成App的views.py下新增自己的接口





```
npm install -g @vue/cli
vue create frontend
```















































