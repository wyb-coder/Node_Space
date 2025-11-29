#coding:utf-8
__author__ = "ila"
from django.db import models

from .model import BaseModel

from datetime import datetime



class yonghu(BaseModel):
    __doc__ = u'''yonghu'''
    __tablename__ = 'yonghu'

    __loginUser__='yonghuming'


    __authTables__={}
    __authPeople__='是'#用户表，表属性loginUserColumn对应的值就是用户名字段，mima就是密码字段
    __loginUserColumn__='yonghuming'#用户表，表属性loginUserColumn对应的值就是用户名字段，mima就是密码字段
    __sfsh__='否'#表sfsh(是否审核，”是”或”否”)字段和sfhf(审核回复)字段，后台列表(page)的操作中要多一个”审核”按钮，点击”审核”弹出一个页面，包含”是否审核”和”审核回复”，点击确定调用update接口，修改sfsh和sfhf两个字段。
    __authSeparate__='否'#后台列表权限
    __thumbsUp__='否'#表属性thumbsUp[是/否]，新增thumbsupnum赞和crazilynum踩字段
    __intelRecom__='否'#智能推荐功能(表属性：[intelRecom（是/否）],新增clicktime[前端不显示该字段]字段（调用info/detail接口的时候更新），按clicktime排序查询)
    __browseClick__='否'#表属性[browseClick:是/否]，点击字段（clicknum），调用info/detail接口的时候后端自动+1）、投票功能（表属性[vote:是/否]，投票字段（votenum）,调用vote接口后端votenum+1
    __foreEndListAuth__='否'#前台列表权限foreEndListAuth[是/否]；当foreEndListAuth=是，刷的表新增用户字段userid，前台list列表接口仅能查看自己的记录和add接口后台赋值userid的值
    __foreEndList__='否'#表属性[foreEndList]前台list:和后台默认的list列表页相似,只是摆在前台,否:指没有此页,是:表示有此页(不需要登陆即可查看),前要登:表示有此页且需要登陆后才能查看
    __isAdmin__='否'#表属性isAdmin=”是”,刷出来的用户表也是管理员，即page和list可以查看所有人的考试记录(同时应用于其他表)
    addtime = models.DateTimeField(auto_now_add=False, verbose_name=u'创建时间')
    yonghuming=models.CharField ( max_length=255,null=False,unique=True, verbose_name='用户名' )
    mima=models.CharField ( max_length=255,null=False, unique=False, verbose_name='密码' )
    xingming=models.CharField ( max_length=255, null=True, unique=False, verbose_name='姓名' )
    xingbie=models.CharField ( max_length=255, null=True, unique=False, verbose_name='性别' )
    touxiang=models.TextField   (  null=True, unique=False, verbose_name='头像' )
    youxiang=models.CharField ( max_length=255, null=True, unique=False, verbose_name='邮箱' )
    shouji=models.CharField ( max_length=255, null=True, unique=False, verbose_name='手机' )
    '''
    yonghuming=VARCHAR
    mima=VARCHAR
    xingming=VARCHAR
    xingbie=VARCHAR
    touxiang=Text
    youxiang=VARCHAR
    shouji=VARCHAR
    '''
    class Meta:
        db_table = 'yonghu'
        verbose_name = verbose_name_plural = '用户'
class zhaopinxinxi(BaseModel):
    __doc__ = u'''zhaopinxinxi'''
    __tablename__ = 'zhaopinxinxi'



    __authTables__={}
    __authPeople__='否'#用户表，表属性loginUserColumn对应的值就是用户名字段，mima就是密码字段
    __sfsh__='否'#表sfsh(是否审核，”是”或”否”)字段和sfhf(审核回复)字段，后台列表(page)的操作中要多一个”审核”按钮，点击”审核”弹出一个页面，包含”是否审核”和”审核回复”，点击确定调用update接口，修改sfsh和sfhf两个字段。
    __authSeparate__='否'#后台列表权限
    __thumbsUp__='否'#表属性thumbsUp[是/否]，新增thumbsupnum赞和crazilynum踩字段
    __intelRecom__='是'#智能推荐功能(表属性：[intelRecom（是/否）],新增clicktime[前端不显示该字段]字段（调用info/detail接口的时候更新），按clicktime排序查询)
    __browseClick__='是'#表属性[browseClick:是/否]，点击字段（clicknum），调用info/detail接口的时候后端自动+1）、投票功能（表属性[vote:是/否]，投票字段（votenum）,调用vote接口后端votenum+1
    __foreEndListAuth__='否'#前台列表权限foreEndListAuth[是/否]；当foreEndListAuth=是，刷的表新增用户字段userid，前台list列表接口仅能查看自己的记录和add接口后台赋值userid的值
    __foreEndList__='是'#表属性[foreEndList]前台list:和后台默认的list列表页相似,只是摆在前台,否:指没有此页,是:表示有此页(不需要登陆即可查看),前要登:表示有此页且需要登陆后才能查看
    __isAdmin__='否'#表属性isAdmin=”是”,刷出来的用户表也是管理员，即page和list可以查看所有人的考试记录(同时应用于其他表)
    addtime = models.DateTimeField(auto_now_add=False, verbose_name=u'创建时间')
    xlyq=models.CharField ( max_length=255, null=True, unique=False, verbose_name='学历要求' )
    gsxz=models.CharField ( max_length=255, null=True, unique=False, verbose_name='公司性质' )
    gsmc=models.CharField ( max_length=255, null=True, unique=False, verbose_name='公司名称' )
    gsgm=models.CharField ( max_length=255, null=True, unique=False, verbose_name='公司规模' )
    fabushijian=models.CharField ( max_length=255, null=True, unique=False, verbose_name='发布时间' )
    laiyuan=models.CharField ( max_length=255, null=True, unique=False, verbose_name='来源' )
    biaoti=models.CharField ( max_length=255,null=False, unique=False, verbose_name='标题' )
    gzdz=models.CharField ( max_length=255, null=True, unique=False, verbose_name='工作地址' )
    xinzi=models.CharField ( max_length=255, null=True, unique=False, verbose_name='薪资' )
    gzjy=models.CharField ( max_length=255, null=True, unique=False, verbose_name='工作经验' )
    clicktime=models.DateTimeField  (  null=True, unique=False, verbose_name='最近点击时间' )
    clicknum=models.IntegerField  (  null=True, unique=False,default='0', verbose_name='点击次数' )
    '''
    xlyq=VARCHAR
    gsxz=VARCHAR
    gsmc=VARCHAR
    gsgm=VARCHAR
    fabushijian=VARCHAR
    laiyuan=VARCHAR
    biaoti=VARCHAR
    gzdz=VARCHAR
    xinzi=VARCHAR
    gzjy=VARCHAR
    clicktime=DateTime
    clicknum=Integer
    '''
    class Meta:
        db_table = 'zhaopinxinxi'
        verbose_name = verbose_name_plural = '招聘信息'
class news(BaseModel):
    __doc__ = u'''news'''
    __tablename__ = 'news'



    __authTables__={}
    addtime = models.DateTimeField(auto_now_add=False, verbose_name=u'创建时间')
    title=models.CharField ( max_length=255,null=False, unique=False, verbose_name='标题' )
    introduction=models.TextField   (  null=True, unique=False, verbose_name='简介' )
    picture=models.TextField   ( null=False, unique=False, verbose_name='图片' )
    content=models.TextField   ( null=False, unique=False, verbose_name='内容' )
    '''
    title=VARCHAR
    introduction=Text
    picture=Text
    content=Text
    '''
    class Meta:
        db_table = 'news'
        verbose_name = verbose_name_plural = '新闻资讯'
class systemintro(BaseModel):
    __doc__ = u'''systemintro'''
    __tablename__ = 'systemintro'



    __authTables__={}
    addtime = models.DateTimeField(auto_now_add=False, verbose_name=u'创建时间')
    title=models.CharField ( max_length=255,null=False, unique=False, verbose_name='标题' )
    subtitle=models.CharField ( max_length=255, null=True, unique=False, verbose_name='副标题' )
    content=models.TextField   ( null=False, unique=False, verbose_name='内容' )
    picture1=models.TextField   (  null=True, unique=False, verbose_name='图片1' )
    picture2=models.TextField   (  null=True, unique=False, verbose_name='图片2' )
    picture3=models.TextField   (  null=True, unique=False, verbose_name='图片3' )
    '''
    title=VARCHAR
    subtitle=VARCHAR
    content=Text
    picture1=Text
    picture2=Text
    picture3=Text
    '''
    class Meta:
        db_table = 'systemintro'
        verbose_name = verbose_name_plural = '关于我们'
class messages(BaseModel):
    __doc__ = u'''messages'''
    __tablename__ = 'messages'



    __authTables__={}
    __hasMessage__='是'#表属性hasMessage为是，新增留言板表messages,字段content（内容），userid（用户id）
    addtime = models.DateTimeField(auto_now_add=False, verbose_name=u'创建时间')
    userid=models.BigIntegerField  ( null=False, unique=False, verbose_name='留言人id' )
    username=models.CharField ( max_length=255, null=True, unique=False, verbose_name='用户名' )
    avatarurl=models.TextField   (  null=True, unique=False, verbose_name='头像' )
    content=models.TextField   ( null=False, unique=False, verbose_name='留言内容' )
    cpicture=models.TextField   (  null=True, unique=False, verbose_name='留言图片' )
    reply=models.TextField   (  null=True, unique=False, verbose_name='回复内容' )
    rpicture=models.TextField   (  null=True, unique=False, verbose_name='回复图片' )
    '''
    userid=BigInteger
    username=VARCHAR
    avatarurl=Text
    content=Text
    cpicture=Text
    reply=Text
    rpicture=Text
    '''
    class Meta:
        db_table = 'messages'
        verbose_name = verbose_name_plural = '留言板'
class discusszhaopinxinxi(BaseModel):
    __doc__ = u'''discusszhaopinxinxi'''
    __tablename__ = 'discusszhaopinxinxi'



    __authTables__={}
    addtime = models.DateTimeField(auto_now_add=False, verbose_name=u'创建时间')
    refid=models.BigIntegerField  ( null=False, unique=False, verbose_name='关联表id' )
    userid=models.BigIntegerField  ( null=False, unique=False, verbose_name='用户id' )
    avatarurl=models.TextField   (  null=True, unique=False, verbose_name='头像' )
    nickname=models.CharField ( max_length=255, null=True, unique=False, verbose_name='用户名' )
    content=models.TextField   ( null=False, unique=False, verbose_name='评论内容' )
    reply=models.TextField   (  null=True, unique=False, verbose_name='回复内容' )
    '''
    refid=BigInteger
    userid=BigInteger
    avatarurl=Text
    nickname=VARCHAR
    content=Text
    reply=Text
    '''
    class Meta:
        db_table = 'discusszhaopinxinxi'
        verbose_name = verbose_name_plural = 'zhaopinxinxi评论表'
class zhaopinxinxiRecomm(BaseModel):
    __doc__ = u'''zhaopinxinxiRecomm'''
    __tablename__ = 'zhaopinxinxirecomm'



    __authTables__={}
    yonghuName=models.CharField ( max_length=255, null=True, unique=False, verbose_name='用户名' )
    Like1=models.CharField ( max_length=255, null=True, unique=False, verbose_name='Like1' )
    Like2=models.CharField ( max_length=255, null=True, unique=False, verbose_name='Like2' )
    Like3=models.CharField ( max_length=255, null=True, unique=False, verbose_name='Like3' )
    Like4=models.CharField ( max_length=255, null=True, unique=False, verbose_name='Like4' )
    Like5=models.CharField ( max_length=255, null=True, unique=False, verbose_name='Like5' )
    Like6=models.CharField ( max_length=255, null=True, unique=False, verbose_name='Like6' )
    Like7=models.CharField ( max_length=255, null=True, unique=False, verbose_name='Like7' )
    Like8=models.CharField ( max_length=255, null=True, unique=False, verbose_name='Like8' )
    class Meta:
        db_table = 'zhaopinxinxirecomm'
        verbose_name = verbose_name_plural = '招聘信息推荐表'

    # '''
    # content=models.TextField   ( null=False, unique=False, verbose_name='评论内容' )
    # reply=models.TextField   (  null=True, unique=False, verbose_name='回复内容' )
    # '''
    # refid=BigInteger
    # userid=BigInteger
    # avatarurl=Text
    # nickname=VARCHAR
    # content=Text
    # reply=Text
    # '''
    # class Meta:
    #     db_table = 'discusszhaopinxinxi'
    #     verbose_name = verbose_name_plural = 'zhaopinxinxi评论表'