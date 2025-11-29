# 数据容器文件

import scrapy

class SpiderItem(scrapy.Item):
    pass

class ZhaopinxinxiItem(scrapy.Item):
    # 来源
    laiyuan = scrapy.Field()
    # 标题
    biaoti = scrapy.Field()
    # 工作地址
    gzdz = scrapy.Field()
    # 薪资
    xinzi = scrapy.Field()
    # 工作经验
    gzjy = scrapy.Field()
    # 学历要求
    xlyq = scrapy.Field()
    # 公司名称
    gsmc = scrapy.Field()
    # 公司性质
    gsxz = scrapy.Field()
    # 公司规模
    gsgm = scrapy.Field()
    # 发布时间
    fabushijian = scrapy.Field()

