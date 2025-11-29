# 数据爬取文件

import scrapy
import pymysql
import pymssql
from ..items import ZhaopinxinxiItem
import time
import re
import random
import platform
import json
import os
from urllib.parse import urlparse
import requests
import emoji

# 招聘信息
class ZhaopinxinxiSpider(scrapy.Spider):
    name = 'zhaopinxinxiSpider'
    spiderUrl = 'https://cupid.51job.com/open/noauth/search-pc?api_key=51job&timestamp=1677042252&keyword=IT&searchType=2&function=&industry=&jobArea=000000&jobArea2=&landmark=&metro=&salary=&workYear=&degree=&companyType=&companySize=&jobType=&issueDate=&sortType=0&pageNum={}&requestId=93c1897338e6048f265dfc743365ecb2&pageSize=50&source=1&accountId=&pageCode=sou%7Csou%7Csoulb'
    start_urls = spiderUrl.split(";")
    protocol = ''
    hostname = ''
    headers = {
        "Referer":"https://we.51job.com/",
"sign":"1b6fc833afc4bbdfeb814c2d29fd34a1015e8d21eb0f1e7d61f45cd504c12f5f"
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_requests(self):

        plat = platform.system().lower()
        if plat == 'linux' or plat == 'windows':
            connect = self.db_connect()
            cursor = connect.cursor()
            if self.table_exists(cursor, '05zp2_zhaopinxinxi') == 1:
                cursor.close()
                connect.close()
                self.temp_data()
                return

        pageNum = 1 + 1
        for url in self.start_urls:
            if '{}' in url:
                for page in range(1, pageNum):
                    next_link = url.format(page)
                    yield scrapy.Request(
                        url=next_link,
                        headers=self.headers,
                        callback=self.parse
                    )
            else:
                yield scrapy.Request(
                    url=url,
                    callback=self.parse
                )

    # 列表解析
    def parse(self, response):
        
        _url = urlparse(self.spiderUrl)
        self.protocol = _url.scheme
        self.hostname = _url.netloc
        plat = platform.system().lower()
        if plat == 'windows_bak':
            pass
        elif plat == 'linux' or plat == 'windows':
            connect = self.db_connect()
            cursor = connect.cursor()
            if self.table_exists(cursor, '05zp2_zhaopinxinxi') == 1:
                cursor.close()
                connect.close()
                self.temp_data()
                return

        data = json.loads(response.body)
        list = data["resultbody"]["job"]["items"]
        
        for item in list:

            fields = ZhaopinxinxiItem()




            fields["laiyuan"] = item["jobHref"]
            fields["biaoti"] = item["jobName"]
            fields["gzdz"] = item["jobAreaString"]
            fields["xinzi"] = item["provideSalaryString"]
            fields["gzjy"] = item["workYearString"]
            fields["xlyq"] = item["degreeString"]
            fields["gsmc"] = item["fullCompanyName"]
            fields["gsxz"] = item["companyTypeString"]
            fields["gsgm"] = item["companySizeString"]
            fields["fabushijian"] = item["issueDateString"]


            detailUrlRule = item["jobHref"]

            if detailUrlRule.startswith('http') or self.hostname in detailUrlRule:
                pass
            else:
                detailUrlRule = self.protocol + '://' + self.hostname + detailUrlRule
                fields["laiyuan"] = detailUrlRule

            yield scrapy.Request(url=detailUrlRule, meta={'fields': fields}, callback=self.detail_parse)

    # 详情解析
    def detail_parse(self, response):
        fields = response.meta['fields']



        return fields

    # 去除多余html标签
    def remove_html(self, html):
        if html == None:
            return ''
        pattern = re.compile(r'<[^>]+>', re.S)
        return pattern.sub('', html).strip()

    # 数据库连接
    def db_connect(self):
        type = self.settings.get('TYPE', 'mysql')
        host = self.settings.get('HOST', 'localhost')
        port = int(self.settings.get('PORT', 3306))
        user = self.settings.get('USER', 'root')
        password = self.settings.get('PASSWORD', '123456')

        try:
            database = self.databaseName
        except:
            database = self.settings.get('DATABASE', '')

        if type == 'mysql':
            connect = pymysql.connect(host=host, port=port, db=database, user=user, passwd=password, charset='utf8')
        else:
            connect = pymssql.connect(host=host, user=user, password=password, database=database)

        return connect

    # 断表是否存在
    def table_exists(self, cursor, table_name):
        cursor.execute("show tables;")
        tables = [cursor.fetchall()]
        table_list = re.findall('(\'.*?\')',str(tables))
        table_list = [re.sub("'",'',each) for each in table_list]

        if table_name in table_list:
            return 1
        else:
            return 0

    # 数据缓存源
    def temp_data(self):

        connect = self.db_connect()
        cursor = connect.cursor()
        sql = '''
            insert into zhaopinxinxi(
                laiyuan
                ,biaoti
                ,gzdz
                ,xinzi
                ,gzjy
                ,xlyq
                ,gsmc
                ,gsxz
                ,gsgm
                ,fabushijian
            )
            select
                laiyuan
                ,biaoti
                ,gzdz
                ,xinzi
                ,gzjy
                ,xlyq
                ,gsmc
                ,gsxz
                ,gsgm
                ,fabushijian
            from 05zp2_zhaopinxinxi
            where(not exists (select
                laiyuan
                ,biaoti
                ,gzdz
                ,xinzi
                ,gzjy
                ,xlyq
                ,gsmc
                ,gsxz
                ,gsgm
                ,fabushijian
            from zhaopinxinxi where
             zhaopinxinxi.laiyuan=05zp2_zhaopinxinxi.laiyuan
            and zhaopinxinxi.biaoti=05zp2_zhaopinxinxi.biaoti
            and zhaopinxinxi.gzdz=05zp2_zhaopinxinxi.gzdz
            and zhaopinxinxi.xinzi=05zp2_zhaopinxinxi.xinzi
            and zhaopinxinxi.gzjy=05zp2_zhaopinxinxi.gzjy
            and zhaopinxinxi.xlyq=05zp2_zhaopinxinxi.xlyq
            and zhaopinxinxi.gsmc=05zp2_zhaopinxinxi.gsmc
            and zhaopinxinxi.gsxz=05zp2_zhaopinxinxi.gsxz
            and zhaopinxinxi.gsgm=05zp2_zhaopinxinxi.gsgm
            and zhaopinxinxi.fabushijian=05zp2_zhaopinxinxi.fabushijian
            ))
            limit {0}
        '''.format(random.randint(20,30))

        cursor.execute(sql)
        connect.commit()

        connect.close()
