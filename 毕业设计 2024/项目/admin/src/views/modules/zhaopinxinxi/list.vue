<template>
	<div class="main-content">
		<!-- 列表页 -->
		<template v-if="showFlag">
			<el-form class="center-form-pv" :style='{"width":"250px","padding":"0 6px","margin":"0 0 20px"}' :inline="true" :model="searchForm">
				<el-row :style='{"flexDirection":"column","display":"flex"}' >
					<div :style='{"margin":"0","flexDirection":"column","display":"flex"}'>
						<label :style='{"margin":"0","color":"#000","textAlign":"center","display":"inline-block","width":"100%","lineHeight":"40px","fontSize":"18px","fontWeight":"500","height":"40px"}' class="item-label">学历要求</label>
						<el-input v-model="searchForm.xlyq" placeholder="学历要求" clearable></el-input>
					</div>
					<div :style='{"margin":"0","flexDirection":"column","display":"flex"}'>
						<label :style='{"margin":"0","color":"#000","textAlign":"center","display":"inline-block","width":"100%","lineHeight":"40px","fontSize":"18px","fontWeight":"500","height":"40px"}' class="item-label">公司名称</label>
						<el-input v-model="searchForm.gsmc" placeholder="公司名称" clearable></el-input>
					</div>
					<div :style='{"margin":"0","flexDirection":"column","display":"flex"}'>
						<label :style='{"margin":"0","color":"#000","textAlign":"center","display":"inline-block","width":"100%","lineHeight":"40px","fontSize":"18px","fontWeight":"500","height":"40px"}' class="item-label">标题</label>
						<el-input v-model="searchForm.biaoti" placeholder="标题" clearable></el-input>
					</div>
					<div :style='{"margin":"0","flexDirection":"column","display":"flex"}'>
						<label :style='{"margin":"0","color":"#000","textAlign":"center","display":"inline-block","width":"100%","lineHeight":"40px","fontSize":"18px","fontWeight":"500","height":"40px"}' class="item-label">工作地址</label>
						<el-input v-model="searchForm.gzdz" placeholder="工作地址" clearable></el-input>
					</div>
					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0 24px","boxShadow":"0px 2px 2px 0px #78ABC3","outline":"none","margin":"10px 0 0","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"auto","fontSize":"18px","height":"40px"}' type="success" @click="search()">查询</el-button>
				</el-row>

				<el-row :style='{"margin":"20px 0","flexDirection":"column","display":"flex"}'>
					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0","boxShadow":"0px 2px 2px 0px #78ABC3","margin":"0 0 10px","outline":"none","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"100%","fontSize":"18px","height":"40px"}' v-if="isAuth('zhaopinxinxi','新增')" type="success" @click="addOrUpdateHandler()">新增</el-button>
					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0","boxShadow":"0px 2px 2px 0px #78ABC3","margin":"0 0 10px","outline":"none","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"100%","fontSize":"18px","height":"40px"}' v-if="isAuth('zhaopinxinxi','删除')" :disabled="dataListSelections.length <= 0" type="danger" @click="deleteHandler()">删除</el-button>


					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0","boxShadow":"0px 2px 2px 0px #78ABC3","margin":"0 0 10px","outline":"none","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"100%","fontSize":"18px","height":"40px"}' v-if="isAuth('zhaopinxinxi','爬取数据')" type="success" @click="spider()">爬取数据</el-button>


					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0","boxShadow":"0px 2px 2px 0px #78ABC3","margin":"0 0 10px","outline":"none","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"100%","fontSize":"18px","height":"40px"}' v-if="isAuth('zhaopinxinxi','学历统计')" type="warning" @click="chartDialog1()">学历统计</el-button>
					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0","boxShadow":"0px 2px 2px 0px #78ABC3","margin":"0 0 10px","outline":"none","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"100%","fontSize":"18px","height":"40px"}' v-if="isAuth('zhaopinxinxi','公司性质统计')" type="warning" @click="chartDialog2()">公司性质统计</el-button>
					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0","boxShadow":"0px 2px 2px 0px #78ABC3","margin":"0 0 10px","outline":"none","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"100%","fontSize":"18px","height":"40px"}' v-if="isAuth('zhaopinxinxi','公司规模统计')" type="warning" @click="chartDialog3()">公司规模统计</el-button>
					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0","boxShadow":"0px 2px 2px 0px #78ABC3","margin":"0 0 10px","outline":"none","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"100%","fontSize":"18px","height":"40px"}' v-if="isAuth('zhaopinxinxi','工作地址统计')" type="warning" @click="chartDialog4()">工作地址统计</el-button>
					<el-button :style='{"border":"1px solid #97C9D6","cursor":"pointer","padding":"0","boxShadow":"0px 2px 2px 0px #78ABC3","margin":"0 0 10px","outline":"none","color":"#78ABC3","borderRadius":"30px","background":"#fff","width":"100%","fontSize":"18px","height":"40px"}' v-if="isAuth('zhaopinxinxi','资薪统计')" type="warning" @click="chartDialog5()">资薪统计</el-button>
				</el-row>
			</el-form>
			
			<!-- <div> -->
				<el-table class="tables"
					:stripe='false'
					:style='{"padding":"0","boxShadow":"0px 4px 10px 0px rgba(0,0,0,0.3020)","borderColor":"#eee","borderRadius":"30px","borderWidth":"1px 0 0 1px","background":"#fff","width":"100%","borderStyle":"solid","height":"fit-content"}' 
					v-if="isAuth('zhaopinxinxi','查看')"
					:data="dataList"
					v-loading="dataListLoading"
				@selection-change="selectionChangeHandler">
					<el-table-column :resizable='true' type="selection" align="center" width="50"></el-table-column>
					<el-table-column :resizable='true' :sortable='false' label="索引" type="index" width="50" />
					<el-table-column :resizable='true' :sortable='false'  
						prop="xlyq"
					label="学历要求">
						<template slot-scope="scope">
							{{scope.row.xlyq}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="gsxz"
					label="公司性质">
						<template slot-scope="scope">
							{{scope.row.gsxz}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="gsmc"
					label="公司名称">
						<template slot-scope="scope">
							{{scope.row.gsmc}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="gsgm"
					label="公司规模">
						<template slot-scope="scope">
							{{scope.row.gsgm}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="fabushijian"
					label="发布时间">
						<template slot-scope="scope">
							{{scope.row.fabushijian}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="laiyuan"
					label="来源">
						<template slot-scope="scope">
							<span @click="linkOthers(scope.row.laiyuan)">{{scope.row.laiyuan}}</span>
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="biaoti"
					label="标题">
						<template slot-scope="scope">
							{{scope.row.biaoti}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="gzdz"
					label="工作地址">
						<template slot-scope="scope">
							{{scope.row.gzdz}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="xinzi"
					label="薪资">
						<template slot-scope="scope">
							{{scope.row.xinzi}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="gzjy"
					label="工作经验">
						<template slot-scope="scope">
							{{scope.row.gzjy}}
						</template>
					</el-table-column>
					<el-table-column :resizable='true' :sortable='false'  
						prop="clicknum"
					label="点击次数">
						<template slot-scope="scope">
							{{scope.row.clicknum}}
						</template>
					</el-table-column>
					<el-table-column width="300" label="操作">
						<template slot-scope="scope">
							<el-button :style='{"border":"0","cursor":"pointer","padding":"0 24px","margin":"0 10px 6px 0","outline":"none","color":"#fff","borderRadius":"30px","background":"#93C7B3","width":"auto","fontSize":"18px","height":"32px"}' v-if=" isAuth('zhaopinxinxi','查看')" type="success" size="mini" @click="addOrUpdateHandler(scope.row.id,'info')">详情</el-button>
							<el-button :style='{"border":"0","cursor":"pointer","padding":"0 24px","margin":"0 10px 6px 0","outline":"none","color":"#fff","borderRadius":"30px","background":"#93C7B3","width":"auto","fontSize":"18px","height":"32px"}' v-if=" isAuth('zhaopinxinxi','修改')" type="primary" size="mini" @click="addOrUpdateHandler(scope.row.id)">修改</el-button>


							<el-button :style='{"border":"0","cursor":"pointer","padding":"0 24px","margin":"0 10px 6px 0","outline":"none","color":"#fff","borderRadius":"30px","background":"#93C7B3","width":"auto","fontSize":"18px","height":"32px"}' v-if="isAuth('zhaopinxinxi','查看评论')" type="primary" size="mini" @click="disscussListHandler(scope.row.id)">查看评论</el-button>



							<el-button :style='{"border":"0","cursor":"pointer","padding":"0 24px","margin":"0 10px 6px 0","outline":"none","color":"#fff","borderRadius":"30px","background":"#93C7B3","width":"auto","fontSize":"18px","height":"32px"}' v-if="isAuth('zhaopinxinxi','删除') " type="danger" size="mini" @click="deleteHandler(scope.row.id)">删除</el-button>
						</template>
					</el-table-column>
				</el-table>
				<el-pagination
					@size-change="sizeChangeHandle"
					@current-change="currentChangeHandle"
					:current-page="pageIndex"
					background
					:page-sizes="[10, 20, 30, 50]"
					:page-size="pageSize"
					:layout="layouts.join()"
					:total="totalPage"
					prev-text="<"
					next-text=">"
					:hide-on-single-page="true"
					:style='{"width":"100%","padding":"0","margin":"20px 0 0","whiteSpace":"nowrap","color":"#333","fontWeight":"500"}'
				></el-pagination>
			<!-- </div> -->
		</template>
		
		<!-- 添加/修改页面  将父组件的search方法传递给子组件-->
		<add-or-update v-if="addOrUpdateFlag" :parent="this" ref="addOrUpdate"></add-or-update>





		<el-dialog
		  title="学历统计"
		  :visible.sync="chartVisiable1"
		  width="800">
			<div id="xlyqChart1" style="width:100%;height:600px;"></div>
		  <span slot="footer" class="dialog-footer">
			<el-button @click="chartDialog1">返回</el-button>
		  </span>
		</el-dialog>
		<el-dialog
		  title="公司性质统计"
		  :visible.sync="chartVisiable2"
		  width="800">
			<div id="gsxzChart2" style="width:100%;height:600px;"></div>
		  <span slot="footer" class="dialog-footer">
			<el-button @click="chartDialog2">返回</el-button>
		  </span>
		</el-dialog>
		<el-dialog
		  title="公司规模统计"
		  :visible.sync="chartVisiable3"
		  width="800">
			<div id="gsgmChart3" style="width:100%;height:600px;"></div>
		  <span slot="footer" class="dialog-footer">
			<el-button @click="chartDialog3">返回</el-button>
		  </span>
		</el-dialog>
		<el-dialog
		  title="工作地址统计"
		  :visible.sync="chartVisiable4"
		  width="800">
			<div id="gzdzChart4" style="width:100%;height:600px;"></div>
		  <span slot="footer" class="dialog-footer">
			<el-button @click="chartDialog4">返回</el-button>
		  </span>
		</el-dialog>
		<el-dialog
		  title="资薪统计"
		  :visible.sync="chartVisiable5"
		  width="800">
			<div id="xinziChart5" style="width:100%;height:600px;"></div>
		  <span slot="footer" class="dialog-footer">
			<el-button @click="chartDialog5">返回</el-button>
		  </span>
		</el-dialog>
	</div>
</template>

<script>
import * as echarts from 'echarts'
import axios from 'axios'
import AddOrUpdate from "./add-or-update";
export default {
  data() {
    return {
      searchForm: {
        key: ""
      },
      form:{},
      dataList: [],
      pageIndex: 1,
      pageSize: 10,
      totalPage: 0,
      dataListLoading: false,
      dataListSelections: [],
      showFlag: true,
      sfshVisiable: false,
      shForm: {},
      chartVisiable: false,
      chartVisiable1: false,
      chartVisiable2: false,
      chartVisiable3: false,
      chartVisiable4: false,
      chartVisiable5: false,
      addOrUpdateFlag:false,
      layouts: ["total","prev","pager","next","sizes","jumper"],

    };
  },
  created() {
    this.init();
    this.getDataList();
    this.contentStyleChange()
  },
  mounted() {
  },
  filters: {
    htmlfilter: function (val) {
      return val.replace(/<[^>]*>/g).replace(/undefined/g,'');
    }
  },
  components: {
    AddOrUpdate,
  },
  methods: {
    linkOthers(url) {
        window.open(url);//打开一个新的标签页
    },

    contentStyleChange() {
      this.contentPageStyleChange()
    },
    // 分页
    contentPageStyleChange(){
      let arr = []

      // if(this.contents.pageTotal) arr.push('total')
      // if(this.contents.pageSizes) arr.push('sizes')
      // if(this.contents.pagePrevNext){
      //   arr.push('prev')
      //   if(this.contents.pagePager) arr.push('pager')
      //   arr.push('next')
      // }
      // if(this.contents.pageJumper) arr.push('jumper')
      // this.layouts = arr.join()
      // this.contents.pageEachNum = 10
    },



//统计接口
    chartDialog1() {
      this.chartVisiable1 = !this.chartVisiable1;
      this.$nextTick(()=>{

        var xlyqChart1 = echarts.init(document.getElementById("xlyqChart1"),'green');
        this.$http({
            url: "zhaopinxinxi/group/xlyq",
            method: "get",
        }).then(({ data }) => {
            if (data && data.code === 0) {
                let res = data.data;
                let xAxis = [];
                let yAxis = [];
                let pArray = []
                for(let i=0;i<res.length;i++){
                    xAxis.push(res[i].xlyq);
                    yAxis.push(parseFloat((res[i].total)));
                    pArray.push({
                        value: parseFloat((res[i].total)),
                        name: res[i].xlyq
                    })
                }
                var option = {};
                option = {
                        title: {
                            text: '学历统计',
                            left: 'center'
                        },
                        legend: {
                          orient: 'vertical',
                          left: 'left'
                        },
                        tooltip: {
                          trigger: 'item',
                          formatter: '{b} : {c} ({d}%)'
                        },
                        series: [
                            {
                                type: 'pie',
                                radius: ['25%', '55%'],
                                roseType: 'area',
                                center: ['50%', '60%'],
                                data: pArray,
                                emphasis: {
                                    itemStyle: {
                                        shadowBlur: 10,
                                        shadowOffsetX: 0,
                                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                                    }
                                }
                            }
                        ]
                };
                // 使用刚指定的配置项和数据显示图表。
                xlyqChart1.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    xlyqChart1.resize();
                };
            }
        });
      })
    },

//统计接口
    chartDialog2() {
      this.chartVisiable2 = !this.chartVisiable2;
      this.$nextTick(()=>{

        var gsxzChart2 = echarts.init(document.getElementById("gsxzChart2"),'green');
        this.$http({
            url: "zhaopinxinxi/group/gsxz",
            method: "get",
        }).then(({ data }) => {
            if (data && data.code === 0) {
                let res = data.data;
                let xAxis = [];
                let yAxis = [];
                let pArray = []
                for(let i=0;i<res.length;i++){
                    xAxis.push(res[i].gsxz);
                    yAxis.push(parseFloat((res[i].total)));
                    pArray.push({
                        value: parseFloat((res[i].total)),
                        name: res[i].gsxz
                    })
                }
                var option = {};
                option = {
                    title: {
                        text: '公司性质统计',
                        left: 'center'
                    },
                    tooltip: {
                      trigger: 'item',
                      formatter: '{b} : {c}'
                    },
                    xAxis: {
                        type: 'category',
                        data: xAxis
                    },
                    yAxis: {
                        type: 'value'
                    },
                    series: [{
                        data: yAxis,
                        type: 'bar'
                    }]
                };
                // 使用刚指定的配置项和数据显示图表。
                gsxzChart2.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    gsxzChart2.resize();
                };
            }
        });
      })
    },

//统计接口
    chartDialog3() {
      this.chartVisiable3 = !this.chartVisiable3;
      this.$nextTick(()=>{

        var gsgmChart3 = echarts.init(document.getElementById("gsgmChart3"),'green');
        this.$http({
            url: "zhaopinxinxi/group/gsgm",
            method: "get",
        }).then(({ data }) => {
            if (data && data.code === 0) {
                let res = data.data;
                let xAxis = [];
                let yAxis = [];
                let pArray = []
                for(let i=0;i<res.length;i++){
                    xAxis.push(res[i].gsgm);
                    yAxis.push(parseFloat((res[i].total)));
                    pArray.push({
                        value: parseFloat((res[i].total)),
                        name: res[i].gsgm
                    })
                }
                var option = {};
                option = {
                    title: {
                        text: '公司规模统计',
                        left: 'center'
                    },
                    tooltip: {
                      trigger: 'item',
                      formatter: '{b} : {c}'
                    },
                    xAxis: {
                        type: 'value'
                    },
                    yAxis: {
                        type: 'category',
                        data: xAxis
                    },
                    series: [{
                        data: yAxis,
                        type: 'bar'
                    }]
                };
                // 使用刚指定的配置项和数据显示图表。
                gsgmChart3.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    gsgmChart3.resize();
                };
            }
        });
      })
    },

//统计接口
    chartDialog4() {
      this.chartVisiable4 = !this.chartVisiable4;
      this.$nextTick(()=>{

        var gzdzChart4 = echarts.init(document.getElementById("gzdzChart4"),'green');
        this.$http({
            url: "zhaopinxinxi/group/gzdz",
            method: "get",
        }).then(({ data }) => {
            if (data && data.code === 0) {
                let res = data.data;
                let xAxis = [];
                let yAxis = [];
                let pArray = []
                for(let i=0;i<res.length;i++){
                    xAxis.push(res[i].gzdz);
                    yAxis.push(parseFloat((res[i].total)));
                    pArray.push({
                        value: parseFloat((res[i].total)),
                        name: res[i].gzdz
                    })
                }
                var option = {};
                option = {
                    title: {
                        text: '工作地址统计',
                        left: 'center'
                    },
                    tooltip: {
                      trigger: 'item',
                      formatter: '{b} : {c}'
                    },
                    xAxis: {
                        type: 'category',
                        boundaryGap: false,
                        data: xAxis
                    },
                    yAxis: {
                        type: 'value'
                    },
                    series: [{
                        data: yAxis,
                        type: 'line',
                        areaStyle: {}
                    }]
                };
                // 使用刚指定的配置项和数据显示图表。
                gzdzChart4.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    gzdzChart4.resize();
                };
            }
        });
      })
    },

//统计接口
    chartDialog5() {
      this.chartVisiable5 = !this.chartVisiable5;
      this.$nextTick(()=>{

        var xinziChart5 = echarts.init(document.getElementById("xinziChart5"),'green');
        this.$http({
            url: "zhaopinxinxi/group/xinzi",
            method: "get",
        }).then(({ data }) => {
            if (data && data.code === 0) {
                let res = data.data;
                let xAxis = [];
                let yAxis = [];
                let pArray = []
                for(let i=0;i<res.length;i++){
                    xAxis.push(res[i].xinzi);
                    yAxis.push(parseFloat((res[i].total)));
                    pArray.push({
                        value: parseFloat((res[i].total)),
                        name: res[i].xinzi
                    })
                }
                var option = {};
                option = {
                    title: {
                        text: '资薪统计',
                        left: 'center'
                    },
                    tooltip: {
                      trigger: 'item',
                      formatter: '{b} : {c}'
                    },
                    xAxis: {
                        type: 'category',
                        data: xAxis
                    },
                    yAxis: {
                        type: 'value'
                    },
                    series: [{
                        data: yAxis,
                        type: 'bar'
                    }]
                };
                // 使用刚指定的配置项和数据显示图表。
                xinziChart5.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    xinziChart5.resize();
                };
            }
        });
      })
    },

    init () {
    },
    search() {
      this.pageIndex = 1;
      this.getDataList();
    },

    // 获取数据列表
    getDataList() {
      this.dataListLoading = true;
      let params = {
        page: this.pageIndex,
        limit: this.pageSize,
        sort: 'id',
        order: 'desc',
      }
           if(this.searchForm.xlyq!='' && this.searchForm.xlyq!=undefined){
            params['xlyq'] = '%' + this.searchForm.xlyq + '%'
          }
           if(this.searchForm.gsmc!='' && this.searchForm.gsmc!=undefined){
            params['gsmc'] = '%' + this.searchForm.gsmc + '%'
          }
           if(this.searchForm.biaoti!='' && this.searchForm.biaoti!=undefined){
            params['biaoti'] = '%' + this.searchForm.biaoti + '%'
          }
           if(this.searchForm.gzdz!='' && this.searchForm.gzdz!=undefined){
            params['gzdz'] = '%' + this.searchForm.gzdz + '%'
          }
      this.$http({
        url: "zhaopinxinxi/page",
        method: "get",
        params: params
      }).then(({ data }) => {
        if (data && data.code === 0) {
          this.dataList = data.data.list;
          this.totalPage = data.data.total;
        } else {
          this.dataList = [];
          this.totalPage = 0;
        }
        this.dataListLoading = false;
      });
    },
    // 每页数
    sizeChangeHandle(val) {
      this.pageSize = val;
      this.pageIndex = 1;
      this.getDataList();
    },
    // 当前页
    currentChangeHandle(val) {
      this.pageIndex = val;
      this.getDataList();
    },
    // 多选
    selectionChangeHandler(val) {
      this.dataListSelections = val;
    },
    // 添加/修改
    addOrUpdateHandler(id,type) {
      this.showFlag = false;
      this.addOrUpdateFlag = true;
      this.crossAddOrUpdateFlag = false;
      if(type!='info'){
        type = 'else';
      }
      this.$nextTick(() => {
        this.$refs.addOrUpdate.init(id,type);
      });
    },
    // 查看评论
    disscussListHandler(id,type) {
	this.$router.push({path:'/discusszhaopinxinxi',query:{refid:id}});
    },
    // 下载
    download(file){
      window.open(`${file}`)
    },
    // 删除
    deleteHandler(id) {
      var ids = id
        ? [Number(id)]
        : this.dataListSelections.map(item => {
            return Number(item.id);
          });
      this.$confirm(`确定进行[${id ? "删除" : "批量删除"}]操作?`, "提示", {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning"
      }).then(() => {
        this.$http({
          url: "zhaopinxinxi/delete",
          method: "post",
          data: ids
        }).then(({ data }) => {
          if (data && data.code === 0) {
            this.$message({
              message: "操作成功",
              type: "success",
              duration: 1500,
              onClose: () => {
                this.search();
              }
            });
          } else {
            this.$message.error(data.msg);
          }
        });
      });
    },

    spider() {
        this.$message({
              message: "数据爬取中...",
              type: "success",
              duration: 3000,
              onClose: () => {
                this.search();
              }
            });
        this.$http({
          url: "spider/zhaopinxinxi",
          method: "get",
        }).then(({ data }) => {
          if (data && data.code === 0) {
            this.$message({
                message: "爬取完成",
                type: "success",
                duration: 1500,
                onClose: () => {
                    this.getDataList(this.roleName);
                }
            });
          } else {
            this.$alert(data.msg)
          }
        });
    },

  }

};
</script>
<style lang="scss" scoped>
	
	.center-form-pv {
	  .el-date-editor.el-input {
	    width: auto;
	  }
	}
	
	.el-input {
	  width: auto;
	}
	
	// form
	.center-form-pv .el-input /deep/ .el-input__inner {
				border: 1px solid #78ABC3;
				border-radius: 0;
				padding: 0 12px;
				box-shadow: 0px 2px 2px 0px #78ABC3;
				outline: none;
				color: #000;
				width: 100%;
				font-size: 18px;
				height: 40px;
			}
	
	.center-form-pv .el-select /deep/ .el-input__inner {
				border: 1px solid #78ABC3;
				border-radius: 0;
				padding: 0 10px;
				box-shadow: 0px 2px 2px 0px #78ABC3;
				outline: none;
				color: #000;
				width: 100%;
				font-size: 18px;
				height: 40px;
			}
	
	.center-form-pv .el-date-editor /deep/ .el-input__inner {
				border: 1px solid #78ABC3;
				border-radius: 0;
				padding: 0 10px 0 30px;
				box-shadow: 0px 2px 2px 0px #78ABC3;
				outline: none;
				color: #000;
				width: 100%;
				font-size: 18px;
				height: 40px;
			}
	
	// table
	.el-table /deep/ .el-table__header-wrapper thead {
				color: #999;
				font-weight: 500;
				width: 100%;
			}
	
	.el-table /deep/ .el-table__header-wrapper thead tr {
				background: #93C7B3;
			}
	
	.el-table /deep/ .el-table__header-wrapper thead tr th {
				padding: 12px 0;
				color: #fff;
				background: transparent;
				border-color: #eee;
				border-width: 0 1px 1px 0;
				border-style: solid;
				text-align: center;
			}

	.el-table /deep/ .el-table__header-wrapper thead tr th .cell {
				padding: 0 10px;
				word-wrap: normal;
				word-break: break-all;
				white-space: normal;
				font-weight: bold;
				display: inline-block;
				vertical-align: middle;
				width: 100%;
				line-height: 24px;
				position: relative;
				text-overflow: ellipsis;
			}

	
	.el-table /deep/ .el-table__body-wrapper tbody {
				width: 100%;
			}

	.el-table /deep/ .el-table__body-wrapper tbody tr {
				background: #fff;
			}
	
	.el-table /deep/ .el-table__body-wrapper tbody tr td {
				padding: 12px 0;
				color: #999;
				background: #fff;
				border-color: #6F6F97;
				border-width: 0 1px 1px 0;
				border-style: solid;
				text-align: center;
			}
	
		
	.el-table /deep/ .el-table__body-wrapper tbody tr:hover td {
				padding: 12px 0;
				color: #000;
				background: rgba(147, 199, 179, 0.5);
				border-color: #6F6F97;
				border-width: 0 1px 1px 0;
				border-style: solid;
			}
	
	.el-table /deep/ .el-table__body-wrapper tbody tr td {
				padding: 12px 0;
				color: #999;
				background: #fff;
				border-color: #6F6F97;
				border-width: 0 1px 1px 0;
				border-style: solid;
				text-align: center;
			}

	.el-table /deep/ .el-table__body-wrapper tbody tr td .cell {
				padding: 0 10px;
				overflow: hidden;
				word-break: break-all;
				white-space: normal;
				line-height: 24px;
				text-overflow: ellipsis;
			}
	
	// pagination
	.main-content .el-pagination /deep/ .el-pagination__total {
				margin: 0 10px 0 0;
				color: #666;
				font-weight: 400;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .btn-prev {
				border: none;
				border-radius: 2px;
				padding: 0;
				margin: 0 5px;
				color: #666;
				background: #f4f4f5;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				min-width: 35px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .btn-next {
				border: none;
				border-radius: 2px;
				padding: 0;
				margin: 0 5px;
				color: #666;
				background: #f4f4f5;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				min-width: 35px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .btn-prev:disabled {
				border: none;
				cursor: not-allowed;
				border-radius: 2px;
				padding: 0;
				margin: 0 5px;
				color: #C0C4CC;
				background: #f4f4f5;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .btn-next:disabled {
				border: none;
				cursor: not-allowed;
				border-radius: 2px;
				padding: 0;
				margin: 0 5px;
				color: #C0C4CC;
				background: #f4f4f5;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}

	.main-content .el-pagination /deep/ .el-pager {
				padding: 0;
				margin: 0;
				display: inline-block;
				vertical-align: top;
			}

	.main-content .el-pagination /deep/ .el-pager .number {
				cursor: pointer;
				padding: 0 4px;
				margin: 0 5px;
				color: #666;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				border-radius: 2px;
				background: #f4f4f5;
				text-align: center;
				min-width: 30px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .el-pager .number:hover {
				cursor: pointer;
				padding: 0 4px;
				margin: 0 5px;
				color: #fff;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				border-radius: 2px;
				background: #93C7B3;
				text-align: center;
				min-width: 30px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .el-pager .number.active {
				cursor: default;
				padding: 0 4px;
				margin: 0 5px;
				color: #FFF;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				border-radius: 2px;
				background: #93C7B3;
				text-align: center;
				min-width: 30px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .el-pagination__sizes {
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .el-pagination__sizes .el-input {
				margin: 0 5px;
				width: 100px;
				position: relative;
			}
	
	.main-content .el-pagination /deep/ .el-pagination__sizes .el-input .el-input__inner {
				border: 1px solid #DCDFE6;
				cursor: pointer;
				padding: 0 25px 0 8px;
				color: #606266;
				display: inline-block;
				font-size: 13px;
				line-height: 28px;
				border-radius: 3px;
				outline: 0;
				background: #FFF;
				width: 100%;
				text-align: center;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .el-pagination__sizes .el-input span.el-input__suffix {
				top: 0;
				position: absolute;
				right: 0;
				height: 100%;
			}
	
	.main-content .el-pagination /deep/ .el-pagination__sizes .el-input .el-input__suffix .el-select__caret {
				cursor: pointer;
				color: #C0C4CC;
				width: 25px;
				font-size: 18px;
				line-height: 28px;
				text-align: center;
			}
	
	.main-content .el-pagination /deep/ .el-pagination__jump {
				margin: 0 0 0 24px;
				color: #606266;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .el-pagination__jump .el-input {
				border-radius: 3px;
				padding: 0 2px;
				margin: 0 2px;
				display: inline-block;
				width: 50px;
				font-size: 18px;
				line-height: 18px;
				position: relative;
				text-align: center;
				height: 28px;
			}
	
	.main-content .el-pagination /deep/ .el-pagination__jump .el-input .el-input__inner {
				border: 1px solid #DCDFE6;
				cursor: pointer;
				padding: 0 3px;
				color: #606266;
				display: inline-block;
				font-size: 18px;
				line-height: 28px;
				border-radius: 3px;
				outline: 0;
				background: #FFF;
				width: 100%;
				text-align: center;
				height: 28px;
			}
</style>
