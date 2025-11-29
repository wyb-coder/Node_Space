<template>
<div class="content">
	<div class="text" :style='{"margin":"50px auto","fontSize":"32px","color":"rgb(51, 51, 51)","textAlign":"center","fontWeight":"bold"}'>欢迎使用 {{this.$project.projectName}}</div>
    <div class="cardView">
        <div class="cards" :style='{"margin":"0 0 20px 0","alignItems":"center","justifyContent":"center","display":"flex"}'>
			<div :style='{"boxShadow":"0 1px 6px rgba(0,0,0,.3)","margin":"0 10px","borderRadius":"4px","display":"flex"}' v-if="isAuth('zhaopinxinxi','首页总数')">
				<div :style='{"width":"80px","background":"#93C7B3","height":"80px"}'></div>
				<div :style='{"width":"160px","alignItems":"center","flexDirection":"column","justifyContent":"center","display":"flex"}'>
					<div :style='{"margin":"5px 0","lineHeight":"24px","fontSize":"20px","color":"#333","fontWeight":"bold","height":"24px"}'>{{zhaopinxinxiCount}}</div>
					<div :style='{"margin":"5px 0","lineHeight":"24px","fontSize":"16px","color":"#666","height":"24px"}'>招聘信息总数</div>
				</div>
			</div>
        </div>
        <div style="display: flex;align-items: center;width: 100%;margin-bottom: 10px;">
            <el-card style="width: 20%;margin: 0 10px;" v-if="isAuth('zhaopinxinxi','首页统计')">
                <div id="zhaopinxinxiChart1" style="width:100%;height:400px;"></div>
            </el-card>
            <el-card style="width: 20%;margin: 0 10px;" v-if="isAuth('zhaopinxinxi','首页统计')">
                <div id="zhaopinxinxiChart2" style="width:100%;height:400px;"></div>
            </el-card>
            <el-card style="width: 20%;margin: 0 10px;" v-if="isAuth('zhaopinxinxi','首页统计')">
                <div id="zhaopinxinxiChart3" style="width:100%;height:400px;"></div>
            </el-card>
            <el-card style="width: 20%;margin: 0 10px;" v-if="isAuth('zhaopinxinxi','首页统计')">
                <div id="zhaopinxinxiChart4" style="width:100%;height:400px;"></div>
            </el-card>
            <el-card style="width: 20%;margin: 0 10px;" v-if="isAuth('zhaopinxinxi','首页统计')">
                <div id="zhaopinxinxiChart5" style="width:100%;height:400px;"></div>
            </el-card>
        </div>
    </div>
</div>
</template>
<script>
//5
import router from '@/router/router-static'
import * as echarts from 'echarts'
export default {
	data() {
		return {
            zhaopinxinxiCount: 0,
		};
	},
  mounted(){
    this.init();
    this.getzhaopinxinxiCount();
    this.zhaopinxinxiChat1();
    this.zhaopinxinxiChat2();
    this.zhaopinxinxiChat3();
    this.zhaopinxinxiChat4();
    this.zhaopinxinxiChat5();
  },
  methods:{
    init(){
        if(this.$storage.get('Token')){
        this.$http({
            url: `${this.$storage.get('sessionTable')}/session`,
            method: "get"
        }).then(({ data }) => {
            if (data && data.code != 0) {
            router.push({ name: 'login' })
            }
        });
        }else{
            router.push({ name: 'login' })
        }
    },
    getzhaopinxinxiCount() {
        this.$http({
            url: `zhaopinxinxi/count`,
            method: "get"
        }).then(({
            data
        }) => {
            if (data && data.code == 0) {
                this.zhaopinxinxiCount = data.data
            }
        })
    },

    zhaopinxinxiChat1() {
      this.$nextTick(()=>{

        var zhaopinxinxiChart1 = echarts.init(document.getElementById("zhaopinxinxiChart1"),'green');
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
                zhaopinxinxiChart1.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    zhaopinxinxiChart1.resize();
                };
            }
        });
      })
    },

    zhaopinxinxiChat2() {
      this.$nextTick(()=>{

        var zhaopinxinxiChart2 = echarts.init(document.getElementById("zhaopinxinxiChart2"),'green');
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
                zhaopinxinxiChart2.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    zhaopinxinxiChart2.resize();
                };
            }
        });
      })
    },

    zhaopinxinxiChat3() {
      this.$nextTick(()=>{

        var zhaopinxinxiChart3 = echarts.init(document.getElementById("zhaopinxinxiChart3"),'green');
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
                zhaopinxinxiChart3.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    zhaopinxinxiChart3.resize();
                };
            }
        });
      })
    },


    zhaopinxinxiChat4() {
      this.$nextTick(()=>{

        var zhaopinxinxiChart4 = echarts.init(document.getElementById("zhaopinxinxiChart4"),'green');
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
                    }]
                };
                // 使用刚指定的配置项和数据显示图表。
                zhaopinxinxiChart4.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    zhaopinxinxiChart4.resize();
                };
            }
        });
      })
    },

    zhaopinxinxiChat5() {
      this.$nextTick(()=>{

        var zhaopinxinxiChart5 = echarts.init(document.getElementById("zhaopinxinxiChart5"),'green');
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
                zhaopinxinxiChart5.setOption(option);
                  //根据窗口的大小变动图表
                window.onresize = function() {
                    zhaopinxinxiChart5.resize();
                };
            }
        });
      })
    },

  }
};
</script>
<style lang="scss" scoped>
    .cardView {
        display: flex;
        flex-wrap: wrap;
        width: 100%;

        .cards {
            display: flex;
            align-items: center;
            width: 100%;
            margin-bottom: 10px;
            justify-content: center;
            .card {
                width: calc(25% - 20px);
                margin: 0 10px;
                /deep/.el-card__body{
                    padding: 0;
                }
            }
        }
    }
</style>
