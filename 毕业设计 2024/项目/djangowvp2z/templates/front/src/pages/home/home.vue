<template>
<div class="home-preview" :style='{"width":"100%","margin":"10px auto"}'>



<div class="recommend" :style='{"padding":"0 440px","margin":"0 0 10px","backgroundSize":"100% 100%","backgroundImage":"url(http://codegen.caihongy.cn/20220903/d909d026946e48b98b65af03392e57e5.png)","height":"880px"}'>
	<div v-if="false" class="idea recommendIdea" :style='{"padding":"20px","flexWrap":"wrap","background":"#efefef","justifyContent":"space-between","display":"flex"}'>
		<div class="box1" :style='{"width":"20%","background":"#fff","height":"80px"}'></div>
		<div class="box2" :style='{"width":"20%","background":"#fff","height":"80px"}'></div>
		<div class="box3" :style='{"width":"20%","background":"#fff","height":"80px"}'></div>
		<div class="box4" :style='{"width":"20%","background":"#fff","height":"80px"}'></div>
		<div class="box5" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box6" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box7" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box8" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box9" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box10" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
	</div>
	
    <div class="title" :style='{"width":"200px","padding":"30px 0 10px","margin":"10px auto","lineHeight":"54px","textAlign":"center"}'>
		<span :style='{"color":"rgba(0, 0, 0, 1)","fontSize":"30px","fontWeight":"600"}'>招聘信息推荐</span>
	</div>
	
	
	<!-- 样式一 -->
	<div class="list list1 index-pv1" :style='{"width":"100%","padding":"10px","background":"rgba(216, 216, 216, 1)","height":"650px"}'>
		<div :style='{"margin":"10px 20px","overflow":"hidden","borderRadius":"10px","background":"#f5f5f5","display":"inline-block","width":"210px","position":"relative","height":"240px"}' v-for="(item,index) in zhaopinxinxiRecommend" :key="index" @click="toDetail('zhaopinxinxiDetail', item)" class="list-item animation-box">
			<div class="name line1" :style='{"padding":"10px","overflow":"hidden","color":"rgba(0, 0, 0, 1)","lineHeight":"35px","fontSize":"16px","fontWeight":"550","height":"70px"}'>{{item.xlyq}}</div>
			<div class="name line1" :style='{"padding":"10px","overflow":"hidden","color":"rgba(0, 0, 0, 1)","lineHeight":"35px","fontSize":"16px","fontWeight":"550","height":"70px"}'>{{item.gsmc}}</div>
			<div class="name line1" :style='{"padding":"10px","overflow":"hidden","color":"rgba(0, 0, 0, 1)","lineHeight":"35px","fontSize":"16px","fontWeight":"550","height":"70px"}'>{{item.xinzi}}</div>
		</div>
	</div>
	
	
	
	
	
	
	
	
	
	<div @click="moreBtn('zhaopinxinxi')" :style='{"border":"0","padding":"5px 0","margin":"-70px auto 0","textAlign":"center","borderWidth":" 0 0 6px","display":"block","width":"110px","lineHeight":"32px","borderStyle":"solid"}'>
		<span :style='{"color":"#000","fontSize":"20px"}'>查看更多</span>
		<i v-if="true" :style='{"color":"#000","fontSize":"20px"}' class="el-icon-d-arrow-right"></i>
	</div>
	
</div>

	
<div class="news" :style='{"padding":"0 440px","margin":"0 auto 10px","borderColor":"rgba(216, 216, 216, 1)","background":"#fff","borderWidth":"0 0 2px","borderStyle":"dashed","height":"500px"}'>
	<div v-if="false" class="idea newsIdea" :style='{"padding":"20px","flexWrap":"wrap","background":"#efefef","justifyContent":"space-between","display":"flex"}'>
		<div class="box1" :style='{"width":"20%","background":"#fff","height":"80px"}'></div>
		<div class="box2" :style='{"width":"20%","background":"#fff","height":"80px"}'></div>
		<div class="box3" :style='{"width":"20%","background":"#fff","height":"80px"}'></div>
		<div class="box4" :style='{"width":"20%","background":"#fff","height":"80px"}'></div>
		<div class="box5" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box6" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box7" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box8" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box9" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
		<div class="box10" :style='{"width":"20%","background":"#fff","display":"none","height":"80px"}'></div>
	</div>
	
	<div class="title" :style='{"width":"200px","margin":"20px 0 0","borderColor":"#000","lineHeight":"54px","borderStyle":"solid","borderWidth":"0 0 10px"}'>
		<span :style='{"color":"rgba(0, 0, 0, 1)","fontSize":"30px","fontWeight":"600"}'>新闻资讯</span>
	</div>
	
	
	
	
	
	
	
	
	
	
	
	<!-- 样式十 -->
	<div v-if="newsList.length" class="list list10 index-pv1" :style='{"padding":"10px 0","borderColor":"rgba(216, 216, 216, 1)","flexDirection":"row-reverse","borderWidth":"5px 0 0","display":"flex","width":"100%","borderStyle":"solid","height":"auto"}'>
	  <div v-if="newsList.length>0" @click="toDetail('newsDetail', newsList[0])" :style='{"width":"520px","padding":"10px","margin":"50px 10px 0","position":"relative","background":"rgba(216, 216, 216, 1)","height":"220px"}' class="new10-list-item animation-box">
	    <img :style='{"width":"200px","objectFit":"cover","display":"block","height":"200px"}' :src="baseUrl + newsList[0].picture" alt="">
		<div :style='{"padding":"0 10px","whiteSpace":"nowrap","overflow":"hidden","color":"rgba(0, 0, 0, 1)","top":"10px","left":"210px","width":"300px","fontSize":"18px","lineHeight":"28px","position":"absolute","textOverflow":"ellipsis"}' class="new9-list-item-title line1">{{newsList[0].title}}</div>
		<div :style='{"padding":"0 10px","color":"#fff","top":"195px","background":"rgba(0, 0, 0, 1)","fontSize":"12px","lineHeight":"24px","position":"absolute","right":"0"}' class="new9-list-item-time">{{newsList[0].addtime.split(' ')[0]}}</div>
	    <div :style='{"padding":"0 10px","overflow":"hidden","color":"rgba(158, 158, 158, 1)","top":"45px","left":"210px","fontSize":"14px","lineHeight":"24px","position":"absolute","height":"150px"}' class="new9-list-item-descript line2">{{newsList[0].introduction}}</div>
		<div :style='{"padding":"0 10px","margin":"0 10px 10px","color":"#999","background":"#fff","display":"inline-block","fontSize":"12px","lineHeight":"24px"}' class="new9-list-item-identification"></div>
	  </div>
	  <div v-if="newsList.length>1" :style='{"margin":"0 10px","flexDirection":"column","display":"flex","width":"500px","position":"relative","justifyContent":"space-between","height":"auto"}'>
	    <div v-for="(item,index) in newsList" v-if="index<4 && index>0" :key="index" @click="toDetail('newsDetail', item)" :style='{"borderColor":"rgba(216, 216, 216, 1)","background":"#fff","borderWidth":"0 0 1px","width":"100%","position":"relative","borderStyle":"dashed","height":"110px"}' class="new10-list-item animation-box">
	        <div :style='{"padding":"0 10px","overflow":"hidden","whiteSpace":"nowrap","color":"rgba(158, 158, 158, 1)","width":"80%","fontSize":"18px","lineHeight":"28px","textOverflow":"ellipsis"}' class="new9-list-item-title line1">{{ item.title }}</div>
	        <div :style='{"color":"#fff","top":"0","textAlign":"center","background":"rgba(0, 0, 0, 1)","width":"90px","fontSize":"14px","lineHeight":"30px","position":"absolute","right":"10px"}' class="new9-list-item-time">{{ item.addtime.split(' ')[0] }}</div>
			<div :style='{"padding":"0 10px","margin":"15px 0 0","overflow":"hidden","color":"#666","fontSize":"14px","lineHeight":"24px","height":"50px"}' class="new9-list-item-descript line2">{{ item.introduction }}</div>
			<div :style='{"padding":"0 10px","fontSize":"12px","lineHeight":"24px","color":"#999","background":"#fff","display":"inline-block"}' class="new9-list-item-identification"></div>
	    </div>
	  </div>
	</div>
	
	<div @click="moreBtn('news')" :style='{"border":"0","margin":"-398px 0 0 913px","borderColor":"rgba(0, 0, 0, 1)","textAlign":"center","borderWidth":"0 0 10px","display":"block","width":"110px","lineHeight":"32px","borderStyle":"solid"}'>
		<span :style='{"color":"rgba(0, 0, 0, 1)","fontSize":"20px"}'>查看更多</span>
		<i v-if="true" :style='{"color":"#000","fontSize":"20px"}' class="el-icon-d-arrow-right"></i>
	</div>
	
</div>




</div>
</template>

<script>
  export default {
    //数据集合
    data() {
      return {
        baseUrl: '',
        systemIntroductionDetail: {},
        newsList: [],
        zhaopinxinxiRecommend: [],

      }
    },
    created() {
      this.baseUrl = this.$config.baseUrl;
      this.getNewsList();
      this.getSystemIntroduction();
      this.getList();
    },
    //方法集合
    methods: {
      preHttp(str) {
          return str && str.substr(0,4)=='http';
      },
      getSystemIntroduction() {
          this.$http.get('systemintro/detail/1', {}).then(res => {
            if(res.data.code == 0) {
              this.systemIntroductionDetail = res.data.data;
            }
          })
      },
		getNewsList() {
			this.$http.get('news/list', {params: {
				page: 1,
				limit: 6,
			order: 'desc'}}).then(res => {
				if (res.data.code == 0) {
					this.newsList = res.data.data.list;
					
					
				}
			});
		},
		getList() {
          let autoSortUrl = "";
          autoSortUrl = "zhaopinxinxi/autoSort";
			this.$http.get(autoSortUrl, {params: {
				page: 1,
				limit: 8,
			}}).then(res => {
				if (res.data.code == 0) {
					this.zhaopinxinxiRecommend = res.data.data.list;
					
					
					// 商品列表样式五
					
				}
			});
			
		},
		toDetail(path, item) {
			this.$router.push({path: '/index/' + path, query: {detailObj: JSON.stringify(item)}});
		},
		moreBtn(path) {
			this.$router.push({path: '/index/' + path});
		}
    }
  }
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
	.home-preview {
	
		.recommend {
			.list3 .swiper-button-prev {
				left: 10px;
				right: auto;
			}
			
			.list3 .swiper-button-prev::after {
				color: rgb(64, 158, 255);
			}
			
			.list3 .swiper-button-next {
				left: auto;
				right: 10px;
			}
			
			.list3 .swiper-button-next::after {
				color: rgb(64, 158, 255);
			}
			
			.list5 .swiper-button-prev {
				left: 10px;
				right: auto;
			}
			
			.list5 .swiper-button-prev::after {
				color: rgb(64, 158, 255);
        }
        
        .list5 .swiper-button-next {
				left: auto;
				right: 10px;
			}
			
			.list5 .swiper-button-next::after {
				color: rgb(64, 158, 255);
			}
			
			.list5 {
				.swiper-slide-prev {
					position: relative;
					z-index: 3;
				}
		
				.swiper-slide-next {
					position: relative;
					z-index: 3;
				}
		
				.swiper-slide-active {
					position: relative;
					z-index: 5;
				}
			}
			
			.index-pv1 .animation-box {
				transform: rotate(0deg) scale(1) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
				z-index: initial;
			}
			
			.index-pv1 .animation-box:hover {
				transform: scale(1.05);
				-webkit-perspective: 1000px;
				perspective: 1000px;
				transition: 0.3s;
				z-index: 1;
			}
			
			.index-pv1 .animation-box img {
				transform: rotate(0deg) scale(1) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
			}
			
			.index-pv1 .animation-box img:hover {
				-webkit-perspective: 1000px;
				perspective: 1000px;
				transition: 0.3s;
			}
		}
		
		.news {
			.list3 .swiper-button-prev {
				left: 10px;
				right: auto;
			}
			
			.list3 .swiper-button-prev::after {
				color: rgb(64, 158, 255);
			}
			
			.list3 .swiper-button-next {
				left: auto;
				right: 10px;
			}
			
			.list3 .swiper-button-next::after {
				color: rgb(64, 158, 255);
			}
			
			.list6 .swiper-button-prev {
				left: 10px;
				right: auto;
			}
			
			.list6 .swiper-button-prev::after {
				color: rgb(64, 158, 255);
			}
			
			.list6 .swiper-button-next {
				left: auto;
				right: 10px;
			}
			
			.list6 .swiper-button-next::after {
				color: rgb(64, 158, 255);
			}
			
			.index-pv1 .animation-box {
				transform: rotate(0deg) scale(1) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
				z-index: initial;
			}
			
			.index-pv1 .animation-box:hover {
				transform: scale(1.02) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
				-webkit-perspective: 1000px;
				perspective: 1000px;
				transition: 0.3s;
				z-index: 1;
			}
			
			.index-pv1 .animation-box img {
				transform: rotate(0deg) scale(1) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
			}
			
			.index-pv1 .animation-box img:hover {
				-webkit-perspective: 1000px;
				perspective: 1000px;
				transition: 0.3s;
			}
		}
	
		.lists {
			.list3 .swiper-button-prev {
				left: 10px;
				right: auto;
			}
			
			.list3 .swiper-button-prev::after {
				color: rgb(64, 158, 255);
			}
			
			.list3 .swiper-button-next {
				left: auto;
				right: 10px;
        }
        
        .list3 .swiper-button-next::after {
				color: rgb(64, 158, 255);
			}
			
			.list5 .swiper-button-prev {
				left: 10px;
				right: auto;
			}
			
			.list5 .swiper-button-prev::after {
				color: rgb(64, 158, 255);
			}
			
			.list5 .swiper-button-next {
            left: auto;
            right: 10px;
			}
			
			.list5 .swiper-button-next::after {
				color: rgb(64, 158, 255);
			}
			
			.list5 {
				.swiper-slide-prev {
					position: relative;
					z-index: 3;
				}
		
				.swiper-slide-next {
					position: relative;
					z-index: 3;
				}
		
				.swiper-slide-active {
					position: relative;
					z-index: 5;
				}
			}
			
			.index-pv1 .animation-box {
				transform: rotate(0deg) scale(1) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
				z-index: initial;
			}
			
			.index-pv1 .animation-box:hover {
				transform: scale(1.03) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
				-webkit-perspective: 1000px;
				perspective: 1000px;
				transition: 0.3s;
				z-index: 1;
			}
			
			.index-pv1 .animation-box img {
				transform: rotate(0deg) scale(1) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
			}
			
			.index-pv1 .animation-box img:hover {
				-webkit-perspective: 1000px;
				perspective: 1000px;
				transition: 0.3s;
			}
		}
	}
</style>
