const base = {
    get() {
        return {
            url : "http://localhost:8080/djangowvp2z/",
            name: "djangowvp2z",
            // 退出到首页链接
            indexUrl: 'http://localhost:8080/front/dist/index.html'
        };
    },
    getProjectName(){
        return {
            projectName: "基于协同过滤的互联网岗位推荐系统"
        } 
    }
}
export default base
