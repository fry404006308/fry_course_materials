import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    //初始化数据
    count:0
  },
  mutations: {
    INCREMENT (state,number=1) {
      state.count+=number;
    },
    DECREMENT (state) {
      state.count--
    }
  },
  actions: {
    increment ({commit}) {
      // 提交一个mutation请求
      commit('INCREMENT')
    },
    decrement ({commit}) {
      // 提交一个mutation请求
      commit('DECREMENT')
    },
    incrementIfOdd ({commit, state}) {
      if(state.count%2===1) {
        // 提交一个mutation请求
        commit('INCREMENT')
      }
    },
    incrementAsync ({commit}) {
      setTimeout(() => {
        // 提交一个mutation请求
        commit('INCREMENT')
      }, 1000)
    },
    increment3 ({commit}) {
      commit('INCREMENT',3);
    }
  },
  getters:{
    evenOrOdd (state) { // 当读取属性值时自动调用并返回属性值
      return state.count%2===0 ? '偶数' : '奇数'
    }
  },
  modules: {
  }
})
