<template>
    <div>
        <div>click {{count}} times, count is {{evenOrOdd}}</div>
        <button @click="increment">+</button>
        <button @click="decrement">-</button>
        <button @click="incrementIfOdd">increment if odd</button>
        <button @click="incrementAsync">increment async</button>
        <button @click="increment3">increment 3</button>
<!--        <button @click="test1">test1</button>-->
    </div>
</template>

<script>
    import {mapState,mapActions,mapGetters} from 'vuex';
    export default {
        name: "Counter",
        data:function () {
            return {};
            // return {
            //     count:1
            // };
        },
        computed:{
            ...mapState(['count']), //mapState()返回值：{count(){ return this.$store.state['count']}}
            ...mapGetters(['evenOrOdd']), //mapGetters()返回值：{evenOrOdd(){ this.$store.getters['evenOrOdd'] }}
            // evenOrOdd(){
            //     return this.$store.getters.evenOrOdd;
            // },
            // count(){
            //     return this.$store.state.count;
            // }
        },
        methods:{
            ...mapActions(['increment','decrement','incrementIfOdd','incrementAsync','increment3']),
            // test1(){
            //     let map_action=mapActions(['increment','decrement','incrementIfOdd','incrementAsync','increment3']);
            //     console.log(map_action);
            //     for (let i in map_action){
            //         console.log(i);
            //         console.log(map_action[i]);
            //     }
            // }
        }
        // methods:{
        //     increment:function () {
        //         console.log(mapState);
        //         console.log(mapState(['count']));
        //         for (let i in mapState(['count'])){
        //             console.log(i);
        //             console.log(mapState(['count'])[i])
        //         }
        //         this.$store.dispatch('increment')
        //     },
        //     decrement:function () {
        //         // this.count--;
        //         this.$store.dispatch('decrement')
        //     },
        //     incrementIfOdd:function () {
        //         // if(this.count%2===1){
        //         //     this.count++;
        //         // }
        //         this.$store.dispatch('incrementIfOdd')
        //     },
        //     incrementAsync:function () {
        //         // setTimeout(()=>{
        //         //     this.count++;
        //         // },1000);
        //         this.$store.dispatch('incrementAsync')
        //     },
        //     increment3:function () {
        //         this.$store.dispatch('increment3')
        //     }
        // }
    }
</script>

<style scoped>
button{
    margin-right: 10px;
}
</style>
