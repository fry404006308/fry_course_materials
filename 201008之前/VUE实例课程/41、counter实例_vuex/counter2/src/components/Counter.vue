<template>
    <div>
        <div>click {{this.$store.state.count}} times, count is {{this.$store.getters.evenOrOdd}}</div>
        <button @click="increment">+</button>
        <button @click="decrement">-</button>
        <button @click="incrementIfOdd">increment if odd</button>
        <button @click="incrementAsync">increment async</button>
        <button @click="increment3">increment 3</button>
    </div>
</template>

<script>
    export default {
        name: "Counter",
        data:function () {
            return {};
            // return {
            //     count:1
            // };
        },
        computed:{
            // evenOrOdd:function () {
            //     return this.count%2===0?'偶数':'奇数';
            // }
        },
        methods:{
            increment:function () {
                //console.log(this);
                //console.log(this);
                //console.log(this.$store.state.count);
                // this.count++;
                this.$store.dispatch('increment')
            },
            decrement:function () {
                // this.count--;
                this.$store.dispatch('decrement')
            },
            incrementIfOdd:function () {
                // if(this.count%2===1){
                //     this.count++;
                // }
                this.$store.dispatch('incrementIfOdd')
            },
            incrementAsync:function () {
                // setTimeout(()=>{
                //     this.count++;
                // },1000);
                this.$store.dispatch('incrementAsync')
            },
            increment3:function () {
                this.$store.dispatch('increment3')
            }
        }
    }
</script>

<style scoped>
button{
    margin-right: 10px;
}
</style>
