<template>
    <div>
        <div>click {{count}} times, count is {{evenOrOdd}}</div>
        <button @click="increment">+</button>
        <button @click="decrement">-</button>
        <button @click="incrementIfOdd">increment if odd</button>
        <button @click="incrementAsync">increment async</button>
    </div>
</template>

<script>
    export default {
        name: "Counter",
        data:function () {
            return {
                count:1
            };
        },
        computed:{
            evenOrOdd:function () {
                return this.count%2===0?'偶数':'奇数';
            }
        },
        methods:{
            increment:function () {
                // console.log(this);
                this.count++;
            },
            decrement:function () {
                this.count--;
            },
            incrementIfOdd:function () {
                if(this.count%2===1){
                    this.count++;
                }
            },
            incrementAsync:function () {
                setTimeout(()=>{
                    this.count++;
                },1000);
            }
        }
    }
</script>

<style scoped>
button{
    margin-right: 10px;
}
</style>
