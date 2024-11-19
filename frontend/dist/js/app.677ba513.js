(function(){"use strict";var e={7973:function(e,t,o){var s=o(5130),r=o(6768),n=o(4232);const a={id:"app"},i={class:"image-container"},l=["onClick"],c=["src"],d={class:"categories"},u=["onClick"],h={key:0,class:"user-prompt"},g={class:"actions"},m={class:"model-info",style:{"margin-top":"10px","text-align":"center"}},p={key:0};function f(e,t,o,s,f,w){return(0,r.uX)(),(0,r.CE)("div",a,[t[3]||(t[3]=(0,r.Lk)("h1",null,"HITL Animal Classification Interface",-1)),(0,r.Lk)("div",i,[((0,r.uX)(!0),(0,r.CE)(r.FK,null,(0,r.pI)(f.images,((e,t)=>((0,r.uX)(),(0,r.CE)("div",{key:t,onClick:e=>w.userSelectsCorrectImage(t),class:(0,n.C4)({clickable:null!==f.guessedIndex})},[(0,r.Lk)("img",{src:e.url,class:(0,n.C4)({guessed:f.guessedIndex===t})},null,10,c)],10,l)))),128))]),(0,r.Lk)("div",d,[((0,r.uX)(!0),(0,r.CE)(r.FK,null,(0,r.pI)(f.categories,((e,t)=>((0,r.uX)(),(0,r.CE)("button",{key:t,class:(0,n.C4)(["category",{selected:f.selectedCategoryIndex===t}]),onClick:e=>w.selectCategory(t)},(0,n.v_)(e),11,u)))),128))]),f.showUserPrompt?((0,r.uX)(),(0,r.CE)("div",h," Please select the image you had in mind. ")):(0,r.Q3)("",!0),(0,r.Lk)("div",g,[(0,r.Lk)("button",{class:"save-model",onClick:t[0]||(t[0]=(...e)=>w.saveModelWeights&&w.saveModelWeights(...e))}," Save Model Weights "),(0,r.Lk)("button",{class:"reset-model",onClick:t[1]||(t[1]=(...e)=>w.resetModel&&w.resetModel(...e))},"Reset Model"),(0,r.Lk)("button",{class:"use-trained-model",onClick:t[2]||(t[2]=(...e)=>w.useTrainedModel&&w.useTrainedModel(...e))}," Use Trained Model ")]),(0,r.Lk)("div",m,[null!==f.guessedIndex?((0,r.uX)(),(0,r.CE)("p",p," Model guessed with "+(0,n.v_)(f.guessedCertaintyFormatted)+" certainty. ",1)):(0,r.Q3)("",!0)])])}o(4114),o(7642),o(8004),o(3853),o(5876),o(2475),o(5024),o(1698),o(8992),o(3949),o(1454);const w={NODE_ENV:"production",BASE_URL:"/"}.VUE_APP_BACKEND_URL_PRODUCTION;console.log("Using backend URL:",w);var v={data(){return{showUserPrompt:!1,images:[],categories:["Mammal","Bird","Reptile","Fish","Amphibian","Insect","Invertebrate"],iterationCount:0,maxIterations:50,guessedIndex:null,allowUserSelection:!1,availableImages:[],guessedCertainty:0,selectedCategoryIndex:null,guessedCertaintyFormatted:"0%"}},mounted(){this.loadZooDataset()},methods:{resetModel(){fetch(`${w}/reset_model`,{method:"POST",headers:{"Content-Type":"application/json"}}).then((e=>{if(!e.ok)throw new Error("Network response was not ok");return e.json()})).then((e=>{console.log("Received response from /reset_model:",e),alert(e.message),this.iterationCount=0,this.loadNewImages()})).catch((e=>{console.error("Error during /reset_model request:",e)}))},useTrainedModel(){fetch(`${w}/use_trained_model`,{method:"POST",headers:{"Content-Type":"application/json"}}).then((e=>{if(!e.ok)throw new Error("Network response was not ok");return e.json()})).then((e=>{console.log("Received response from /use_trained_model:",e),alert(e.message),this.iterationCount=0,this.loadNewImages()})).catch((e=>{console.error("Error during /use_trained_model request:",e)}))},userSelectsCorrectImage(e){if(this.showUserPrompt&&(this.showUserPrompt=!1),this.allowUserSelection){console.log("User selected the correct image:",this.images[e]);const t=this.images[e],o=this.categories[this.selectedCategoryIndex],s={correct_image:{name:t.name,attributes:t.attributes,category:o}};console.log("Sending feedback to backend:",JSON.stringify(s)),fetch(`${w}/train`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(s)}).then((e=>{if(!e.ok)throw new Error("Network response was not ok");return e.json()})).then((e=>{console.log("Received response from /train:",e),this.allowUserSelection=!1,this.loadNewImages()})).catch((e=>{console.error("Error during /train request:",e)}))}},loadZooDataset(){fetch("/zoo_dataset.json").then((e=>{if(!e.ok)throw new Error("Failed to fetch zoo dataset");return e.json()})).then((e=>{console.log("Received response from /zoo_dataset:",e),this.availableImages=e.map((e=>({name:e.animal_name,type:e.type,url:`/animal_images/${e.animal_name}.jpg`,attributes:Object.values(e).slice(1,-1)}))),this.loadNewImages()})).catch((e=>{console.error("Error loading zoo dataset:",e)}))},selectCategory(e){if(this.selectedCategoryIndex=e,this.iterationCount<this.maxIterations){const e=this.images.map((e=>({name:e.name,attributes:e.attributes})));fetch(`${w}/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({selected_images:e,selected_category:this.categories[this.selectedCategoryIndex]})}).then((e=>{if(!e.ok)throw new Error("Network response was not ok");return e.json()})).then((e=>{console.log("Received response from /predict:",e),this.iterationCount++,this.guessedIndex=e.guessed_index,this.allowUserSelection=!0,setTimeout((()=>{this.showUserPrompt=!0}),1e3),this.guessedCertainty=e.certainty_percentage,this.guessedCertaintyFormatted=`${this.guessedCertainty.toFixed(2)}%`})).catch((e=>{console.error("Error during /predict request:",e)}))}else alert("Maximum number of iterations reached.")},loadNewImages(){this.showUserPrompt=!1,this.selectedCategoryIndex=null;const e=[],t=new Set;while(e.length<3&&this.availableImages.length>0){const o=Math.floor(Math.random()*this.availableImages.length),s=this.availableImages[o];t.has(s.type)||(e.push(s),t.add(s.type))}this.images=e,this.guessedIndex=null,this.images.forEach((e=>{console.log("Loaded image:",e.name,"| Type:",this.categories[e.type-1])}))},saveModelWeights(){const e=(new Date).toISOString().replace(/[:.]/g,"-"),t=`/weights/model_weights_${e}.pth`;fetch(`${w}/save_model_weights?weights_path=${t}`,{method:"POST",headers:{"Content-Type":"application/json"}}).then((e=>{if(!e.ok)throw new Error("Network response was not ok");return e.json()})).then((e=>{console.log("Received response from /save_model_weights:",e),alert(e.message),this.iterationCount=0,this.loadNewImages()})).catch((e=>{console.error("Error during /save_model_weights request:",e)}))}}},y=o(1241);const C=(0,y.A)(v,[["render",f],["__scopeId","data-v-3cf97937"]]);var k=C;(0,s.Ef)(k).mount("#app")}},t={};function o(s){var r=t[s];if(void 0!==r)return r.exports;var n=t[s]={exports:{}};return e[s].call(n.exports,n,n.exports,o),n.exports}o.m=e,function(){var e=[];o.O=function(t,s,r,n){if(!s){var a=1/0;for(d=0;d<e.length;d++){s=e[d][0],r=e[d][1],n=e[d][2];for(var i=!0,l=0;l<s.length;l++)(!1&n||a>=n)&&Object.keys(o.O).every((function(e){return o.O[e](s[l])}))?s.splice(l--,1):(i=!1,n<a&&(a=n));if(i){e.splice(d--,1);var c=r();void 0!==c&&(t=c)}}return t}n=n||0;for(var d=e.length;d>0&&e[d-1][2]>n;d--)e[d]=e[d-1];e[d]=[s,r,n]}}(),function(){o.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return o.d(t,{a:t}),t}}(),function(){o.d=function(e,t){for(var s in t)o.o(t,s)&&!o.o(e,s)&&Object.defineProperty(e,s,{enumerable:!0,get:t[s]})}}(),function(){o.g=function(){if("object"===typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"===typeof window)return window}}()}(),function(){o.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)}}(),function(){var e={524:0};o.O.j=function(t){return 0===e[t]};var t=function(t,s){var r,n,a=s[0],i=s[1],l=s[2],c=0;if(a.some((function(t){return 0!==e[t]}))){for(r in i)o.o(i,r)&&(o.m[r]=i[r]);if(l)var d=l(o)}for(t&&t(s);c<a.length;c++)n=a[c],o.o(e,n)&&e[n]&&e[n][0](),e[n]=0;return o.O(d)},s=self["webpackChunkanimal_classification"]=self["webpackChunkanimal_classification"]||[];s.forEach(t.bind(null,0)),s.push=t.bind(null,s.push.bind(s))}();var s=o.O(void 0,[504],(function(){return o(7973)}));s=o.O(s)})();
//# sourceMappingURL=app.677ba513.js.map