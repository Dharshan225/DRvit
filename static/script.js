const input=document.getElementById("imageInput");

const preview=document.getElementById("preview");

input.addEventListener("change",function(){

    const file=this.files[0];

    if(file){

        preview.style.display="block";

        preview.src=URL.createObjectURL(file);

    }

});