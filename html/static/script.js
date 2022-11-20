/*
script.js

*/

// Create DOM elements

var message =  document.createElement('p');
var submit_button = document.getElementById("submit_button");


submit_button.addEventListener('click', () => {
    var model = document.getElementById("model");
    var test = document.getElementById("test");
    fetch('http://localhost:5000/',{
        method:'POST',
        body: JSON.stringify({model:model, test:test}), 
        headers:{'Content-Type':'application/json'}
    })
    .then(res=>res.json())
    .then( res => {
        const text = res.result;
        message.innerHTML = text;
    }).catch(err => {
        message.innerHTML = '<span style="color:red">Error</span><br/>'
        console.log(err);
    });
});

// Add all DOM elements to document body

document.body.appendChild(message);