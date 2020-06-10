
document.addEventListener('DOMContentLoaded', function (){

    let marque_select = document.getElementById("marque");
    let categorie_select = document.getElementById("categorie");
    let modele_select = document.getElementById("modele");
    let carburant_select = document.getElementById("carburant");
    console.log(marque_select)
      marque_select.onchange = function()  {
           
          marque = marque_select.value;
          categorie = categorie_select.value;
         

          fetch('/modele/' + marque + '&' + categorie).then(function(response) {

              response.json().then(function(data) {
                
                  let optionHTML = '';
                  
                  for (let model of data.modeles) {
                      optionHTML += '<option value="' + model.id +'">' + model.modele + '</option>';
                  }

                  document.getElementById("modele").innerHTML = optionHTML;
                  
                  console.table(optionHTML)
                  

              });
              
          });
      }

       
      categorie_select.onchange = function()  {
           
           marque = marque_select.value;
           categorie = categorie_select.value;
           
           fetch('/modele/' + marque + '&' + categorie).then(function(response) {

               response.json().then(function(data) {
                 
                   var optionHTML = '';

                   for (var model of data.modeles) {
                       optionHTML += '<option value="' + model.id + '">' + model.modele + '</option>';
                   }
                 
                   document.getElementById("modele").innerHTML = optionHTML;
                   console.table(optionHTML)
                   

               })
               
           });
       }

      //  modele_select.onchange = function()  {
           
      //      marque = marque_select.value;
      //      categorie = categorie_select.value;
      //      modele = modele_select.value;
           
      //      fetch('/carburant/' + marque + '&' + categorie+ '&' + modele).then(function(response) {

      //          response.json().then(function(data) {
                 
      //              var optionHTML = '';

      //              for (var model of data.carburant) {
      //                  optionHTML += '<option value="' + model.id + '">' + model.carburant + '</option>';
      //              }
                 
      //              document.getElementById("carburant").innerHTML = optionHTML;
      //              console.table(optionHTML)
                   

      //          })
               
      //      });
      //  }

    });
  