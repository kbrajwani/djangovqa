$( document ).ready(function() {
    // $('#blah').hide();
    $("#blah").css("display", "none");

});
function readURL(input) {

    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            // $('#blah').show();
            $("#blah").css("display", "block");


            $('#blah')
                .attr('src', e.target.result)
                // .width(700)
                // .height(400);
        };

        reader.readAsDataURL(input.files[0]);
    }
}

$(document).on('submit', 'form', function (e) {

    var supportMsg = document.getElementById('msg');

    var data = new FormData($('form').get(0));

        console.log( "ready here " );

        console.log(data);


    $.ajax({
        url: "/results",

        type: 'POST',
        data: data,
        processData: false,
        contentType: false,

        dataType: 'json',

        success: function (data) {
            supportMsg.innerHTML = data.question;


        }

    });
    e.preventDefault();
});
