$(function() {

  // We can attach the `fileselect` event to all file inputs on the page
  $(document).on('change', ':file', function() {
    var input = $(this),
        numFiles = input.get(0).files ? input.get(0).files.length : 1,
        label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.trigger('fileselect', [numFiles, label]);
  });

  // We can watch for our custom `fileselect` event like this
  $(document).ready( function() {
    $(':file').on('fileselect', function(event, numFiles, label) {

      var input = $(this).parents('.input-group').find(':text'),
          log = numFiles > 1 ? numFiles + ' files selected' : label;

      if(input.length) {
          input.val(log);
      } else {
          if( log ) alert(log);
      }
      });
  });

  $("#i_wd").submit(function(event)
  {
    $("#content").css("display", "none")
    $("#loading").css("display", "block")
    $("#waiting_msg").text("Please be patient. Each image takes about 20 seconds to be generated and rendered.");
  });

  $("#i_pc").submit(function(event)
  {
    $("#content").css("display", "none")
    $("#loading").css("display", "block")
    $("#waiting_msg").text("Please be patient. Each image takes about 20 seconds to be generated and rendered.");
  });

  $("#v_wd").submit(function(event)
  {
    $("#content").css("display", "none")
    $("#loading").css("display", "block")
    $("#waiting_msg").text("Please be patient. A 20 seconds-long video takes about a minute to be analysed, processed and rendered.");
  });

  $("#v_pc").submit(function(event)
  {
    $("#content").css("display", "none")
    $("#loading").css("display", "block")
    $("#waiting_msg").text("Please be patient. A 20 seconds-long video takes about a minute to be analysed, processed and rendered.");
  });



  $("a").each(function(){
              if ($(this).prop('href') == window.location.href)
              {
                $(this).removeClass('text-dark');
                $(this).addClass('text-link');
              }
          });
});

$(".text-dark").each(function(){
             $(this).prop('href').print();
           });

function preloader(){
    document.getElementById("loading").style.display = "none";
    document.getElementById("content").style.display = "block";
}//preloader
window.onload = preloader;
