// Invoke the code tab to be chosen by default on page load
  $(document).ready(function() {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    tablinks = document.getElementsByClassName("tablinks");

    var pageNumbersDict = {
      "#2" : 2, 
      "#4" : 4,
      "#5" : 5,
      "#6" : 6,
      "#7" : 7,
      "#8" : 8,
      "#9" : 9,
      "#10": 10,
      "#11": 11
    };

    const pageNumbers = Object.keys(pageNumbersDict);
    var urlPageNumber = location.hash;
    console.log(urlPageNumber);

    if (pageNumbers.includes(urlPageNumber) && window.location.href.indexOf(urlPageNumber) != -1){
      i = pageNumbersDict[urlPageNumber];
      tabcontent[i].style.display = "block";
      tablinks[i].className = tablinks[i].className + " active";

    }
  });