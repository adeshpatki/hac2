# Core pkgs
# where all the features will be inside
import streamlit.components.v1 as components
import sweetviz as sv
from streamlit_pandas_profiling import st_profile_report
import streamlit as st

# EDA pkgs
import pandas as pd
import codecs  # will help to load our files
from pandas_profiling import ProfileReport

# natural language processing
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import string
import nltk
nltk.download('stopwords')
ps = PorterStemmer()

# component pkgs


def st_display_sweetviz(report_html, width=1000, height=500):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    components.html(page, width=width, height=height, scrolling=True)


def main():
    """A simple EDA app with streamlit components"""
    menu = ["Home", "Product's Analysis",
            "Pandas Profile", "Sweetviz", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Pandas Profile":
        st.subheader("Automated EDA with Pandas Profile")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if data_file is not None:
            df = pd.read_csv(data_file, encoding="unicode_escape")
            st.dataframe(df.head())
            profile = ProfileReport(df)
            st_profile_report(profile)

    elif choice == "Product's Analysis":
        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)
            y = []
            for i in text:
                if i.isalnum():
                    y.append(i)
            text = y[:]
            y.clear()
            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(i)
            text = y[:]
            y.clear()
            for i in text:
                y.append(ps.stem(i))

            return " ".join(y)

        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))

        st.title("Product's Sentiment Analysis")
        inputs = st.text_input("Enter Product's Name")

        if st.button('Output'):
            transformed = transform_text(inputs)
            # vectorizing

            vec = tfidf.transform([transformed])

            result = model.predict(vec)[0]
            if result == 1:
                st.header("Positive")
            else:
                st.header("Negative")

    elif choice == "Sweetviz":
        st.subheader("Automated EDA with Sweetviz")
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if data_file is not None:
            df = pd.read_csv(data_file, encoding="unicode_escape")
            st.dataframe(df.head())
            if st.button("Generate Sweetviz Report"):
               # Normal workflow
                report = sv.analyze(df)
                report.show_html()
                st_display_sweetviz("SWEETVIZ_REPORT.html")

    elif choice == "About":
        st.subheader("About APP")
        # components.iframe("https://google.com")

    else:
        st.subheader("Home")
        #components.html("<p style='color:red;'> Streamlit components is awsome </p>")
        components.html(""""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {box-sizing: border-box}
body {font-family: Verdana, sans-serif; margin:0}
.mySlides {display: none}
img {vertical-align: middle;}

/* Slideshow container */
.slideshow-container {
  max-width: 1000px;
  position: relative;
  margin: auto;
}

/* Next & previous buttons */
.prev, .next {
  cursor: pointer;
  position: absolute;
  top: 50%;
  width: auto;
  padding: 16px;
  margin-top: -22px;
  color: white;
  font-weight: bold;
  font-size: 18px;
  transition: 0.6s ease;
  border-radius: 0 3px 3px 0;
  user-select: none;
}

/* Position the "next button" to the right */
.next {
  right: 0;
  border-radius: 3px 0 0 3px;
}

/* On hover, add a black background color with a little bit see-through */
.prev:hover, .next:hover {
  background-color: rgba(0,0,0,0.8);
}

/* Caption text */
.text {
  color: #f2f2f2;
  font-size: 15px;
  padding: 8px 12px;
  position: absolute;
  bottom: 8px;
  width: 100%;
  text-align: center;
}

/* Number text (1/3 etc) */
.numbertext {
  color: #f2f2f2;
  font-size: 12px;
  padding: 8px 12px;
  position: absolute;
  top: 0;
}

/* The dots/bullets/indicators */
.dot {
  cursor: pointer;
  height: 15px;
  width: 15px;
  margin: 0 2px;
  background-color: #bbb;
  border-radius: 50%;
  display: inline-block;
  transition: background-color 0.6s ease;
}

.active, .dot:hover {
  background-color: #717171;
}

/* Fading animation */
.fade {
  -webkit-animation-name: fade;
  -webkit-animation-duration: 1.5s;
  animation-name: fade;
  animation-duration: 1.5s;
}

@-webkit-keyframes fade {
  from {opacity: .4} 
  to {opacity: 1}
}

@keyframes fade {
  from {opacity: .4} 
  to {opacity: 1}
}

/* On smaller screens, decrease text size */
@media only screen and (max-width: 300px) {
  .prev, .next,.text {font-size: 11px}
}
</style>
</head>
<body>

<div class="slideshow-container">

<div class="mySlides fade">
  <div class="numbertext">1 / 3</div>
  <img src="https://www.w3schools.com/howto/img_nature_wide.jpg" style="width:100%">
  <div class="text">Caption Text</div>
</div>

<div class="mySlides fade">
  <div class="numbertext">2 / 3</div>
  <img src="https://www.w3schools.com/howto/img_snow_wide.jpg" style="width:100%">
  <div class="text">Caption Two</div>
</div>

<div class="mySlides fade">
  <div class="numbertext">3 / 3</div>
  <img src="https://www.w3schools.com/howto/img_mountains_wide.jpg" style="width:100%">
  <div class="text">Caption Three</div>
</div>

<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
<a class="next" onclick="plusSlides(1)">&#10095;</a>

</div>
<br>

<div style="text-align:center">
  <span class="dot" onclick="currentSlide(1)"></span> 
  <span class="dot" onclick="currentSlide(2)"></span> 
  <span class="dot" onclick="currentSlide(3)"></span> 
</div>

<script>
var slideIndex = 1;
showSlides(slideIndex);

function plusSlides(n) {
  showSlides(slideIndex += n);
}

function currentSlide(n) {
  showSlides(slideIndex = n);
}

function showSlides(n) {
  var i;
  var slides = document.getElementsByClassName("mySlides");
  var dots = document.getElementsByClassName("dot");
  if (n > slides.length) {slideIndex = 1}    
  if (n < 1) {slideIndex = slides.length}
  for (i = 0; i < slides.length; i++) {
      slides[i].style.display = "none";  
  }
  for (i = 0; i < dots.length; i++) {
      dots[i].className = dots[i].className.replace(" active", "");
  }
  slides[slideIndex-1].style.display = "block";  
  dots[slideIndex-1].className += " active";
}
</script>
</body>
</html> 

""")


if __name__ == '__main__':
    main()
