# TFIDF Text Classification

The purpose of this project is to calculate the Term Frequency-Inverse Document Frequency (TFIDF) value to perform text classifcation on an SMS dataset and Movie Review dataset. 

The resulting model can predict if new texts should be classified as Ham or Spam for the first dataset and if a movie review is positive or negative for the second dataset. 

## Datasets and Overview
Scikit-Learn and PySpark were used for this project for their feature extraction libraries, specifically containing a TFIDF vectorizer.
The first dataset, Ham-or-Spam, contains 5572 text messages and labelled as ham or spam. The second and third datasets, Movie Reviews and Amazon Reviews, contain 2000 and 10000 reviews respectively, both labelled as as positive or negative. All datasets can be found in the Github repository. An updated Movie review dataset can also be found in the link below.
http://www.cs.cornell.edu/people/pabo/movie-review-data/

## Term Frequency-Inverse Document Frequency (TFIDF)
To perform text feature extraction, the raw texts first need to be vectorized. As such, Count Vectorization is performed to count the occurences of each unique word and logs them into a Document Term Matrix (DTM). A DTM keeps track of every unique word's occurence through every document (text message).
Using the DTM, the TFIDF has the term frequency of each word. Then the inverse document frequency is calculated on each word, which diminishes the weight of terms that occur often in the document set and increases the weight of rarer terms. This is significant, since common words like 'the' and 'is' will now have less importance during feature extraction compares to less common words like 'dog' or 'blue'.

## Machine Learning Model
For this classifier, a Support Vector Machine was used, particularly Sci-kit Learn's LinearSVC (Support Vector Classifier). Linear SVC returns the best fit hyperplane which categorizes the data. This hyperplane can then be used to predict the classification for new data.

## Results

Please visit the following link for results: https://hjmok.github.io/josephmok_portfolio/#/TFE 
