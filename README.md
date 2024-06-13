Team Members:

- Loridge Anne A. Gacho
- Zandrew Peter C. Garais

Overview:
As a partial fulfillment for our CS 180 Artificial Intelligence class, we present a project where we process raw text in the form of movie reviews and classify them into two different sentiments: positive and negative. The reviews, written by critics and sourced from a Kaggle dataset, will undergo pre-processing and analysis using Multinomial Naive Bayes and Logistic Regression. Our goal is to understand how these textual reviews justify the movie ratings.

The first part of the code focuses on data preprocessing by removing null and invalid data. After that, the inconsistent movie ratings were filtered, standardized, and converted into positive and negative labels. Then, the movie review text data is tokenized and lemmatized. Stopword removal was also done, but certain context words were omitted due to their possible significance in providing the true meaning of the tokenized texts.

The second part of the code focuses on building and comparing two machine learning models: Multinomial Naive Bayes and Logistic Regression. The performance of these models areevaluated and the results will be utilized for application purposes.

Objectives:

- Pre-process and Analyze Movie Reviews: We will pre-process the textual data to make it suitable for analysis. This includes cleaning the text, tokenization, and other NLP techniques.
- Feature Extraction: Features will be derived from the textual reviews to serve as input for our predictive models.
- Sentiment Classification: Using the extracted features, we will determine whether textual reviews are positive, neutral, or negative.

Links:

- The dataset can be accessed through this link.(https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset/data)
- The relevant code, including the trained model and vectorizer can be accessed through this link. (https://drive.google.com/drive/folders/1xPOgs2WK3FyPXAl3UNN6OL21AbBr0fkm?usp=drive_link)

To test the model, run the movie_review_site.py streamlit web app found in the Google drive by running:
`streamlit run movie_review_site.py`

Alternatively, you may run the FreeRobux_Project_Application.ipynb file found in the drive.
