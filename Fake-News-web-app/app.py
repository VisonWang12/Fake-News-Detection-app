from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
word_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()




HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

from flaskext.markdown import Markdown

app = Flask(__name__)
Markdown(app)


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase



def get_cleaned_data(input_data, mode='df'):
    stop = stopwords.words('english')

    input_df = ''

    if mode != 'df':
      input_df = pd.DataFrame([input_data], columns=['Article'])
    else:
      input_df = input_data

    #lowercase the text
    input_df['Article'] = input_df['Article'].str.lower()

    input_df['Article'] = input_df['Article'].apply(lambda elem: decontracted(elem))

    #remove special characters
    input_df['Article'] = input_df['Article'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))

    # remove numbers
    input_df['Article'] = input_df['Article'].apply(lambda elem: re.sub(r"\d+", "", elem))

    #remove stopwords
    input_df['Article'] = input_df['Article'].apply(lambda x: ' '.join([word.strip() for word in x.split() if word not in (stop)]))

    #stemming, changes the word to root form
    #     input_df['text'] = input_df['text'].apply(lambda words: [word_stemmer.stem(word) for word in words])

    #lemmatization, same as stemmer, but language corpus is used to fetch the root form, so resulting words make sense
    #     more description @ https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
    input_df['Article'] = input_df['Article'].apply(lambda words: (wordnet_lemmatizer.lemmatize(words)))
    #     print(input_df.head(3))

    return input_df

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extract',methods=["GET","POST"])
def extract():
    if request.method == 'POST':
        raw_text = request.form['rawtext']
        input_art = [raw_text]
        input_art = [i for i in input_art if i]
        df_input_art = pd.DataFrame(data={"Article": input_art})
        input_news_df = get_cleaned_data(df_input_art)
        MAX_SEQUENCE_LENGTH = 500
        MAX_NUM_WORDS = 10000
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(input_news_df.Article)
        tokenized_input = tokenizer.texts_to_sequences(input_news_df.Article)
        inputnews_test = pad_sequences(tokenized_input, maxlen=MAX_SEQUENCE_LENGTH)
        model = tf.keras.models.load_model("cnn_model2.h5")
        pred2 = model.predict_classes(inputnews_test)
        html = np.empty((pred2.shape), dtype=object)
        html[pred2 == 0] = 'Fake News'
        html[pred2 == 1] = 'Real News'
        html = html[0][0]

        result = HTML_WRAPPER.format(html)

    return render_template('result.html',rawtext=raw_text,result=result)


#@app.route('/previewer')
#def previewer():
    #return render_template('previewer.html')

#@app.route('/preview',methods=["GET","POST"])
#def preview():
    #if request.method == 'POST':
        #newtext = request.form['newtext']
        #result = newtext

    #return render_template('preview.html',newtext=newtext,result=result)


if __name__ == '__main__':
	app.run(debug=True)