import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

#Load the sentiment model
model = pickle.load(open('pickle/xgboost_model_final', 'rb'))

#Load the recommendation engine
item_final_rating = pd.read_csv('pickle/item_final_rating')

#Load the cleaned lemmatised review comments
cleaned_data = pd.read_csv('pickle/cleaned_data')
cleaned_data.dropna(inplace = True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend',methods=['GET','POST'])
def recommend():
    
    user_name = request.form['username']
    
    
    reviews_df = item_final_rating[[user_name,'name']].sort_values(by = user_name, ascending=False)[0:20]
    reviews_df = pd.DataFrame(reviews_df)
    reviews_df.columns = ['Cosine Similarity', 'name']


    merged_df = reviews_df.merge(cleaned_data, on="name", how = 'inner')

    data = merged_df[['all_reviews_lemmatised']]
    tfidf = model['transform'].transform(data['all_reviews_lemmatised'])

    predictions = model['model'].predict(tfidf)
    predictions = pd.DataFrame(predictions)

    frames = [merged_df,predictions]
    merged = pd.concat(frames,axis = 1)
    merged.columns = ['Cosine Similarity','Product Name','reviews_username','all_reviews_lemmatised','Sentiment']
    
    subset_1_df = pd.DataFrame(merged['Product Name'].value_counts())
    subset_1_df.reset_index(inplace = True)
    subset_1_df.sort_values(by = ['Product Name'], inplace = True)
    subset_1_df.columns = ['Product Name', 'Total Sum']

    subset_2_df = pd.DataFrame(merged.groupby(['Product Name'])['Sentiment'].agg('sum'))
    subset_2_df.reset_index(inplace = True)
    subset_2_df.sort_values(by = ['Product Name'], inplace = True)
    subset_2_df.columns = ['Product Name', 'Positive Sentiments']
    
    merged_df = subset_1_df.merge(subset_2_df, on="Product Name", how = 'inner')
    merged_df['Percentage of Positive reviews'] = round((merged_df['Positive Sentiments']/merged_df['Total Sum'])*100,2)
    merged_df = merged_df.sort_values(by=['Percentage of Positive reviews'], ascending = False)

        
    top_5_recommendations = merged_df[['Product Name','Percentage of Positive reviews']]
    top_5_recommendations.columns = [['Product', 'Positivity(%)']]
    top_5_recommendations.reset_index(drop=True, inplace = True)
    top_5_recommendations = top_5_recommendations.head()
    
    top_5_recommendations.reset_index(drop=True, inplace = True)


    top_5_recommendations_html = top_5_recommendations.to_html()

    return render_template('index.html',tables=[top_5_recommendations_html])

    
if __name__ == "__main__":
    app.run(debug=True)