# stock_app.py
from flask import Flask, render_template, request, session, send_file, redirect, url_for
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import io
import csv
from datetime import datetime
import secrets
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from textblob import TextBlob
from mykeys import news_key #api keys - make this yourself. class structure in keys.py
import requests


app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data = data.asfreq('B')  # Ensure data has business day frequency
    return data

def analyze_and_predict(data, days=7, sentiment_score=0):
    if data.empty:
        raise ValueError("No data found for analysis and prediction.")

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    if data.index.freq is None:
        data = data.asfreq('B')

    try:
        model = ARIMA(data['Close'], order=(5, 1, 0))  # ARIMA(p,d,q) parameters
        model_fit = model.fit()

        forecast = model_fit.get_forecast(steps=days)
        prediction = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Adjust predictions based on sentiment score
        adjusted_prediction = prediction * (1 + sentiment_score * 0.1)  # Example adjustment

        return prediction, conf_int, adjusted_prediction

    except Exception as e:
        print(f"An error occurred during the ARIMA model fitting or prediction: {e}")
        return None, None, None
    
def get_news_sentiment(ticker):
    api_url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={news_key.key}"
    response = requests.get(api_url)
    articles = response.json()['articles']
    
    sentiment_scores = []
    
    for article in articles:
        description = article.get('description')
        if description:
            analysis = TextBlob(article['description'])
            sentiment_scores.append(analysis.sentiment.polarity)
        
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

def plot_stock_data_and_predictions(data, predictions, conf_int, prediction_days, adjusted_predictions):
    # Combine historical data and predictions for the plot
    last_date = data.index[-1]
    prediction_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=prediction_days, freq='B')
    
    # Creating DataFrame for plotting
    df_pred = pd.DataFrame({'Date': prediction_dates, 'Predicted Price': predictions, 'Adjusted Predicted Price': adjusted_predictions,
                            'Lower Bound': conf_int.iloc[:, 0], 'Upper Bound': conf_int.iloc[:, 1]})
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=data.index, y='Close', label='Historical Prices', color='blue')
    sns.lineplot(data=df_pred, x='Date', y='Predicted Price', label='Predicted Prices', color='green')
    sns.lineplot(data=df_pred, x='Date', y='Adjusted Predicted Price', label='Sentiment Adjusted Prices', color='orange')
    plt.fill_between(df_pred['Date'], df_pred['Lower Bound'], df_pred['Upper Bound'], color='lightgreen', alpha=0.3)
    plt.title('Stock Prices with Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'queries' not in session:
        session['queries'] = []

    context = {}
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        prediction_days = int(request.form['prediction_days'])
        try:
            data = get_stock_data(ticker, start_date, end_date)
            
            # Get news sentiment
            sentiment_score = get_news_sentiment(ticker)
            
            prediction, conf_int, adjusted_prediction = analyze_and_predict(data, days=prediction_days, sentiment_score=sentiment_score)

            historical_data = data.tail().to_html(classes='table table-striped')

            predictions = [
                {
                    'day': i + 1,
                    'predicted': prediction[i],
                    'adjusted_predicted': adjusted_prediction[i],
                    'lower': conf_int.iloc[i, 0],
                    'upper': conf_int.iloc[i, 1]
                } for i in range(len(prediction))
            ]
            query = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'prediction_days': prediction_days,
                'predictions': predictions
            }
            session['queries'].append(query)
            session.modified = True
            
            plot_url = plot_stock_data_and_predictions(data, prediction, conf_int, prediction_days, adjusted_prediction)
            
            context['historical_data'] = historical_data
            context['predictions'] = predictions
            context['prediction_days'] = prediction_days
            context['plot_url'] = plot_url
            
            
        except Exception as e:
            context['error'] = str(e)
            
    context['queries'] = session['queries']
    context['current_date'] = current_date
    return render_template('index.html', **context)

@app.route('/download')
def download():
    if 'queries' not in session or len(session['queries']) == 0:
        return "No queries to download.", 400

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Ticker', 'Start Date', 'End Date', 'Prediction Days', 'Day', 'Predicted Price', 'Adjusted Predicted Price', 'Lower Bound', 'Upper Bound'])

    for query in session['queries']:
        for prediction in query['predictions']:
            cw.writerow([query['ticker'], query['start_date'], query['end_date'], query['prediction_days'],
                         prediction['day'], prediction['predicted'], prediction['adjusted_predicted'], prediction['lower'], prediction['upper']])

    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype='text/csv', download_name='query_history.csv', as_attachment=True )



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)