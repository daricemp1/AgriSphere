#making http requests to the open-Meteo API 
import requests
import pandas as pd
from prophet import Prophet
from datetime import date, timedelta


#fetches historical weather data for punjab and resturs a data frame with comlumns
# ds: datetime of each day
# y: daily precipitation sum (mm)

def fetch_weather(
    latitude=30.7333,
    longitude=76.7794,
    history_years=2
):
    today = date.today()
    yesterday = today - timedelta(days=1)
    start = yesterday - timedelta(days=365 * history_years)

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&start_date={start.isoformat()}"
        f"&end_date={yesterday.isoformat()}"
        "&daily=precipitation_sum"
        "&timezone=auto"
    )

    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame({
        "ds": pd.to_datetime(data["time"]),
        "y": data["precipitation_sum"]
    })
    return df


#Uses Prophet to forecast rainfall 
# params : feeds the past 2 years data (df) , threshold--> 0.1mm is the minimum amount of rain to be considered as raining
def predict_rain(df, rain_threshold=0.1):
    # 1) Pre-built prophet model will only look for daily patterns ( ignore any weekly and seasonal patterns )
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
    )
    # trains the model on the actual historical weather data 
    model.fit(df)

    # 2) make future & predict
    # tells prohpet to make table for predictions over the next 2 days, each day one row in the table 
    future   = model.make_future_dataframe(periods=2, freq="D")
    # fills in the future amount of rain wih  the prediction 
    forecast = model.predict(future)

    # 3) sum up rain forecasts
    # grabs the last 2 rows, selects the predicted rainfall values (Prophet calls its prediction column yhat), 
    rain_amt = forecast.tail(2)["yhat"].sum()

    # 4) return mtrained prophet model , the full forecast table , and a comparison to say if it will rain a decent amount 
    return model, forecast, (rain_amt > rain_threshold), rain_amt
