PG 6
!pip install -q accelerate transformers[torch] datasets
import pandas as pd
from transformers import pipeline
senti_model = pipeline("sentiment-analysis")
print(senti_model("This movie is damn good. I loved it"))
print(senti_model("This is a bad phone. The screen and battery are of poor quality."))
print(senti_model("Over heating issue don't by this product camera was good"))
df = pd.read_csv("https://raw.githubusercontent.com/venkatareddykonasani/Datasets/master/Amazon_Yelp_Reviews/Review_Data.csv").sample(50)
df["Predicted_Sentiment"] = df["Review"].apply(lambda x: senti_model(x)[0]["label"])
df
