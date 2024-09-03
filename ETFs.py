import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class ETF:
    def __init__(self, name):
        self.name = name
        self.df = self.read_etfs()
        self.cleaned_df = None
        self.r2Score = None
        self.mse = None

    def read_etfs(self) -> pd.DataFrame:
        df = pd.read_csv(f"./Data/ETFs/{self.name}.us.txt")
        return df

    def describe(self):
        print("DataFrame Information:")
        self.df.info()
        
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        
        print("\nFirst 5 Rows:")
        print(self.df.head())
        
        print("\nLast 5 Rows:")
        print(self.df.tail())
        
        print("\nDataFrame Shape (rows, columns):")
        print(self.df.shape)
        
        print("\nRandom Sample of 5 Rows:")
        print(self.df.sample(5))
        
        print("\nMissing Values Count per Column:")
        print(self.df.isnull().sum())
        print()

    def clean(self) -> pd.DataFrame:
        df = self.df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop("OpenInt", axis=1)
        df["Volume"] = df["Volume"]/10**7
        df = df.rename(columns={"Volume":"Volume in Millions"})
        self.cleaned_df = df
        return df

    def plot_folder(self):
        if not os.path.exists(f"./Plots/ETFs/{self.name}"):
            os.makedirs(f"./Plots/ETFs/{self.name}")
        return

    def pre_plot(self):
        if self.cleaned_df is None:
            print("Data is not cleaned before plotting.")
        self.plot_folder()
        df = self.cleaned_df
        self.plot_series(df, 'Open', 'Open Price')
        self.plot_series(df, 'Close', 'Close Price')
        self.plot_series(df, 'High', 'Highs')
        self.plot_series(df, 'Low', 'Lows')
        self.plot_bar(df, 'Volume in Millions', 'Volume')

    def plot_series(self, df, column, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df[column], mode='lines', name=title))
        fig.update_layout(
            title=f'ETF {title}',
            xaxis_title='Date',
            yaxis_title=title,
            template='plotly_white'
        )
        fig.write_html(f"./Plots/ETFs/{self.name}/{title}.html")

    def plot_bar(self, df, column, title):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Date"], y=df[column], name=title))
        fig.update_layout(
            title=f'ETF {title}',
            xaxis_title='Date',
            yaxis_title=title,
            template='plotly_white'
        )
        fig.write_html(f"./Plots/ETFs/{self.name}/{title}.html")

    def ml_preprocess(self) -> pd.DataFrame:
        df = self.cleaned_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        return df

    def linearModel(self):
        if self.cleaned_df is None:
            raise ValueError("Data must be cleaned before processing.")
        
        df = self.ml_preprocess()
        features = ['Open', 'High', 'Low', 'Volume in Millions', 'Day', 'Month', 'Year']
        target = 'Close'
        X = df[features]
        Y = df[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(trainX, trainY)
        predY = model.predict(testX)
        self.mse = mean_squared_error(testY, predY)
        self.r2Score = r2_score(testY, predY)

def main():
    ticker = input("Enter the ticker: ").strip().lower()
    etf = ETF(ticker)
    etf.describe()
    etf.clean()
    etf.pre_plot()
    etf.linearModel()

if __name__ == '__main__':
    main()