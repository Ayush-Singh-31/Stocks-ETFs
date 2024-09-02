import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class ETF:
    def __init__(self, name):
        self.name = name
        self.df = self.read_etfs()
        self.cleaned_df = None

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
            raise ValueError("Data must be cleaned before plotting.")

        df = self.cleaned_df
        self.plot_series(df, 'Open', 'Open Price')
        self.plot_series(df, 'Close', 'Close Price')
        self.plot_series(df, 'High', 'Highs')
        self.plot_series(df, 'Low', 'Lows')
        self.plot_bar(df, 'Volume in Millions', 'Volume')

    def plot_series(self, df, column, title):
        plt.figure(figsize=(8, 5))
        plt.plot(df["Date"], df[column], color='royalblue', linestyle='-', linewidth=1, label=title)
        plt.title(f'ETF {title}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"./Plots/ETFs/{self.name}/{title}.png")

    def plot_bar(self, df, column, title):
        plt.figure(figsize=(8, 5))
        plt.bar(df["Date"], df[column], color='royalblue', label=title)
        plt.title(f'ETF {title}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"./Plots/ETFs/{self.name}/{title}.png")

    def ml_preprocess(self) -> pd.DataFrame:
        df = self.cleaned_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        return df

    def ml_processing(self):
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
        mse = mean_squared_error(testY, predY)
        R2 = r2_score(testY, predY)
        print(f"Mean Squared Error is {mse}")
        print(f"R2 Score is {R2}")

def main():
    ticker = input("Enter the ticker: ").strip().lower()
    etf = ETF(ticker)
    etf.describe()
    etf.clean()
    etf.plot_folder()
    etf.pre_plot()
    etf.ml_processing()

if __name__ == '__main__':
    main()
