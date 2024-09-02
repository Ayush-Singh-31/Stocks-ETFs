import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def readETFs(name) -> pd.DataFrame:
    df = pd.read_csv(f"./Data/ETFs/{name}.us.txt")
    return df

def describeETF(df: pd.DataFrame):
    print("DataFrame Information:")
    df.info()
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nFirst 5 Rows:")
    print(df.head())
    
    print("\nLast 5 Rows:")
    print(df.tail())
    
    print("\nDataFrame Shape (rows, columns):")
    print(df.shape)
    
    print("\nRandom Sample of 5 Rows:")
    print(df.sample(5))
    
    print("\nMissing Values Count per Column:")
    print(df.isnull().sum())
    print()

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop("OpenInt", axis=1)
    df["Volume"] = df["Volume"]/10**7
    df = df.rename(columns={"Volume":"Volume in Millions"})
    return df

def plotFolder(ticker):
    if not os.path.exists(f"./Plots/ETFs/{ticker}"):
        os.makedirs(f"./Plots/ETFs/{ticker}")
    return

def prePlot(df: pd.DataFrame, ticker):
    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['Open'], color='royalblue', linestyle='-', linewidth=1, label='Open Price')
    plt.title('ETF Opening Prices', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Open Price', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./Plots/ETFs/{ticker}/Open.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['Close'], color='royalblue', linestyle='-', linewidth=1, label='Close Price')
    plt.title('ETF Closing Prices', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./Plots/ETFs/{ticker}/Close.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['High'], color='royalblue', linestyle='-', linewidth=1, label='High')
    plt.title('ETF Highs', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Highs', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./Plots/ETFs/{ticker}/High.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['Low'], color='royalblue', linestyle='-', linewidth=1, label='Low')
    plt.title('ETF Lows', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Lows', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./Plots/ETFs/{ticker}/Low.png")

    plt.figure(figsize=(8, 5))
    plt.bar(df['Date'], df['Volume in Millions'], color='royalblue', label='Volume')
    plt.title('ETF Volume', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volume in Millions', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./Plots/ETFs/{ticker}/Volume.png")

    return


def main():
    ticker = input("Enter the ticke: ").strip().lower()
    df = readETFs(ticker)
    describeETF(df)
    cleanDF = clean(df)
    describeETF(cleanDF)
    plotFolder(ticker)
    prePlot(cleanDF, ticker)

if __name__ == '__main__':
    main()