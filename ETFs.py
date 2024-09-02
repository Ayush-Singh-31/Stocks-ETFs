import pandas as pd
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

def prePlot(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['Open'], color='royalblue', linestyle='-', linewidth=1, label='Open Price')
    plt.title('ETF Opening Prices', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Open Price', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['Close'], color='royalblue', linestyle='-', linewidth=1, label='Close Price')
    plt.title('ETF Closing Prices', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['High'], color='royalblue', linestyle='-', linewidth=1, label='High')
    plt.title('ETF Highs', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Highs', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['Low'], color='royalblue', linestyle='-', linewidth=1, label='Low')
    plt.title('ETF Lows', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Lows', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df['Date'], df['Volume in Millions'], color='royalblue', linestyle='-', linewidth=1, label='Volume')
    plt.title('ETF Volume', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volume in Millions', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    ticker = input("Enter the ticke: ").strip().lower()
    ticker = readETFs(ticker)
    # describeETF(ticker)
    cleanTicker = clean(ticker)
    describeETF(cleanTicker)
    prePlot(cleanTicker)

if __name__ == '__main__':
    main()