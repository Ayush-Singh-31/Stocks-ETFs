import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score

class ETF:
    def __init__(self, name):
        self.name = name
        self.df = self.read_etfs()
        self.cleaned_df = None
        self.r2Score = None
        self.mse = None
        self.trainX = None
        self.trainY  = None
        self.testX = None
        self.testY = None
        self.model = None
        self.features = None
        self.target = None

    def read_etfs(self) -> pd.DataFrame:
        df = pd.read_csv(f"./Data/ETFs/{self.name}.us.txt")
        return df

    def describe(self, df: pd.DataFrame):
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

    def clean(self) -> pd.DataFrame:
        df = self.df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop("OpenInt", axis=1)
        df["Volume"] = df["Volume"] / 10**7
        df = df.rename(columns={"Volume": "Volume in Millions"})
        self.cleaned_df = df
        return df

    def plot_folder(self):
        if not os.path.exists(f"./Plots/ETFs/{self.name}"):
            os.makedirs(f"./Plots/ETFs/{self.name}")
        return

    def plot_open_price(self):
        if self.cleaned_df is None:
            raise ValueError("Data is not cleaned before plotting.")
        self.plot_folder()
        self.plot_series(self.cleaned_df, 'Open', 'Open Price')

    def plot_close_price(self):
        if self.cleaned_df is None:
            raise ValueError("Data is not cleaned before plotting.")
        self.plot_folder()
        self.plot_series(self.cleaned_df, 'Close', 'Close Price')

    def plot_highs(self):
        if self.cleaned_df is None:
            raise ValueError("Data is not cleaned before plotting.")
        self.plot_folder()
        self.plot_series(self.cleaned_df, 'High', 'Highs')

    def plot_lows(self):
        if self.cleaned_df is None:
            raise ValueError("Data is not cleaned before plotting.")
        self.plot_folder()
        self.plot_series(self.cleaned_df, 'Low', 'Lows')

    def plot_volume(self):
        if self.cleaned_df is None:
            raise ValueError("Data is not cleaned before plotting.")
        self.plot_folder()
        self.plot_bar(self.cleaned_df, 'Volume in Millions', 'Volume')

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

    def ml_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Day"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df = df.drop('Date',axis=1)
        df.dropna(inplace=True)
        return df

    def linearModel(self, df, features, target):
        X = df[features]
        Y = df[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, random_state = 0)
        model = LinearRegression()
        model.fit(trainX, trainY)
        predY = model.predict(testX)
        self.mse = mean_squared_error(testY, predY)
        self.r2Score = r2_score(testY, predY)
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.model = model
        self.features = features
        self.target = target

    def dtr(self, df, features, target, mln = None):
        X = df[features]
        Y = df[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, random_state = 0)
        model = DecisionTreeRegressor(random_state = 0, max_leaf_nodes = mln)
        model.fit(trainX, trainY)
        predY = model.predict(testX)
        self.mse = mean_squared_error(testY, predY)
        self.r2Score = r2_score(testY, predY)
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.model = model
        self.features = features
        self.target = target

    def dtrCheck(self):
        def get_mae(maxLeaf, trainX, testX, trainY, testY):
            model = DecisionTreeRegressor(max_leaf_nodes = maxLeaf, random_state = 0)
            model.fit(trainX, trainY)
            preds_val = model.predict(testX)
            mae = mean_absolute_error(testY, preds_val)
            return(mae)
        
        for maxLeaf in [5, 50, 500, 5000]:
            my_mae = get_mae(maxLeaf, self.trainX, self.testX, self.trainY, self.testY)
            print(f"Max leaf nodes: {maxLeaf}  \t\t Mean Absolute Error:  {my_mae}")

    def rfr(self, df, features, target):
        X = df[features]
        Y = df[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, random_state = 0)
        model = RandomForestRegressor(random_state = 0)
        model.fit(trainX, trainY)
        predY = model.predict(testX)
        self.mse = mean_squared_error(testY, predY)
        self.r2Score = r2_score(testY, predY)
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.model = model
        self.features = features
        self.target = target

    def plotPredictions(self):
        if self.testX is None or self.testY is None:
            raise ValueError("Model has not been trained or data is not prepared.")
        model = self.model
        predY = model.predict(self.testX)  
        plt.figure(figsize=(10, 6))
        plt.scatter(self.testY, predY, color='royalblue', alpha=0.7, edgecolors='w', linewidth=0.5)
        plt.plot([min(self.testY), max(self.testY)], [min(self.testY), max(self.testY)], color='red', linestyle='--')
        plt.title('Predictions vs Actuals', fontsize=16)
        plt.xlabel('Actual Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

ticker = "qqq"
etf = ETF(ticker)

etf.describe(etf.df)

etf.clean()
etf.describe(etf.cleaned_df)

etf.plot_open_price()

etf.plot_close_price()

etf.plot_highs()

etf.plot_lows()

etf.plot_volume()

def expand(df):
        df["Daily Return"] = (df["Close"] - df["Open"]) / df["Open"] * 100

        df["20-Day MA"] = df["Close"].rolling(window=20).mean()
        df["50-Day MA"] = df["Close"].rolling(window=50).mean()
        df["200-Day MA"] = df["Close"].rolling(window=200).mean()
        
        df["30-Day Volatility"] = df["Daily Return"].rolling(window=30).std()

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["Middle Band"] = df["20-Day MA"]
        df["Upper Band"] = df["Middle Band"] + (2 * df["Close"].rolling(window=20).std())
        df["Lower Band"] = df["Middle Band"] - (2 * df["Close"].rolling(window=20).std())

        df["Cumulative Return"] = (df["Close"] / df["Close"].iloc[0]) - 1

        df["VWAP"] = (df["Volume in Millions"] * df["Close"]).cumsum() / df["Volume in Millions"].cumsum()

expand(etf.cleaned_df)
etf.describe(etf.cleaned_df)

plt.figure(figsize=(12, 8))
sns.histplot(etf.cleaned_df['Daily Return'].dropna(), kde=True, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Daily Returns', fontsize=16, fontweight='bold')
plt.xlabel('Daily Return (%)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"./Plots/ETFs/{etf.name}/Daily Returns.png")

plt.figure(figsize=(16, 8))
sns.lineplot(x='Date', y='Close', data=etf.cleaned_df, label='Close', color='blue', linewidth=2)
sns.lineplot(x='Date', y='20-Day MA', data=etf.cleaned_df, label='20-Day MA', color='orange', linestyle='--', linewidth=2)
sns.lineplot(x='Date', y='50-Day MA', data=etf.cleaned_df, label='50-Day MA', color='green', linestyle='--', linewidth=2)
sns.lineplot(x='Date', y='200-Day MA', data=etf.cleaned_df, label='200-Day MA', color='red', linestyle='--', linewidth=2)
plt.title('Close Price with Moving Averages', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend(title='Legend', title_fontsize='13', fontsize='12')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"./Plots/ETFs/{etf.name}/Moving Average.png")

plt.figure(figsize=(16, 8))
sns.lineplot(x='Date', y='Close', data=etf.cleaned_df, label='Close', color='blue', linewidth=2)
sns.lineplot(x='Date', y='Middle Band', data=etf.cleaned_df, label='Middle Band', color='green', linewidth=2)
sns.lineplot(x='Date', y='Upper Band', data=etf.cleaned_df, label='Upper Band', color='red', linestyle='--', linewidth=2)
sns.lineplot(x='Date', y='Lower Band', data=etf.cleaned_df, label='Lower Band', color='orange', linestyle='--', linewidth=2)
plt.fill_between(etf.cleaned_df['Date'], etf.cleaned_df['Lower Band'], etf.cleaned_df['Upper Band'], color='gray', alpha=0.3, label='Bollinger Bands Range')
plt.title('Bollinger Bands', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend(title='Legend', title_fontsize='13', fontsize='12')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"./Plots/ETFs/{etf.name}/Bollinger Bands.png")

plt.figure(figsize=(16, 8))
sns.lineplot(x='Date', y='RSI', data=etf.cleaned_df, color='blue', linewidth=2)
plt.axhline(70, linestyle='--', color='red', linewidth=1.5, label='Overbought Threshold (70)')
plt.axhline(30, linestyle='--', color='green', linewidth=1.5, label='Oversold Threshold (30)')
plt.title('RSI Over Time', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('RSI', fontsize=14)
plt.legend(title='Legend', title_fontsize='13', fontsize='12')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"./Plots/ETFs/{etf.name}/RSI.png")

plt.figure(figsize=(16, 8))
sns.lineplot(x='Date', y='Cumulative Return', data=etf.cleaned_df, color='teal', linewidth=2)
plt.title('Cumulative Return Over Time', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Return', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"./Plots/ETFs/{etf.name}/Cumulative Returns.png")

plt.figure(figsize=(16, 8))
sns.lineplot(x='Date', y='VWAP', data=etf.cleaned_df, color='purple', linewidth=2)
plt.title('Volume Weighted Average Price (VWAP) Over Time', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('VWAP', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"./Plots/ETFs/{etf.name}/VWAP.png")

plt.figure(figsize=(12, 10))
corr_matrix = etf.cleaned_df.corr()
sns.heatmap(corr_matrix,annot=True, fmt=".2f", cmap="coolwarm",linewidths=0.5,linecolor='white',cbar_kws={'shrink': 0.8, 'orientation': 'vertical'})
plt.title(f'Correlation Matrix', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(f"./Plots/ETFs/{etf.name}/CorrelationMatrix.png", dpi=300)

df = etf.cleaned_df.set_index('Date')
decomposition = seasonal_decompose(df['Close'], model="multiplicative", period=252)  
# Assuming 252 trading days in a year
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
components = {"Observed": decomposition.observed,"Trend": decomposition.trend,"Seasonal": decomposition.seasonal,"Residual": decomposition.resid}
for ax, (label, data) in zip(axes, components.items()):
    data.plot(ax=ax, color='tab:blue')
    ax.set_ylabel(label, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f'{label} Component', fontsize=14, fontweight='bold', pad=10)
plt.suptitle(f'Seasonal Decomposition', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"./Plots/ETFs/{etf.name}/SeasonalDecomposition.png", dpi=300)


df = etf.cleaned_df.copy()
df['Price Momentum'] = df['Close'].pct_change(periods=10)
plt.figure(figsize=(12, 7))
plt.plot(df["Date"], df["Price Momentum"], color='dodgerblue', label="Price Momentum", linewidth=2)
plt.title(f'10-Day Price Momentum', fontsize=18, fontweight='bold', color='navy')
plt.xlabel('Date', fontsize=14, fontweight='bold', color='darkblue')
plt.ylabel('Momentum', fontsize=14, fontweight='bold', color='darkblue')
plt.grid(True, linestyle='--', alpha=0.5, color='gray')
plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, facecolor='white', edgecolor='gray')
plt.tight_layout()
plt.savefig(f"./Plots/ETFs/{etf.name}/Momentum10Days.png", dpi=300)

etf.cleaned_df = etf.ml_preprocess(etf.cleaned_df)
etf.describe(etf.cleaned_df)

features = list(etf.cleaned_df.columns)
target = features.pop(features.index("Close"))
etf.linearModel(etf.cleaned_df, features, target)

print(etf.mse)

print(etf.r2Score)

etf.plotPredictions()

etf.dtr(etf.cleaned_df, features, target)
print(etf.mse)
print(etf.r2Score)

etf.plotPredictions()

etf.dtrCheck()

etf.dtr(etf.cleaned_df, features, target, 50)
print(etf.mse)
print(etf.r2Score)

etf.plotPredictions()

etf.rfr(etf.cleaned_df, features, target)
print(etf.mse)
print(etf.r2Score)

etf.plotPredictions()

class Neural(nn.Module):
    def __init__(self, input_size):
        super(Neural, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def neuralNetwork(df, features, target, epochs=100, lr=0.001):
    X = torch.tensor(df[features].values, dtype=torch.float32)
    Y = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(1)
    trainX, testX, trainY, testY = train_test_split(X, Y, random_state=0)
    model = Neural(trainX.shape[1])
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(trainX)
        loss = criterion(output, trainY)
        loss.backward() 
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        predictions = model(testX)
        mse = criterion(predictions, testY).item()
        print(f'MSE on test set: {mse:.4f}')
        
    return model, mse, trainX, trainY, testX, testY

count = 0
while True:
    count += 1
    model, mse, trainX, trainY, testX, testY = neuralNetwork(etf.cleaned_df, features, target)
    if mse < 80 or count == 200:
        break