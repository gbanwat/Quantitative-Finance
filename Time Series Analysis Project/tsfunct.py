#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Time Series Analysis of Stocks: Project
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: stock_data
# Takes ticker, start_date, end_date as parameters.
# Fetches historical stock data from the Yahoo Finance API, prints summary statistics,
# displays major and institutional holders, and returns the ticker name with closing prices.
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
def stock_data(ticker,start,end):
    """
    Fetches historical stock data for a given ticker symbol between the specified
    start and end dates. The function downloads stock price data, prints summary
    statistics, shows major and institutional holders, and returns the ticker name
    along with the 'Close' price series.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        tuple: (ticker_name, pandas.Series of closing prices)
    """
    import yfinance as yf
    import pandas as pd
    ticker_name=ticker
    start_date=start       #Format: YYYY-MM-DD
    end_date=end           #Format: YYYY-MM-DD
    ticker=yf.Ticker(ticker_name)
    stock_data=yf.download(ticker_name,start=start_date,end=end_date)
    print(stock_data.describe())
    stock_series=stock_data['Close']
    print(ticker.major_holders)
    pd.set_option("display.max_rows", None)
    print(ticker.institutional_holders)
    return ticker_name,stock_series
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------# Function: stock_line_chart
# Takes ticker_name and stock_series as parameters.
# Plots a line chart of the stock's closing prices with properly formatted date labels,
# grid, title, and legend. Displays the chart using Matplotlib.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def stock_line_chart(ticker_name,stock_series):
    """
    Plots a line chart of a stock’s closing price series.

    The function generates a time series line chart for the given stock, formats
    the x-axis with month-year labels, applies major and minor tick locators, 
    and displays the chart with title, axis labels, legend, and grid.

    Parameters:
        ticker_name (str): The stock ticker symbol (e.g., 'AAPL').
        stock_series (pandas.Series): The closing price series with DateTime index.

    Returns:
        None: Displays the Matplotlib chart.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.figure(figsize=(20,5))
    plt.plot(stock_series.index,stock_series,label=ticker_name + ' Close Price')
    # Format x-axis to show Month-Year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., Jan 2022
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # show every 3 months
    # Minor ticks: every month
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.title(f'{ticker_name} Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.grid()
    print('CHECK WHAT TYPE OF DECOMPOSITION IS REQUIRED, DEFAULT WILL BE ADDITIVE')
    return plt.show()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------# Function: plot_decomposition_charts
# Takes ticker_name, stock_series, trend, seasonal, and residual as parameters.
# Creates a 4-panel chart to visualize the stock's original close price, trend, 
# seasonality, and residual components from a time series decomposition.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_decomposition_charts(ticker_name,stock_series,trend, seasonal, residual):
    """
    Plots decomposition charts of a stock’s time series data.

    The function generates a 4-subplot figure:
        1. Stock closing price
        2. Trend component
        3. Seasonal component
        4. Residual component

    Each subplot is styled with month-year formatted x-axis labels, grid lines,
    and appropriate titles for clear interpretation of decomposition results.

    Parameters:
        ticker_name (str): The stock ticker symbol (e.g., 'AAPL').
        stock_series (pandas.Series): Original closing price series with DateTime index.
        trend (pandas.Series): Extracted trend component of the series.
        seasonal (pandas.Series): Extracted seasonal component of the series.
        residual (pandas.Series): Extracted residual component of the series.

    Returns:
        None: Displays the Matplotlib decomposition plots.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.figure(figsize=(20,15))
    plt.subplot(411)
    plt.title(f'{ticker_name} Stock Close Price')
    plt.plot(stock_series.index,stock_series,label=ticker_name + ' Close Price',color='blue')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., Jan 2022
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # show every 3 months
    # Minor ticks: every month
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
    plt.grid(axis='y', linestyle='--', color='lightgray', alpha=0.7)

    plt.subplot(412)
    plt.title(f'{ticker_name} Stock Trend')
    plt.plot(stock_series.index,trend,label=ticker_name+' Trend',color='blue')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., Jan 2022
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # show every 6 months
    # Minor ticks: every month
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xlabel('Date')
    plt.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
    plt.grid(axis='y', linestyle='--', color='lightgray', alpha=0.7)
    plt.ylabel('Price in USD')

    plt.subplot(413)
    plt.title(f'{ticker_name} Stock Seasonality')
    plt.plot(stock_series.index,seasonal,label=ticker_name+' Seasonality',color='blue')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., Jan 2022
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # show every 3 months
    # Minor ticks: every month
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xlabel('Date')
    plt.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
    plt.grid(axis='y', linestyle='--', color='lightgray', alpha=0.7)
    #plt.ylabel('Price in USD')


    plt.subplot(414)
    plt.plot(stock_series.index,residual,label=ticker_name+' Residual',color='blue')
    plt.title(f'{ticker_name} Stock Residual')
    plt.xlabel('Date')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # e.g., Jan 2022
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # show every 3 months
    # Minor ticks: every month
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
    plt.grid(axis='y', linestyle='--', color='lightgray', alpha=0.7)
    plt.tight_layout()
    return plt.show()
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------# Function: decompose_model
# Takes ticker_name, stock_series, decomposition_type, model_type, and period as parameters.
# Performs time series decomposition using either Classical (seasonal_decompose) or STL.
# Returns a set of decomposition plots showing trend, seasonality, and residuals.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def decompose_model(ticker_name,stock_series,decomposition_type='classical',model_type='additive',period=180):
    """
    Performs time series decomposition on a stock’s closing price series.

    The function supports both Classical decomposition (using seasonal_decompose)
    and STL decomposition. Depending on the chosen method and model type, it extracts
    trend, seasonal, and residual components, then calls plot_decomposition_charts()
    to visualize the results.

    Parameters:
        ticker_name (str): The stock ticker symbol (e.g., 'AAPL').
        stock_series (pandas.Series): Closing price series with DateTime index.
        decomposition_type (str, optional): Type of decomposition to use.
                                            Options: 'classical' (default), 'stl'.
        model_type (str, optional): Type of model for decomposition.
                                    Options: 'additive' (default), 'multiplicative'.
                                    Note: STL only supports 'additive'.
        period (int, optional): Period of seasonality. Default is 180.

    Raises:
        ValueError: If invalid decomposition_type or model_type is passed.

    Returns:
        None: Displays decomposition plots (trend, seasonal, residual).
    """
    if decomposition_type not in ['classical','stl']:
        raise ValueError(f'Decomposition Type must be {'classical'} or {'stl'}')

    if decomposition_type=='stl' and model_type !='additive':
        raise ValueError(f'STL Decomposition only supports {'additive'} model')
    
    if model_type not in ['additive','multiplicative']:
        raise ValueError(f'Model must be either {'additive'} or {'multiplicative'}')
    
    if decomposition_type=='classical':
        from statsmodels.tsa.seasonal import seasonal_decompose
        sea_dec=seasonal_decompose(stock_series,model=model_type,period=period)
        trend=sea_dec.trend
        seasonal=sea_dec.seasonal
        residual=sea_dec.resid

    if decomposition_type=='stl':
        from statsmodels.tsa.seasonal import STL
        stl_dec=STL(stock_series,period=period)
        result=stl_dec.fit()
        trend=result.trend
        seasonal=result.seasonal
        residual=result.resid

    return plot_decomposition_charts(ticker_name,stock_series,trend, seasonal, residual)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------# Function: adf_statistic
# Takes a stock_series (time series data) as input.
# Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.
# Prints test results and returns the p-value.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#ADF Test Description:
#H0:The time series has a unit root -->The time series is non-stationary
#H1:The time series does not have a unit root --> The time series is stationary
# Check with ADF statistics p-value if the data is stationary or non-stationary
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def adf_statistic(stock_series):
    """
    Performs the Augmented Dickey-Fuller (ADF) test on a time series.

    The ADF test checks whether the series is stationary or non-stationary.
    The function prints a detailed interpretation of the results and returns 
    the p-value.

    Parameters:
        stock_series (pandas.Series): Time series data (e.g., stock closing prices).

    Returns:
        float: The p-value from the ADF test, rounded to 4 decimal places.

    Interpretation:
        - p-value < 0.05: Reject null hypothesis → The time series is stationary.
        - p-value >= 0.05: Fail to reject null hypothesis → The time series is non-stationary.
    """
    import numpy as np
    from statsmodels.tsa.stattools import adfuller
    adf_test=adfuller(stock_series)
    p_value=adf_test[1]
    print('---------------------------------------------------------------')
    print('Augmented Dickey Fuller (ADF) Statistic Results:')
    print('---------------------------------------------------------------')
    if p_value<0.05:
        print(f'The p_value ({np.round(p_value,4)}) is less than 0.05')
        print(f'Reject the null hypotheses (Time series is non-stationary).')
        print(f'The time series is stationary ')
        print('---------------------------------------------------------------')
    else:
        print(f'The p_value ({np.round(p_value,4)}) is greater than or equal to 0.05')
        print(f'Failed to reject the null hypotheses (Time series is non-stationary).')
        print(f'The time series is non-stationary ')
        print('---------------------------------------------------------------')
    return np.round(p_value,4)
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------# Function: kpss_statistic
# Takes a stock_series (time series data) and a regression type ('c' or 'ct').
# Performs the KPSS test to check for stationarity.
# Prints the test results and returns the p-value.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#KPSS Test Description:
#H0:The time series is stationary (trend stationary)
#H1:The time series is non-stationary (has a unit root)
# Check with KPSS statistics p-value if the data is stationary or non-stationary
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def kpss_statistic(stock_series,reg_type):
    """
    Performs the KPSS (Kwiatkowski–Phillips–Schmidt–Shin) test on a time series.

    The KPSS test checks whether a series is stationary around a mean (level) 
    or around a deterministic trend. The function prints the results and 
    returns the p-value.

    Parameters:
        stock_series (pandas.Series): Time series data (e.g., stock closing prices).
        reg_type (str): Type of regression to include in the test.
                        - 'c'  : Constant (level stationarity)
                        - 'ct' : Constant and trend (trend stationarity)

    Returns:
        float: The p-value from the KPSS test, rounded to 4 decimal places.

    Interpretation:
        - p-value >= 0.05: Fail to reject null hypothesis → The time series is stationary.
        - p-value < 0.05 : Reject null hypothesis → The time series is non-stationary.

    Raises:
        ValueError: If reg_type is not 'c' or 'ct'.
    """
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    if reg_type not in ['ct','c']:
        raise ValueError('KPSS Statistics Regresion Type cannot be other than (c) or (ct)')
    from statsmodels.tsa.stattools import kpss
    kpss_stat=kpss(stock_series,regression=reg_type)
    p_value=kpss_stat[1]
    print('---------------------------------------------------------------')
    print('Kwiatkowski–Phillips–Schmidt–Shin (KPSS) Statistic Results:')
    print('---------------------------------------------------------------')
    if p_value>=0.05:
        print(f'The p-value ({np.round(p_value,4)}) is greater than 0.05')
        print(f'Failed to reject the null-hupotheses')
        print(f'The time series is stationary.')
        print('---------------------------------------------------------------')
    else:
        print(f'The p_value ({np.round(p_value,4)}) is less than or equal to 0.05')
        print(f'Reject the null hypotheses  (Time series is stationary).')
        print(f'The time series is non-stationary ')
        print('---------------------------------------------------------------')
    return np.round(p_value,4)
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: check_stationarity_type
# Takes a stock_series (time series data) as input.
# Splits the series into two halves and performs a two-sample Kolmogorov-Smirnov (KS) test
# to compare the distributions of the two halves.
# Returns the p-value indicating whether the series is stationary in distribution.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_stationarity_type(stock_series):
    """
    Checks the stationarity of a time series by comparing distributions of its two halves.

    The function splits the series into the first and second halves and uses the
    two-sample Kolmogorov-Smirnov (KS) test to determine if the two halves come
    from the same distribution. A higher p-value suggests the series is more likely
    stationary in distribution.

    Parameters:
        stock_series (pandas.Series): Time series data (e.g., stock closing prices).

    Returns:
        float: p-value from the KS test.

    Interpretation:
        - p-value >= 0.05: Fail to reject the null hypothesis → the two halves are likely from the same distribution (stationary).
        - p-value < 0.05 : Reject the null hypothesis → the two halves differ significantly (non-stationary).
    """
    from scipy.stats import ks_2samp
    split=len(stock_series)//2
    first_half=stock_series[:split]
    second_half=stock_series[split:]
    ks_2samp_stat,p_value=ks_2samp(first_half,second_half)
    return p_value


#--------------------------------------------------------------------------------------------------------------------------------------------------------------# Function: stationarity_check
# Takes a stock_series (time series data) and optional KPSS regression type.
# Uses both ADF and KPSS tests to determine if the series is stationary.
# Prints an interpretation and suggests whether transformation is required.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def stationarity_check(stock_series,kpss_reg_type='ct'):
    """
    Checks the stationarity of a time series using both ADF and KPSS tests.

    The function applies the Augmented Dickey-Fuller (ADF) test and the
    KPSS test to the input series. Based on the combination of results:
        - If ADF indicates stationarity and KPSS also indicates stationarity, 
          the series is confidently stationary.
        - If both tests indicate non-stationarity or disagree, the series may
          require transformation (e.g., differencing, log transform) to become stationary.

    Parameters:
        stock_series (pandas.Series): Time series data (e.g., stock closing prices).
        kpss_reg_type (str, optional): Regression type for KPSS test.
                                       Options: 'c' (level) or 'ct' (trend). Default is 'ct'.

    Returns:
        str: Recommendation for transformation.
             - "No Transformation Required" → series is stationary.
             - "Transformation Required"    → series is non-stationary or borderline.

    Side Effects:
        Prints detailed interpretations of both ADF and KPSS results, 
        along with a summary conclusion on stationarity.
    """
    adf_pvalue=adf_statistic(stock_series)
    kpss_pvalue=kpss_statistic(stock_series,reg_type=kpss_reg_type)
    
    if adf_pvalue<0.05 and kpss_pvalue>=0.05:
        print(f'The time series is confidently STATIONARY.')
        return "No Transformation Required"
        
    elif adf_pvalue>=0.05 and kpss_pvalue<0.05:
        print(f'The time series is confidently NON-STATIONARY')
        return "Transformation Required"
        
    else:
        print(f'KPSS stat and ADF stat disagree on the series being stationary')
        print(f'The time series is on the borderline between being stationary & non-stationary')
        print(f'Hence further transformation is required to make the time series stationary')
        return "Transformation Required"

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: make_stationary
# Takes a stock_series (time series data) as input.
# Iteratively applies a set of transformations in priority order to make the series stationary.
# Prints stationarity statistics for each transformation using ADF and KPSS tests.
# Returns the first transformed series that passes stationarity checks.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_stationary(stock_series):
    """
    Attempts to make a time series stationary by applying multiple transformations in order of priority.

    The function applies the following transformations sequentially:
        1. Logarithmic transformation
        2. Square-root transformation
        3. Box-Cox transformation
        4. Linear detrending
        5. Moving average detrending
        6. Seasonal detrending using STL
        7. First order differencing
        8. Second order differencing

    After each transformation, it uses `stationarity_check` (ADF + KPSS) to test stationarity.
    The first transformation that results in a stationary series is returned.

    Parameters:
        stock_series (pandas.Series): Time series data (e.g., stock closing prices).

    Returns:
        pandas.Series: Transformed stationary series.

    Raises:
        ValueError: If none of the transformation methods result in a stationary series.

    Side Effects:
        Prints detailed statistics and transformation type for each step.
    """
    import numpy as np
    priority=['log','sqrt','box_cox','detrending_line','detrendeing_ma','detrending_seasonal','diff','diff_2']
    for method in priority:
        
        if method=='log':
            transformed=np.log(stock_series)
            print(f'------------------------------------------------------------------------------------------------------')
            print(f'Statistics for Logarithmic Transformation of the Time Series:')
            print(f'------------------------------------------------------------------------------------------------------')
            result=stationarity_check(transformed)
            if result=='No Transformation Required':
                return transformed
                
        if method=='sqrt':
            transformed=np.sqrt(stock_series)
            print(f'------------------------------------------------------------------------------------------------------')
            print(f'Statistics for Square-root Transformation of the Time Series:')
            print(f'------------------------------------------------------------------------------------------------------')
            result=stationarity_check(transformed)
            if result=='No Transformation Required':
                return transformed
                
        if method=='box_cox':
            #Box-Cox Transformation
            #Note: Box-Cox requires all positive values
            #Filter positive values and force 1D
            from scipy import stats
            stock_series_boxcox=stock_series[stock_series>0]
            transformed,lam=stats.boxcox(stock_series_boxcox.values.reshape(1,len(stock_series_boxcox)).ravel())
            print(f'------------------------------------------------------------------------------------------------------')
            print(f'Statistics for Box-Cox Transformation of the Time Series:')
            print(f'------------------------------------------------------------------------------------------------------')
            result=stationarity_check(transformed)
            if result=='No Transformation Required':
                return transformed
                
        if method=='detrending_line':
            from sklearn.linear_model import LinearRegression
            lin_reg=LinearRegression()
            x=np.arange(len(stock_series)).reshape(-1,1)
            y=stock_series.values
            lin_reg.fit(x,y)
            line=lin_reg.predict(x)
            transformed=stock_series-line
            print(f'------------------------------------------------------------------------------------------------------')
            print(f'Statistics for Detrending Line Transformation of the Time Series:')
            print(f'------------------------------------------------------------------------------------------------------')
            result=stationarity_check(transformed)
            if result=='No Transformation Required':
                return transformed
                
        if method=='detrending_ma':
            transformed=stock_series-stock_series.rolling(window=50).mean()
            transformed=transformed.dropna()
            print(f'------------------------------------------------------------------------------------------------------')
            print(f'Statistics for Detrending Moving Average Transformation of the Time Series:')
            print(f'------------------------------------------------------------------------------------------------------')
            result=stationarity_check(transformed)
            if result=='No Transformation Required':
                return transformed

        if method=='detrending_seasonal':
            from statsmodels.tsa.seasonal import STL
            stl_dec=STL(stock_series,period=30)
            dec_result=stl_dec.fit()
            seasonal=dec_result.seasonal
            print(f'------------------------------------------------------------------------------------------------------')
            print(f'Statistics for Detrending Seasonal Transformation of the Time Series:')
            print(f'------------------------------------------------------------------------------------------------------')
            transformed=stock_series.iloc[:,0]-seasonal
            result=stationarity_check(transformed)
            if result=='No Transformation Required':
                return transformed
            

        if method=='diff':
            transformed=stock_series.diff().dropna()
            print(f'------------------------------------------------------------------------------------------------------')
            print(f'Statistics for First Order Differencing Transformation of the Time Series:')
            print(f'------------------------------------------------------------------------------------------------------')
            result=stationarity_check(transformed)
            if result=='No Transformation Required':
                return transformed
                
        if method=='diff_2':
            transformed=stock_series.diff().diff().dropna()
            print(f'------------------------------------------------------------------------------------------------------')
            print(f'Statistics for Second Order Differencing Transformation of the Time Series:')
            print(f'------------------------------------------------------------------------------------------------------')
            result=stationarity_check(transformed)
            if result=='No Transformation Required':
                return transformed
    else:
        raise ValueError('None of the Transformations Methods Worked')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: plot_stationary_time_series
# Takes a stationary_series (time series data) as input.
# Plots the stationary series with proper labels, grid, and title using Matplotlib.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_stationary_time_series(stationary_series):
    """
    Plots a stationary time series.

    The function generates a line chart of the input stationary series, adds a title,
    axis labels, legend, and grid for visualization.

    Parameters:
        stationary_series (pandas.Series): Stationary time series data with DateTime index.

    Returns:
        None: Displays the Matplotlib plot.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.plot(stationary_series.index,stationary_series,label='Stationary Series',color='red')
    plt.legend()
    plt.title('Stationary Time Series')
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------# Function: acf_pacf_plots
# Takes a stationary_series (time series data) as input.
# Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) 
# diagrams using Matplotlib and statsmodels.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def acf_pacf_plots(stationary_series):
    """
    Plots the ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) of a stationary time series.

    The function generates two side-by-side plots showing the correlation of the series
    with its own lagged values, which helps in determining appropriate AR and MA orders
    for time series modeling (ARIMA).

    Parameters:
        stationary_series (pandas.Series): Stationary time series data with DateTime index.

    Returns:
        None: Displays the Matplotlib plots for ACF and PACF.
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plt.figure(figsize=(12,7))
    plt.subplot(121)
    plot_acf(stationary_series,ax=plt.gca(),lags=30)
    plt.grid()
    plt.title('ACF Diagram')

    plt.subplot(122)
    plot_pacf(stationary_series,ax=plt.gca(),lags=30)
    plt.grid()
    plt.title('PACF Diagram')
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: run_arima
# Takes train_data, test_data, and ARIMA order parameters (p,d,q).
# Fits an ARIMA model on the training data, predicts on the test data,
# plots actual vs predicted values, and returns the RMSE.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_arima(train_data,test_data,p,d,q):
    """
    Fits an ARIMA model to training data and evaluates it on test data.

    The function builds an ARIMA(p,d,q) model using the training series,
    predicts the test series, plots actual vs predicted values, and computes
    the root mean squared error (RMSE) as a measure of prediction accuracy.

    Parameters:
        train_data (pandas.Series): Training portion of the time series.
        test_data (pandas.Series): Test portion of the time series for evaluation.
        p (int): AR (autoregressive) order.
        d (int): Degree of differencing.
        q (int): MA (moving average) order.

    Returns:
        float: Root Mean Squared Error (RMSE) between predicted and actual test values.

    Side Effects:
        Plots a line chart comparing actual vs predicted values of the test data.
    """
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    import numpy as np
    model=ARIMA(train_data,order=(p,d,q))
    model_fit=model.fit()
    pred=model_fit.predict(start=len(train_data),
                          end=len(train_data)+len(test_data)-1,
                          dynamic=False)
    plt.figure(figsize=(10,5))
    plt.plot(test_data.index,test_data,label='Actual Values',color='blue')
    plt.plot(test_data.index,pred,label='Predicted Values',color='red',linestyle='--')
    plt.grid()
    plt.ylabel('Close Price (USD)')
    plt.xlabel('Date')
    plt.legend()
    plt.title(f'ARIMA Model Check (p={p},d={d},q={q})')
    plt.show()
    rmse=np.round(np.sqrt(mean_squared_error(test_data,pred)),4)
    return rmse

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function: forecast_arima
# Takes a stock_series (time series data) and ARIMA order parameters (p,d,q) along with forecast steps.
# Fits an ARIMA model, forecasts future values, and plots the forecast along with recent actual values.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
def forecast_arima(stock_series,p,d,q,step):
    """
    Performs ARIMA forecasting on a time series and visualizes the forecast.

    The function fits an ARIMA(p,d,q) model on the input series, generates forecasts
    for the specified number of steps, and plots the forecasted values alongside
    the most recent actual data points.

    Parameters:
        stock_series (pandas.Series): Time series data (e.g., stock closing prices).
        p (int): AR (autoregressive) order.
        d (int): Degree of differencing.
        q (int): MA (moving average) order.
        step (int): Number of future time steps to forecast.

    Returns:
        None: Displays a Matplotlib plot of forecasted and recent actual values.

    Side Effects:
        Plots a line chart showing the forecast values in red and recent actual values in blue.
    """
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    import pandas as pd
    #from sklearn.metrics import mean_squared_error
    import numpy as np
    model=ARIMA(stock_series,order=(p,d,q))
    model_fit=model.fit()
    forecast=model_fit.forecast(steps=step)
    last_date=stock_series.index[-1]
    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=step, freq='D')
    forecast_series=pd.Series(forecast.values,index=new_dates)
    plt.figure(figsize=(15,5))
    plt.plot(forecast_series.index,forecast_series,label='Forecast Values',color='red',linestyle='--')
    plt.plot(stock_series.index[-5:],stock_series[-5:],label='Actual Values',color='blue',linestyle='--')
    plt.grid()
    plt.ylabel('Close Price (USD)')
    plt.xlabel('Date')
    plt.legend()
    plt.title(f'ARIMA Model (p={p},d={d},q={q}) : Forecast Prices')
    plt.show()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ 


    
    