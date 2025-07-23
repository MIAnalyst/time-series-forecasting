import pandas as pd
import numpy as np
# from phik import phik_matrix
from scipy.stats import entropy
import holidays
import matplotlib.pyplot as plt
import requests
import time
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def load_elec_hourly_data(paths:list,exclude_countries=None)->pd.DataFrame:
    """
    Loads hourly electricity load for EU countries 

    Args:
        paths (list): List of file paths to load. 
        exclude_countries (list): List of country codes to exclude. Defaults to None.

    Returns:
        pd.DataFrame: Cleaned, merged dataframe containing only relevant EU countries.
    """
    dfs=[]
    for p in paths:
        sep=',' if p.endswith('2025.csv') else '\t'
        dfs.append(pd.read_csv(p,sep=sep))
    df=pd.concat(dfs,ignore_index=True)
    df=df[['DateUTC','CountryCode','Value']]
    df.columns=['Date','CountryCode','Value']
    df['Date']=pd.to_datetime(df['Date'],dayfirst=True,errors='coerce')
    # normalizing to hour-level precision to be safe
    df['Date'] = df['Date'].dt.floor('h')
    df=df[~df['CountryCode'].isin(exclude_countries)]
    return df


def fetch_eu_weather(eu_countries: dict, start_date: str, end_date: str, sleep_sec: int = 5) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo for given countries and date range."""
    
    variables = [
        'temperature_2m',
        'relative_humidity_2m',
        'wind_speed_10m',
        'shortwave_radiation',
        'cloud_cover'
    ]
    
    all_weather = []
    
    for code, (lat, lon) in eu_countries.items():
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(variables),
            'timezone': 'UTC'
        }

        try:
            r = requests.get('https://archive-api.open-meteo.com/v1/archive', params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"⚠️ Failed to fetch data for {code}: {e}")
            continue

        if 'hourly' in data and 'time' in data['hourly']:
            hourly = data['hourly']
            temp_df = pd.DataFrame({
                'Date': pd.to_datetime(hourly['time']),
                'Temp_C': hourly.get('temperature_2m'),
                'Humidity': hourly.get('relative_humidity_2m'),
                'WindSpeed': hourly.get('wind_speed_10m'),
                'SolarRadiation': hourly.get('shortwave_radiation'),
                'CloudCover': hourly.get('cloud_cover'),
                'CountryCode': code
            })
            all_weather.append(temp_df)
        
        time.sleep(sleep_sec)  
    
    if not all_weather:
        return pd.DataFrame()  

    return pd.concat(all_weather, ignore_index=True)


def datType(c):
    """
    Infer the variable type of a pandas Series.

    Parameters:
    -----------
    c : pandas.Series
        A single column from a DataFrame.

    Returns:
    --------
    str
        The inferred type: "Continuous", "Categorical", "Text", "Date", or "Other".
    """
    if c.nunique() > 20 and c.dtype.kind in "iufc":
        return "Continuous"
    elif c.nunique() < 20 and c.dtype.kind in "iufc":
        return "Categorical"
    elif c.dtype == object:
        return "Text"
    elif pd.api.types.is_datetime64_any_dtype(c):
        return "Date"
    else:
        return "Other"

def meta(c):
    """
    Extract basic metadata about a pandas Series.

    Parameters:
    -----------
    c : pandas.Series
        A single column from a DataFrame.

    Returns:
    --------
    tuple
        A tuple containing:
        - dtype: data type of the Series
        - mem: memory usage in MB
        - uniq: number of unique values
        - nulls: number of null entries
        - empty: number of empty string entries
    """
    dtype = c.dtype
    mem = round(c.memory_usage(deep=True) / (1024 * 1024), 2)
    uniq = c.nunique()
    nulls = c.isnull().sum()
    empty = c.astype(str).str.strip().eq("").sum()
    return dtype, mem, uniq, nulls, empty

def stats(c):
    """
    Compute statistical summaries of a pandas Series.

    Parameters:
    -----------
    c : pandas.Series
        A single column from a DataFrame.

    Returns:
    --------
    tuple
        A tuple with statistical summaries:
        - mode_val: mode of the series
        - ent: entropy of the value distribution
        - maxent: maximum possible entropy
        - mean, std, min_val, max_val: basic statistics
        - skew, kurt: skewness and kurtosis
        - q1, q2, q3: 25th, 50th, and 75th percentiles
        - outliers: percentage of outlier values (based on IQR)
    """
    mode_val = c.mode()[0] if not c.mode().empty else "N/A"
    ent = round(entropy(c.value_counts(normalize=True), base=2), 2)
    maxent = round(np.log2(c.nunique()), 2) if c.nunique() > 0 else 0
    mean = std = min_val = max_val = skew = kurt = q1 = q2 = q3 = outliers = "N/A"

    if c.dtype.kind in "iufcM":
        mean = c.mean()
        std = c.std()
        min_val = c.min()
        max_val = c.max()
        q1 = c.quantile(0.25)
        q2 = c.quantile(0.5)
        q3 = c.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = round(((c < lower) | (c > upper)).sum() / len(c) * 100, 2)
        if c.dtype.kind in "iufc" and c.nunique() > 20:
            skew = round(c.skew(), 2)
            kurt = round(c.kurtosis(), 2)

    return mode_val, ent, maxent, mean, std, min_val, max_val, skew, kurt, q1, q2, q3, outliers

def DatasetMetadata(df, toPrint=True, Matrix=False):
    """
    Generate a detailed metadata and statistical summary for each column in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset to be summarized.
    toPrint : bool, default=True
        Whether to print a summary of the dataset structure and preview rows.
    Matrix : bool, default=False
        Whether to compute and print a Phik correlation matrix (requires `phik` library).

    Returns:
    --------
    pandas.DataFrame
        A summary table where each row corresponds to a column in `df` and includes:
        - Basic metadata: dtype, memory usage, cardinality, null/empty counts
        - Statistical summaries: mode, mean, std, min, max, quantiles, outliers, entropy
        - Variable type classification: Continuous, Categorical, Text, Date, Other
    """
    summary_data = []
    memUsage = list(df.memory_usage(deep=True) / (1024 * 1024))[1:]

    for c in df.columns:
        col = df[c]
        dtype, mem, uniq, nulls, empty = meta(col)
        mode_val, ent, maxent, mean, std, min_val, max_val, skew, kurt, q1, q2, q3, outliers = stats(col)
        vartype = datType(col)

        summary_data.append([
            c, str(dtype), mem, vartype, uniq, nulls, empty, mode_val,
            mean, std, min_val, max_val, skew, kurt, q1, q2, q3, outliers, ent, maxent
        ])

    SummaryDF = pd.DataFrame(summary_data, columns=[
        "CName", "DType", "UsageMB", "VarType", "Cardinality", "Null*", "Empty*", "Mode",
        "Mean", "Std", "Min", "Max", "Skewness", "Kurtosis", "Q1", "Q2/Median", "Q3",
        "Outliers %", "Entropy", "Max Entropy"
    ])

    if toPrint:
        print(f"The dataset includes {df.shape[1]} columns and {format(df.shape[0],',')} rows")
        print(f"Dataset has {df.duplicated().sum()} duplicate rows")
        print(f"Memory Usage of the DataFrame: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        print("--------------------------")
        print("First rows")
        print(df.head())
        print("--------------------------")

    if Matrix:
        print("Phik Correlation Matrix")
        numeric = [c for c in df.columns if df[c].dtype.kind in "iufc"]
        corr_matrix = df.phik_matrix(interval_cols=numeric)
        print(corr_matrix)

    return SummaryDF



def CheckGap(df,dateColumn,freq,showGaps=True):
  FullDate=pd.date_range(df[dateColumn].min(),df[dateColumn].max(),freq=freq)
  minDate=set(FullDate)-set(df[dateColumn])
  if showGaps:
    print(minDate)
  return f"The {dateColumn} has gap = {len(minDate)}"
  

def CheckRangePerColumn(df: pd.DataFrame, column: str, dateColumn: str, ToPrint=True):
    """
    Checks for missing hourly timestamps within each group of a specified column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing time series data.
    - column (str): Column name to group by (e.g., 'CountryCode').
    - dateColumn (str): Name of the datetime column.
    - ToPrint (bool): Whether to print details of missing timestamps per group.

    Prints the number of expected, actual, and missing hourly timestamps per group
    if any are missing.
    """
    df[dateColumn] = pd.to_datetime(df[dateColumn])
    full_range = pd.date_range(df[dateColumn].min(), df[dateColumn].max(), freq='h')
    expected_len = len(full_range)

    for c in df[column].unique():
        sub_df = df[df[column] == c]
        actual_dates = pd.to_datetime(sub_df[dateColumn]).dt.floor('h').unique()
        missing = sorted(set(full_range) - set(actual_dates))

        if missing and ToPrint:
            print(f"{c}: expected {expected_len}, actual {len(actual_dates)}, missing {len(missing)}")

def full_date_range(df: pd.DataFrame, date_col: str, country_code: str, freq='h'):
    """
    Generate a complete date range DataFrame for a given country and frequency, 
    filling in missing timestamps via a left join.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with at least a datetime column.
    date_col : str
        Name of the column containing datetime values.
    country_code : str
        Country identifier to be assigned to the output DataFrame.
    freq : str, optional (default='h')
        Frequency string (e.g., 'h' for hourly, 'd' for daily) for the date range.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with a continuous datetime range for the specified frequency,
        containing original data where available, and NaNs where data was missing.
        The 'CountryCode' is reassigned after the merge.
    """
    full_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=freq)
    full_range_df = pd.DataFrame({date_col: full_range})
    merged = pd.merge(full_range_df, df, on=date_col, how='left')
    merged['CountryCode'] = country_code  # re-assign after merge
    return merged.sort_values(by=date_col)

def Cstats(df: pd.DataFrame, column: str):
    """
    Compute basic statistics (min, max, mean, std) of the 'Value' column
    grouped by each unique value in the specified column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame that contains at least a 'Value' column and a grouping column.
    column : str
        Column name to group by (typically 'CountryCode').

    Returns:
    --------
    pd.DataFrame
        A summary DataFrame with one row per unique group, including:
        - Group name (CName)
        - Minimum of 'Value'
        - Maximum of 'Value'
        - Mean of 'Value'
        - Standard deviation of 'Value'
    """
    cname, maxx, minn, meann, stdd = [], [], [], [], []
    for c in df['CountryCode'].unique():
        cname.append(c)
        maxx.append(df[df[column] == c]['Value'].max())
        minn.append(df[df[column] == c]['Value'].min())
        meann.append(df[df[column] == c]['Value'].mean())
        stdd.append(df[df[column] == c]['Value'].std())

    return pd.DataFrame({"CName": cname, "Min": minn, "Max": maxx, "Mean": meann, "Std": stdd})


def InterDQ(df1: pd.DataFrame, df2: pd.DataFrame, groupColumn: str, valueColumn: str):
    """
    Compare statistical differences between two DataFrames for a specified column.

    Calculates percentage differences in mean, std, min, and max per group.

    Parameters:
    - df1: Original DataFrame
    - df2: Interpolated DataFrame
    - groupColumn: Grouping column name (e.g., 'CountryCode')
    - valueColumn: Target column name (e.g., 'Value')

    Returns:
    - DataFrame of percentage differences per group
    """
    stats1 = df1.groupby(groupColumn)[valueColumn].agg(['min', 'max', 'mean', 'std'])
    stats2 = df2.groupby(groupColumn)[valueColumn].agg(['min', 'max', 'mean', 'std'])

    merged = stats1.merge(stats2, on=groupColumn, suffixes=('_original', '_interpolated'))

    merged['mean_pct_diff'] = ((merged['mean_interpolated'] - merged['mean_original']) / merged['mean_original']).abs() * 100
    merged['std_pct_diff'] = ((merged['std_interpolated'] - merged['std_original']) / merged['std_original']).abs() * 100
    merged['min_pct_diff'] = ((merged['min_interpolated'] - merged['min_original']) / merged['min_original']).abs() * 100
    merged['max_pct_diff'] = ((merged['max_interpolated'] - merged['max_original']) / merged['max_original']).abs() * 100

    return merged[['mean_pct_diff', 'std_pct_diff', 'min_pct_diff', 'max_pct_diff']].round(3)


def duplicate_check(df: pd.DataFrame, columnToexclude: list, ToPrint: bool = True) -> str:
    """
    Check for duplicate records in a DataFrame based on a set of key columns,
    excluding the specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to check for duplicates.
    columnToexclude : list
        List of column names to exclude from the duplicate key check.
    ToPrint : bool, default=True
        If True, prints the duplicate rows and their counts.

    Returns:
    --------
    str
        A string message indicating the number of duplicated records found.
    """
    dup = df.groupby(df.columns.drop(columnToexclude).to_list()).size() \
            .reset_index(name='Count') \
            .query('Count > 1')


def dateTrans(df: pd.DataFrame, Dcolumn: str, ToPlot=True) -> pd.DataFrame:
    """
    Convert a datetime column into cyclical time features using sine and cosine transformation.

    This is useful when modeling periodic patterns (e.g., daily or hourly cycles) for time series or machine learning models.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing a datetime column.
    Dcolumn : str
        Name of the datetime column to be transformed.
    ToPlot : bool, default=True
        If True, plots a sample scatter of the sine and cosine features for visualization.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame where the original datetime column is replaced by two new columns:
        - 'day_sin': Sine-transformed timestamp (daily periodicity)
        - 'day_cos': Cosine-transformed timestamp (daily periodicity)
    """
    df_process = df.copy()
    
    # Convert datetime to timestamp in seconds
    timestamp_s = df_process[Dcolumn].map(pd.Timestamp.timestamp)
    
    # Number of seconds in a day
    day_seconds = 24 * 60 * 60

    # Apply sine and cosine transformation
    df_process["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day_seconds))
    df_process["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day_seconds))
    # df_process["week_sin"] = np.sin(timestamp_s * (2 * np.pi / week_seconds))
    # df_process["week_cos"] = np.cos(timestamp_s * (2 * np.pi / week_seconds))

    # Optional plot for visualization
    if ToPlot:
        df_process.sample(50).plot.scatter("day_sin", "day_cos").set_aspect("equal")

    return df_process.drop([Dcolumn], axis=1)



def split_by_country(data: pd.DataFrame, countryC: str, dateC: str) -> pd.DataFrame:
    """
    Split a time series DataFrame into training, validation, and test sets for each country.

    This function slices the data chronologically by predefined time ranges for each unique
    country code, and returns concatenated train/val/test sets across all countries.

    Time ranges:
    - Train: up to 2024-06-30 (18 months)
    - Validation: 2024-07-01 to 2024-12-31 (6 months)
    - Test: 2025-01-01 and onward (3 months in your context)

    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing country and datetime columns.
    countryC : str
        Name of the column containing country codes.
    dateC : str
        Name of the datetime column used for time-based splitting.

    Returns:
    --------
    tuple of pd.DataFrame
        (train, validation, test) DataFrames with data split by country and time.
    """
    train, vald, test = [], [], []

    for c in data[countryC].unique():
        sub_data = data[data[countryC] == c].copy()

        train_df = sub_data[sub_data[dateC] < pd.to_datetime('2024-07-01')].copy()
        valid_df = sub_data[
            (sub_data[dateC] >= pd.to_datetime('2024-07-01')) &
            (sub_data[dateC] < pd.to_datetime('2025-01-01'))
        ].copy()
        test_df = sub_data[sub_data[dateC] >= pd.to_datetime('2025-01-01')].copy()

        train.append(train_df)
        vald.append(valid_df)
        test.append(test_df)

    train = pd.concat(train).reset_index(drop=True)
    vald = pd.concat(vald).reset_index(drop=True)
    test = pd.concat(test).reset_index(drop=True)

    return train, vald, test


def split_by_country_check(data: pd.DataFrame, countryC: str, countryNr: int):
    """
    Perform a data quality check to ensure all countries in the dataset
    have the same number of records.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing time series data.
    countryC : str
        The name of the column containing country codes.
    countryNr : int
        The expected number of unique countries.

    Raises:
    -------
    ValueError
        If the number of unique country codes does not match `countryNr`.

    Prints:
    -------
    A message for each country indicating whether it has the expected number of records.
    """
    unique_countries = data[countryC].unique()

    if countryNr != len(unique_countries):
        raise ValueError("The number of country codes in the data does not match the expected count.")

    expected_rows_per_country = len(data) / countryNr
    counts = data[countryC].value_counts().to_dict()

    for country, count in counts.items():
        if count != expected_rows_per_country:
            print(f"{country} has {count} rows; expected {int(expected_rows_per_country)}.")
        else:
            print(f"{country} has the correct number of records: {int(count)}.")



def add_features_per_country(data:pd.DataFrame,countryC:str,dateC:str)->pd.DataFrame:    
    "add is_holiday and is_weekend columns"
    data['is_holiday'] = False  # initialize

    # Extract all years in your dataset
    years = data[dateC].dt.year.unique()

    # Loop through each country
    for c in data[countryC].unique():
        # Build holiday calendar for all relevant years
        cal = holidays.country_holidays(c, years=years)
        
        # Mask for that country
        mask = data[countryC] == c
        
        # Convert timestamp to date and check if in calendar
        data.loc[mask, 'is_holiday'] = data.loc[mask, dateC].dt.date.isin(cal)
    
    data["is_weekend"]=data[dateC].dt.weekday>=5

    data['is_holiday'] = data['is_holiday'].astype(int)
    data['is_weekend'] = data['is_weekend'].astype(int)

    return data


def sclaing_per_country(
        train:pd.DataFrame,
        vald:pd.DataFrame,
        test:pd.DataFrame,
        minmax_feature:list,
        robust_feature:list,
        group_c:str,
        target_c:str)->pd.DataFrame:  
    
    """
    Scales features per country for train/validation/test using appropriate scalers.

    Args:
        train, vald, test: pd.DataFrame
            Input splits 
        minmax_feature: list
            Features to scale with MinMaxScaler.
        robust_feature: list
            Features to scale with RobustScaler + MinMaxScaler.
        target_c: str
            Name of the target variable. Keep the target scaler separate for inverse scaling

    Returns:
        train_scaled, vald_scaled, test_scaled: pd.DataFrame
            Scaled data splits.
        scaler_target: dict
            Dictionary mapping country to fitted target scaler (for inverse transforms).
    """
    scaler_target={}
    scaled_train,scaled_val,scaled_test=[],[],[]

    for country,group in train.groupby(group_c):
        train_group=train[train[group_c]==country].copy()
        val_group=vald[vald[group_c]==country].copy()
        test_group=test[test[group_c]==country].copy()

        # MinMax features
        minmax_scaler=MinMaxScaler()
        train_group[minmax_feature]=minmax_scaler.fit_transform(train_group[minmax_feature])
        val_group[minmax_feature]=minmax_scaler.transform(val_group[minmax_feature])
        test_group[minmax_feature]=minmax_scaler.transform(test_group[minmax_feature])
        
        #Robust features removes outlier influence by centering on the median and scaling by IQR.
        robust = RobustScaler()
        train_solar_robust = robust.fit_transform(train_group[robust_feature])
        val_solar_robust = robust.transform(val_group[robust_feature])
        test_solar_robust = robust.transform(test_group[robust_feature])
        # normlize 
        minmax_solar = MinMaxScaler()
        train_group[robust_feature] = minmax_solar.fit_transform(train_solar_robust)
        val_group[robust_feature] = minmax_solar.transform(val_solar_robust)
        test_group[robust_feature] = minmax_solar.transform(test_solar_robust)

        #Target 
        target_scaler=MinMaxScaler()
        train_group[[target_c]]=target_scaler.fit_transform(train_group[[target_c]])
        val_group[[target_c]]=target_scaler.transform(val_group[[target_c]])
        test_group[[target_c]]=target_scaler.transform(test_group[[target_c]])
        # Used for inverse-transforming predictions
        scaler_target[country]=target_scaler
        
        scaled_train.append(train_group)
        scaled_val.append(val_group)
        scaled_test.append(test_group)
    
    # sort_index() used to restore the original chronological order after concatenating per-country splits
    train_scaled=pd.concat(scaled_train).sort_index()
    vald_scaled=pd.concat(scaled_val).sort_index()
    test_scaled=pd.concat(scaled_test).sort_index()

    return train_scaled,vald_scaled,test_scaled


def calculate_entropy(df: pd.DataFrame, groupC: str, valueC: str, bins: int = 10) -> pd.DataFrame:
    """
    Calculate entropy and max entropy for each group in the DataFrame based on a value column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing numerical values.
    - groupC (str): Column name to group by (e.g., 'CountryCode').
    - valueC (str): Name of the numeric column to compute entropy on (e.g., 'Value').
    - bins (int): Number of bins to discretize the value column.

    Returns:
    - pd.DataFrame: A DataFrame with columns: groupC, Entropy, MaxEntropy.
    """
    result = []

    for name, group in df.groupby(groupC):
        series = group[valueC].dropna()
        if len(series) == 0:
            ent, max_ent = 0.0, 0.0
        else:
            binned = pd.cut(series, bins=bins, labels=False)
            probs = binned.value_counts(normalize=True)
            ent = round(entropy(probs, base=2), 2)
            num_bins = probs.size
            max_ent = round(np.log2(num_bins) if num_bins > 0 else 0, 2)

        result.append({groupC: name, 'Entropy': ent, 'MaxEntropy': max_ent})

    return pd.DataFrame(result)