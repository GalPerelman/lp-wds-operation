import os
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype
import datetime


def is_col_datetime(x: pd.Series):
    if is_numeric_dtype(x):
        return False
    elif is_string_dtype(x):
        return pd.to_datetime(x, dayfirst=True)
    elif is_datetime64_any_dtype(x):
        return x


def load_from_csv(path, t1, t2):
    demands = pd.read_csv(path)
    demands['Datetime'] = pd.to_datetime(demands['Datetime'], dayfirst=True)
    demands = demands.loc[(demands['Datetime'] >= t1) & (demands['Datetime'] < t2)]
    if demands.empty:
        return
    else:
        return demands['Demand']


def patterns_to_dem(t1, t2, year_dem, pattern_zone, data_folder):
    days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    time_range = pd.date_range(start=t1, end=t2, freq='60min')
    if pattern_zone == 0:
        return pd.DataFrame(index=time_range, data={'demand': 0})['demand']
    else:
        month = pd.read_csv(data_folder + '/Demands/' + str(pattern_zone) + '_year-month.csv', encoding='windows-1255')
        month.columns = ['month', 'rate', 'cumulative']
        month['month'] = [i + 1 for i in range(12)]
        month.set_index('month', inplace=True)

        week = pd.read_csv(data_folder + '/Demands/' + str(pattern_zone) + '_week-day.csv', encoding='windows-1255')
        week.columns = ['weekday', 'rate', 'cumulative']
        week['weekday'] = [i + 1 for i in range(7)]
        week.set_index('weekday', inplace=True)

        hr = pd.read_csv(data_folder + '/Demands/' + str(pattern_zone) + '_day-hr.csv', encoding='windows-1255')
        hr.columns = ['hr'] + [i + 1 for i in range(7)]
        hr.set_index('hr', inplace=True)

        df = pd.DataFrame(index=time_range)
        df['month'] = pd.DatetimeIndex(df.index).month
        df['weekday'] = (pd.DatetimeIndex(df.index).weekday + 1) % 7 + 1
        df['hr'] = pd.DatetimeIndex(df.index).hour

        df['demand'] = df.apply(lambda x: year_dem * (month.loc[x['month'], 'rate'] / 100)
                                          * (7 / days_in_month[x['month']])
                                          * (week.loc[x['weekday'], 'rate'] / 100)
                                          * (hr.loc[x['hr'], x['weekday']] / 100), axis=1)

        df = df.loc[(df.index >= t1) & (df.index < t2)]
        return df['demand']


def patterns_to_dem2(t1, t2, year_dem, pattern_zone, data_folder):
    hourly_pattern_dict = {}
    days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    time_range = pd.date_range(start=t1, end=t2, freq='60min')
    if pattern_zone == 0:
        return pd.DataFrame(index=time_range, data={'demand': 0})['demand']
    else:
        dfMonth = pd.read_csv(data_folder + '/Demands/' + str(pattern_zone) + '/Patterns/Monthly_Demand_Pattern.dat',
                              index_col=None, header=None)
        dfMonth = dfMonth.T
        dfMonth = dfMonth.dropna()
        dfMonth.insert(loc=0, column='month', value=range(1, 13))
        dfMonth.set_index('month', inplace=True)

        dfDay = pd.DataFrame(index=range(12), columns=range(1, 8))
        for i in dfDay.index:
            d = (pd.read_csv(
                data_folder + '/Demands/' + str(pattern_zone) + '/Patterns/Daily_Demand_Pattern.' + str(i + 1),
                index_col=None, header=None).T.values)
            dfDay.iloc[i, :] = d
        dfDay.insert(loc=0, column='month', value=range(1, 13))
        dfDay.set_index('month', inplace=True)

        for i in range(12):
            hourly_pattern_dict[i + 1] = pd.read_csv(
                data_folder + '/Demands/' + str(pattern_zone) + '/Patterns/Hourly_Demand_Pattern.' + str(i + 1),
                index_col=None, header=None)
            hourly_pattern_dict[i + 1].columns = [i for i in range(1, 8)]

        df = pd.DataFrame(index=time_range)
        df['month'] = pd.DatetimeIndex(df.index).month
        df['weekday'] = (pd.DatetimeIndex(df.index).weekday + 1) % 7 + 1
        df['hr'] = pd.DatetimeIndex(df.index).hour

        df['demand'] = df.apply(lambda x: year_dem * (dfMonth.loc[x['month']] / 100)
                                          * (7 / days_in_month[x['month']])
                                          * (dfDay.loc[x['month'], x['weekday']] / 100)
                                          * (hourly_pattern_dict[x['month']].loc[x['hr'], x['weekday']] / 100), axis=1)

        return df['demand']


def demand_factorize(zone_id, df_demands, data_folder):
    dem_factor_path = os.path.join(data_folder, 'Demands', 'dem_factor.' + str(zone_id))

    if not os.path.exists(dem_factor_path):
        return df_demands
    else:
        factor = pd.read_csv(dem_factor_path, index_col=None).dropna()

    if factor.empty:
        return df_demands
    else:
        for index, row in factor.iterrows():
            start = datetime.datetime.strptime(row['start'], '%d/%m/%Y')
            end = datetime.datetime.strptime(row['end'], '%d/%m/%Y')
            f = row['factor']
            df_demands.loc[(df_demands.index >= start) & (df_demands.index < end), 'demand'] *= f
        return df_demands