import os
import pandas as pd
import numpy as np
import datetime

pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 1000)


class Electricity:
    periods_season1 = None
    periods_season2 = None
    periods_season3 = None
    all_periods = None

    rates_season1 = None
    rates_season2 = None
    rates_season3 = None
    all_rates = None


def read_tariffs(data_folder):
    Electricity.periods_season1 = pd.read_csv(data_folder + '/electricity/TaozTypeWinter.csv', index_col=0)
    Electricity.periods_season2 = pd.read_csv(data_folder + '/electricity/TaozTypeSummer.csv', index_col=0)
    Electricity.periods_season3 = pd.read_csv(data_folder + '/electricity/TaozTypeFallSpring.csv', index_col=0)
    Electricity.all_periods = pd.concat(
        [Electricity.periods_season1, Electricity.periods_season3, Electricity.periods_season2],
        keys=['winter', 'fallspring', 'summer'])

    Electricity.rates_season1 = pd.read_csv(data_folder + '/electricity/TaozCostWinter.csv', index_col='name')
    Electricity.rates_season2 = pd.read_csv(data_folder + '/electricity/TaozCostSummer.csv', index_col='name')
    Electricity.rates_season3 = pd.read_csv(data_folder + '/electricity/TaozCostFallSpring.csv', index_col='name')
    Electricity.all_rates = pd.concat(
        [Electricity.rates_season1, Electricity.rates_season3, Electricity.rates_season2],
        keys=['winter', 'fallspring', 'summer'])


def get_special_dates(data_folder, years_list):
    holidays_evenings = []
    holidays = []
    for year in years_list:
        holidays_eve_file = os.path.join(data_folder, 'electricity', 'HolyDay2.' + str(year))
        holidays_evenings += pd.read_csv(holidays_eve_file, encoding='windows-1255').iloc[:, 1].tolist()

        holidays_file = os.path.join(data_folder, 'electricity', 'HolyDay1.' + str(year))
        holidays += pd.read_csv(holidays_file, encoding='windows-1255').iloc[:, 1].tolist()

    return holidays_evenings, holidays


def get_combs_for_unit(combs, unit_id):
    return combs[combs['unit ' + str(unit_id)] == 'ON']


def patterns_to_dem(T1, T2, year_dem, path):
    time_range = pd.date_range(start=T1, end=T2, freq='60min')

    month = pd.read_csv(path + '/Month-Year.csv', encoding='windows-1255')
    month.columns = ['month', 'rate', 'cumulative']
    month['month'] = [i + 1 for i in range(12)]
    month.set_index('month', inplace=True)

    week = pd.read_csv(path + '/Day-Week.csv', encoding='windows-1255')
    week.columns = ['weekday', 'rate', 'cumulative']
    week['weekday'] = [i + 1 for i in range(7)]
    week.set_index('weekday', inplace=True)

    hr = pd.read_csv(path + '/Hr-Day.csv', encoding='windows-1255')

    hr.columns = ['hr'] + [i + 1 for i in range(7)]
    hr.set_index('hr', inplace=True)

    df = pd.DataFrame(index=time_range)
    df['month'] = pd.DatetimeIndex(df.index).month
    df['weekday'] = (pd.DatetimeIndex(df.index).weekday + 1) % 7 + 1
    df['hr'] = pd.DatetimeIndex(df.index).hour

    df['demand'] = df.apply(lambda x: year_dem * (month.loc[x['month'], 'rate'] / 100)
                                      * (week.loc[x['weekday'], 'rate'] / 100)
                                      * (hr.loc[x['hr'], x['weekday']]), axis=1)

    return df['demand']


def vectorized_tariff(data_folder, date_range):
    read_tariffs(data_folder)
    df = pd.DataFrame(index=date_range)
    df['month'] = df.index.month
    df['weekday'] = (df.index.weekday + 1) % 7 + 1
    df['day'] = df.index.day
    df['hr'] = df.index.hour

    df['season'] = df['month'].map({1: 'winter', 2: 'winter',
                                    3: 'fallspring', 4: 'fallspring', 5: 'fallspring', 6: 'fallspring',
                                    7: 'summer', 8: 'summer',
                                    9: 'fallspring', 10: 'fallspring', 11: 'fallspring',
                                    12: 'winter'})

    years_list = date_range.year.unique()
    holidays_evenings_list, holidays_list = get_special_dates(data_folder, years_list)
    holidays_evenings_list = [d.strip('#') for d in holidays_evenings_list]
    holidays_list = [d.strip('#') for d in holidays_list]
    df.loc[df.index.to_series().dt.date.astype(str).isin(holidays_evenings_list), 'weekday'] = 6
    df.loc[df.index.to_series().dt.date.astype(str).isin(holidays_list), 'weekday'] = 7

    Tariffs = Electricity.all_periods
    Tariffs = Tariffs.stack().reorder_levels([0, 2, 1])
    Tariffs = Tariffs.rename('Tariff')
    Tariffs.index = [Tariffs.index.get_level_values(0), Tariffs.index.get_level_values(1).astype(int),
                     Tariffs.index.get_level_values(2).astype(int)]

    costs = Electricity.all_rates
    costs = costs.stack().reorder_levels([0, 2, 1])
    costs = costs.rename('costs')
    costs = costs.reset_index().rename(columns={'level_0': 'season', 'level_1': 'voltage'})
    costs = costs.pivot_table(values='costs', index=['season', 'name'], columns='voltage',
                              aggfunc='first').reset_index()

    df = pd.merge(df, Tariffs, how='inner', left_on=['season', 'weekday', 'hr'], right_index=True)
    df = df.sort_index()

    df = pd.merge(df.reset_index(), costs, how='inner', left_on=['season', 'Tariff'], right_on=['season', 'name'])
    df.index = df['index']
    df = df.drop(['index', 'Tariff'], axis=1)
    df = df.sort_index()
    return df


def split_range(date_tariff, n):
    df = date_tariff
    df.index.name = 'time'
    df['group'] = (df['name'].ne(df['name'].shift()) | (df['day'].ne(df['day'].shift()))).cumsum()
    df = df.groupby('group')
    temp = pd.DataFrame()
    for group, data in df:
        data.reset_index(inplace=True)
        data = data.groupby(data.index // n).first()
        temp = pd.concat([temp, data])
    temp['step_size'] = temp['time'].diff().shift(-1)
    temp.index = temp['time']
    return temp


def separate_consecutive_tariffs(df):
    df['group'] = df['tariff'].ne(df['tariff'].shift()).cumsum()
    df['auxiliary'] = df['group'].diff().replace({0: np.nan})
    df.iloc[[0, 0], df.columns.get_loc('auxiliary')] = 1
    df['auxiliary'] = df['auxiliary'].ffill() + df.groupby(df['auxiliary'].notnull().cumsum()).cumcount()
    df['eps'] = ((df['auxiliary'] - 1) / len(df.index.get_level_values('comb').drop_duplicates())).astype(int)
    df = df.drop('auxiliary', axis=1)
    return df


def date_parser(x):
    return datetime.datetime.strptime(x, '%d/%m/%Y %H:%M')


def get_hours_between_timestamps(t1, t2):
    if t1 > t2:
        raise Exception('t2 must be greater than t1')
    delta = t2 - t1
    days, sec = delta.days, delta.seconds
    return days * 24 + sec // 3600


class EnergySupplier:
    """ Future development """

    def __init__(self, name, data_path):
        self.name = name
        self.data_path = data_path

        self.tariff = self.read_tariff()

    def read_tariff(self):
        df = pd.read_csv(self.data_path, index_col='time')
        return df

    def tariff_vector(self, time_range):
        df = pd.DataFrame(index=time_range)
        df['hour'] = df.index % 24
        df['day'] = (df.index / 24).astype(int)
        df = pd.merge(df, self.tariff, left_on='hour', right_index=True, how='left')
        return df
