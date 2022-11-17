import os
import pandas as pd
import numpy as np
import datetime
import json
from itertools import chain, combinations

pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 1000)


class elec_tariff:
    TaozTypeWinter = None
    TaozTypeSummer = None
    TaozTypeFallSpring = None
    Tariffs = None

    TaozCostWinter = None
    TaozCostSummer = None
    TaozCostFallSpring = None
    costs = None


def read_tariffs(data_folder):
    elec_tariff.TaozTypeWinter = pd.read_csv(data_folder + '/electricity/TaozTypeWinter.csv', index_col=0)
    elec_tariff.TaozTypeSummer = pd.read_csv(data_folder + '/electricity/TaozTypeSummer.csv', index_col=0)
    elec_tariff.TaozTypeFallSpring = pd.read_csv(data_folder + '/electricity/TaozTypeFallSpring.csv', index_col=0)
    elec_tariff.Tariffs = pd.concat(
        [elec_tariff.TaozTypeWinter, elec_tariff.TaozTypeFallSpring, elec_tariff.TaozTypeSummer],
        keys=['winter', 'fallspring', 'summer'])

    elec_tariff.TaozCostWinter = pd.read_csv(data_folder + '/electricity/TaozCostWinter.csv', index_col='name')
    elec_tariff.TaozCostSummer = pd.read_csv(data_folder + '/electricity/TaozCostSummer.csv', index_col='name')
    elec_tariff.TaozCostFallSpring = pd.read_csv(data_folder + '/electricity/TaozCostFallSpring.csv', index_col='name')
    elec_tariff.costs = pd.concat(
        [elec_tariff.TaozCostWinter, elec_tariff.TaozCostFallSpring, elec_tariff.TaozCostSummer],
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


def vectorize_tariff(data_folder, date_range):
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

    # Tariffs = pd.concat([TaozTypeWinter, TaozTypeFallSpring, TaozTypeSummer], keys=['winter', 'fallspring', 'summer'])
    Tariffs = elec_tariff.Tariffs
    Tariffs = Tariffs.stack().reorder_levels([0, 2, 1])
    Tariffs = Tariffs.rename('Tariff')
    Tariffs.index = [Tariffs.index.get_level_values(0), Tariffs.index.get_level_values(1).astype(int),
                     Tariffs.index.get_level_values(2).astype(int)]

    # costs = pd.concat([TaozCostWinter, TaozCostFallSpring, TaozCostSummer], keys=['winter', 'fallspring', 'summer'])
    costs = elec_tariff.costs
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



if __name__ == '__main__':
    T1 = datetime.datetime(2020, 3, 2, 0, 00)
    T2 = datetime.datetime(2021, 3, 3, 0, 00)

    e = EnergySupplier('CB', 'data/Richmond/electricity/CB.csv')
    t1 = 0
    t2 = 47
    time_range = pd.Series(np.arange(start=t1, stop=t2 + 1, step=1))
    tariff = e.tariff_vector(time_range)
    tariff = split_range(tariff, 3)
    print(tariff)

    # tr = pd.date_range(start=T1, periods=24, freq="1H")
    # tariff = vectorize_tariff('data/Richmond', tr)
    # print(tariff)

    # read_tariffs('data/Richmond')
    # t = (elec_tariff.Tariffs)
    # print(t.stack().reorder_levels([0, 2, 1]))
