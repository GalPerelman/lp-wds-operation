import os
import pandas as pd

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
    Electricity.periods_season1 = pd.read_csv(data_folder + '/electricity/periods_season1.csv', index_col=0)
    Electricity.periods_season2 = pd.read_csv(data_folder + '/electricity/periods_season2.csv', index_col=0)
    Electricity.periods_season3 = pd.read_csv(data_folder + '/electricity/periods_season3.csv', index_col=0)
    Electricity.all_periods = pd.concat(
        [Electricity.periods_season1, Electricity.periods_season3, Electricity.periods_season2],
        keys=['1', '2', '3'])

    Electricity.rates_season1 = pd.read_csv(data_folder + '/electricity/rates_season1.csv', index_col='name')
    Electricity.rates_season2 = pd.read_csv(data_folder + '/electricity/rates_season2.csv', index_col='name')
    Electricity.rates_season3 = pd.read_csv(data_folder + '/electricity/rates_season3.csv', index_col='name')
    Electricity.all_rates = pd.concat(
        [Electricity.rates_season1, Electricity.rates_season3, Electricity.rates_season2],
        keys=['1', '2', '3'])


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


def vectorized_tariff(data_folder, date_range):
    read_tariffs(data_folder)
    df = pd.DataFrame(index=date_range)
    df['month'] = df.index.month
    df['weekday'] = (df.index.weekday + 1) % 7 + 1
    df['day'] = df.index.day
    df['hr'] = df.index.hour

    df['season'] = df['month'].map({1: '1', 2: '1', 3: '2', 4: '2', 5: '2', 6: '2', 7: '3', 8: '3', 9: '2', 10: '2',
                                    11: '2', 12: '1'})

    years_list = date_range.year.unique()
    holidays_evenings_list, holidays_list = get_special_dates(data_folder, years_list)
    holidays_evenings_list = [d.strip('#') for d in holidays_evenings_list]
    holidays_list = [d.strip('#') for d in holidays_list]
    df.loc[df.index.to_series().dt.date.astype(str).isin(holidays_evenings_list), 'weekday'] = 6
    df.loc[df.index.to_series().dt.date.astype(str).isin(holidays_list), 'weekday'] = 7

    tariffs = Electricity.all_periods
    tariffs = tariffs.stack().reorder_levels([0, 2, 1])
    tariffs = tariffs.rename('Tariff')
    tariffs.index = [tariffs.index.get_level_values(0), tariffs.index.get_level_values(1).astype(int),
                     tariffs.index.get_level_values(2).astype(int)]

    costs = Electricity.all_rates
    costs = costs.stack().reorder_levels([0, 2, 1])
    costs = costs.rename('costs')
    costs = costs.reset_index().rename(columns={'level_0': 'season', 'level_1': 'voltage'})
    costs = costs.pivot_table(values='costs', index=['season', 'name'], columns='voltage',
                              aggfunc='first').reset_index()

    df = pd.merge(df, tariffs, how='inner', left_on=['season', 'weekday', 'hr'], right_index=True)
    df = df.sort_index()

    df = pd.merge(df.reset_index(), costs, how='inner', left_on=['season', 'Tariff'], right_on=['season', 'name'])
    df.index = df['index']
    df = df.drop(['index', 'Tariff'], axis=1)
    df = df.sort_index()

    return df


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
