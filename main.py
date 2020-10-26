# Python code to extract Google Health Trends data from API using Pew Search Sampler code

# pip install search_sampler
# from searcher import SearchSampler
import datetime
import random

from searcher import SearchSampler
import os.path
from os import path
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import plotly.graph_objects as go
import math
import scipy.stats
from scipy.interpolate import BSpline, make_interp_spline, interpolate
from pytrends.request import TrendReq
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# 1. SET PARAMETERS

# 1a. Google Health Trends API key
apikey = 'AIzaSyCfSZJnyW37oJy6dUzMZuONWwN77v8Mtbw'

# 1b. output path
output_path = '/Users/elimichaels/Box Sync/Dissertation /PAPER 0/Data/Google/Raw Google Trends data'

# 1c. number of samples: this tells search sampler how many samples to take. following Pew protocol and setting to 50.

num_samples = 50

# 1d. geography arrays
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands': 'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
# list of all state codes so search_sampler will be able to run script on all states.
# can only pull 4 states each day
states_mast_list = [
    ['US-AL', 'US-AK', 'US-AZ', 'US-AR'],  # 2020-05-10 (12:56pm)
    ['US-CA', 'US-CO', 'US-CT', 'US-DC'],  # 2020-05-10 (12:56pm)
    ['US-DE', 'US-FL', 'US-GA', 'US-HI'],  # 2020-05-11 (1:23pm)
    ['US-ID', 'US-IL', 'US-IN', 'US-IA'],  # 2020-05-11 (1:23pm)
    ['US-KS', 'US-KY', 'US-LA', 'US-ME'],  # 2020-05-12 (1:55pm)
    ['US-MD', 'US-MA', 'US-MI'],  # 2020-05-12 (1:55pm)
    ['US-MN', 'US-MS', 'US-MO', 'US-MT'],  # 2020-05-12 (1:55pm)
    ['US-NE', 'US-NV', 'US-NH', 'US-NJ'],  # 2020-05-13 (1:57pm)
    ['US-NM', 'US-NY', 'US-NC', 'US-ND'],  # 2020-05-13 (1:57pm)
    ['US-OH', 'US-OK', 'US-OR', 'US-PA'],  # 2020-05-14 (3pm)
    ['US-RI', 'US-SC', 'US-SD', 'US-TN'],  # 2020-05-14 (9pm)
    ['US-TX', 'US-UT', 'US-VT', 'US-VA'],  # 2020-05-15 (6pm)
    ['US-WA', 'US-WV', 'US-WI', 'US-WY']  # 2020-05-17 (10am)
]
flatten = lambda l: [item for sublist in l for item in sublist]

deaths = {'aa': datetime.datetime(2020, 2, 23), "bt": datetime.datetime(2020, 5, 17),
          "gf": datetime.datetime(2020, 5, 25)}

state_codes = flatten(states_mast_list)
# list of all DMA codes
dma_mast_list = [
    [698, 686, 522, 630],  # 2020-05-17 (10am)
    [711, 524, 606, 691],  # 2020-05-18 (3pm)
    [673, 743, 745, 747],  # 2020-05-18 (3pm)
    [790, 789, 753, 771],  # 2020-05-19 (3pm)
    [693, 628, 619, 670],  # 2020-05-19 (3pm)
    [647, 734, 612, 640],  # 2020-05-21 (9am)
    [807, 811, 862, 868],  # 2020-05-21 (9am)
    [802, 866, 803, 800],  # 2020-05-22 (10am)
    [828, 825, 855, 813],  # 2020-05-22 (10am)
    [751, 752, 773, 501],  # 2020-05-23 (11am)
    [533, 504, 576, 511],  # 2020-05-23 (11am)
    [592, 561, 656, 534],  # 2020-05-24 (12pm)
    [528, 571, 539, 530],  # 2020-05-24 (12pm)
    [548, 507, 525, 503],  # 2020-05-25 (6:30pm)
    [520, 575, 567, 744],  # 2020-05-25 (6:30pm)
    [757, 758, 770, 881],  # 2020-05-27 (7:30am)
    [760, 717, 632, 609],  # 2020-05-27 (7:30am)
    [610, 682, 648, 581],  # 2020-05-28 (9am)
    [602, 649, 675, 509],  # 2020-05-28 (9am)
    [527, 582, 529, 515],  # 2020-05-29 (10am)
    [588, 542, 679, 637],  # 2020-05-29 (10am)
    [624, 652, 611, 631],  # 2020-05-30 (1pm)
    [725, 603, 616, 678],  # 2020-05-30 (2pm)
    [605, 671, 638, 722],  # 2020-05-31 (7pm)
    [659, 541, 736, 557],  # 2020-05-31 (9:30pm)
    [564, 531, 642, 643],  # 2020-06-02 (8:10am)
    [716, 644, 622, 500],  # 2020-06-02 (8:35am)
    [552, 537, 512, 508],  # 2020-06-04 (9:43am)
    [506, 532, 521, 543],  # 2020-06-04 (10:03am)
    [583, 553, 563, 540],  # 2020-06-05 (12:22pm)
    [513, 551, 676, 505],  # 2020-06-05 (12:22pm)
    [547, 658, 613, 724],  # 2020-06-06 (4pm)
    [737, 702, 718, 710],  # 2020-06-06 (4pm)
    [746, 604, 754, 756],  # 2020-06-07 (5:30pm)
    [755, 766, 764, 687],  # 2020-06-07 (6pm)
    [798, 762, 740, 759],  # 2020-06-08 (7:21pm)
    [839, 523, 634, 765],  # 2020-06-08 (7:40pm)
    [514, 502, 555, 565],  # 2020-06-09 (7:40pm)
    [526, 549, 538, 518],  # 2020-06-09 (9pm)
    [517, 545, 550, 544],  # 2020-06-10 (8:37pm)
    [560, 570, 558, 510],  # 2020-06-10 (9:31pm)
    [535, 554, 536, 596],  # 2020-06-12 (7:40am)
    [597, 650, 657, 627],  # 2020-06-12 (8:30am)
    [820, 801, 821, 810],  # 2020-06-14 (8:10am)
    [566, 574, 577, 516],  # 2020-06-14 (8:10am)
    [519, 546, 639, 623],  # 2020-06-15 (8:10am)
    [633, 709, 600, 641],  # 2020-06-15 (8:20am)
    [618, 651, 635, 625],  # 2020-06-16 (10:30am)
    [662, 636, 661, 692],  # 2020-06-16 (11:50am)
    [626, 749, 584, 573],  # 2020-06-17 (10:40am)
    [556, 569, 559, 819],  # 2020-06-17 (11:36am)
    [598, 705, 669, 617],
    [767]]

region_codes = {
    "South": ['10', '11', '12', '13', '24', '37', '45', '51', '54', '1', '21', '28', '47', '5', '22', '40', '48'],
    "North East": ['42', '34', '9', '44', '25', '33', '50', '23', '36'],
    "Mid West": ['38', '46', '31', '20', '27', '19', '29', '55', '17', '26', '18', '39'],
    "West": ['53', '41', '6', '16', '32', '30', '56', '49', '4', '8', '35', '2', '15']}

abv_code_name = [['AK', '02', 'ALASKA'], ['MS', '28', 'MISSISSIPPI'], ['AL', '01', 'ALABAMA'], ['MT', '30', 'MONTANA'],
                 ['AR', '05', 'ARKANSAS'], ['NC', '37', 'NORTH CAROLINA'], ['AS', '60', 'AMERICAN SAMOA'],
                 ['ND', '38', 'NORTH DAKOTA'], ['AZ', '04', 'ARIZONA'], ['NE', '31', 'NEBRASKA'],
                 ['CA', '06', 'CALIFORNIA'], ['NH', '33', 'NEW HAMPSHIRE'], ['CO', '08', 'COLORADO'],
                 ['NJ', '34', 'NEW JERSEY'], ['CT', '09', 'CONNECTICUT'], ['NM', '35', 'NEW MEXICO'],
                 ['DC', '11', 'DISTRICT OF COLUMBIA'], ['NV', '32', 'NEVADA'], ['DE', '10', 'DELAWARE'],
                 ['NY', '36', 'NEW YORK'], ['FL', '12', 'FLORIDA'], ['OH', '39', 'OHIO'], ['GA', '13', 'GEORGIA'],
                 ['OK', '40', 'OKLAHOMA'], ['GU', '66', 'GUAM'], ['OR', '41', 'OREGON'], ['HI', '15', 'HAWAII'],
                 ['PA', '42', 'PENNSYLVANIA'], ['IA', '19', 'IOWA'], ['PR', '72', 'PUERTO RICO'], ['ID', '16', 'IDAHO'],
                 ['RI', '44', 'RHODE ISLAND'], ['IL', '17', 'ILLINOIS'], ['SC', '45', 'SOUTH CAROLINA'],
                 ['IN', '18', 'INDIANA'], ['SD', '46', 'SOUTH DAKOTA'], ['KS', '20', 'KANSAS'],
                 ['TN', '47', 'TENNESSEE'], ['KY', '21', 'KENTUCKY'], ['TX', '48', 'TEXAS'], ['LA', '22', 'LOUISIANA'],
                 ['UT', '49', 'UTAH'], ['MA', '25', 'MASSACHUSETTS'], ['VA', '51', 'VIRGINIA'],
                 ['MD', '24', 'MARYLAND'], ['VI', '78', 'VIRGIN ISLANDS'], ['ME', '23', 'MAINE'],
                 ['VT', '50', 'VERMONT'], ['MI', '26', 'MICHIGAN'], ['WA', '53', 'WASHINGTON'],
                 ['MN', '27', 'MINNESOTA'], ['WI', '55', 'WISCONSIN'], ['MO', '29', 'MISSOURI'],
                 ['WV', '54', 'WEST VIRGINIA'], ['WY', '56', 'WYOMING']]


# 2. FUNCTIONS

# function for changing df format from long to wide and renaming some columns
def reformatFile(df):
    # use pandas built in pivot to change general format from long to wide
    df = df.pivot(index='timestamp', columns='sample', values='value')
    # rename columns by appending "sample_" as a prefix to each
    df.columns = ["sample_" + str(col) for col in df.columns]
    # create column name to hold each period
    df.index.name = "period"
    # move period values from index list to actual column inside df
    df.reset_index(inplace=True)
    # convert from date_time to just date
    df['period'] = df['period'].dt.date
    return df


# function taking in list of region codes and path to save data to
def query(filePath, params=None, regionCodes="", terms=[""], convert=False,
          i_num_samples=-1):  # we added this code so search_sampler will run on all states
    # for each region in the input regionCodes list passed as a parameter
    if i_num_samples == -1:
        i_num_samples = num_samples
    for region in regionCodes:
        # Print the name of the region for verbosity
        print("running search for following region: ", region)
        # search params
        if params is None:
            params = {
                # Can be any number of search terms, using boolean logic. See report methodology for more info.
                'search_term': terms,

                # Can be country, state, or DMA. States are US-CA. DMA are a 3 digit code; see Nielsen for info.
                'region': region,

                # Must be in format YYYY-MM-DD # just testing for a year
                'period_start': '2004-01-01',
                'period_end': '2020-05-01',

                # Options are day, week, month. WARNING: This has been extensively tested with week only.
                'period_length': 'month'
            }
        params['region'] = region

        sample = SearchSampler(apikey, terms[0], params, output_path=filePath)

        # GETTING DATA

        # rolling set of samples:
        df_results = sample.pull_rolling_window(num_samples=i_num_samples)
        df_results = reformatFile(df_results)
        if convert:
            sample.params['region'] = str(sample.params['region'])
        sample.save_file(df_results, append=True)


# 3. CALL THE FUNCTION

# 3a. STATES
# call query for regionCodes = states
'''query(regionCodes=states, filePath=output_path+"/states")
'''

# 3b. DMAs
# Note: using crontab we set this script to be run every day at 10am
# This function checks to see if folder for a particular DMA exists
def no_file(dma, is_dma=True, search_path=""):
    str_dma = str(dma)

    if search_path == "":
        search_path = output_path

    if (is_dma):
        exist = path.exists(search_path + "/dmas/" + str_dma)
    else:
        exist = path.exists(search_path + "/states/" + str_dma)
    return not exist


def run_query_on_next_states(folder, is_dma=False, terms=[], time_start="", time_end="", window="week", params=None):
    if not os.path.exists(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)

    found_state = False

    if params == None:
        params = {
            # Can be any number of search terms, using boolean logic. See report methodology for more info.
            'search_term': terms,

            # Can be country, state, or DMA. States are US-CA. DMA are a 3 digit code; see Nielsen for info.
            'region': "replace_me",

            # Must be in format YYYY-MM-DD # just testing for a year
            'period_start': time_start,
            'period_end': time_end,

            # Options are day, week, month. WARNING: This has been extensively tested with week only.
            'period_length': window
        }
    if is_dma:
        master_list = dma_mast_list
    else:
        master_list = states_mast_list

    for chunk in master_list:
        # check if a folder exists for the results from the first code in the the set
        already_done = path.exists("{}/{}".format(folder, chunk[-1]))
        if already_done:
            print("already did: {}".format(chunk[-1]))
        if not already_done:
            # if no folder already exists, rune query on set of 4 dma codes
            query(filePath=folder, params=params, regionCodes=chunk, i_num_samples=50)
            # end iteraterating over dma set list
            found_state = True
            break
    return found_state


def old_runs():
    # set the function
    eli_wants_run = True

    # # 3b. call query for regionCodes = DMAs
    # if (eli_wants_run):
    #     foundState = False
    #     # for states in states_mast_list:
    #     #     # check if a folder exists for the results from the first code in the the set
    #     #     if (no_file(states[0], False)):
    #     #         # if no folder already exists, rune query on set of 4 dma codes
    #     #         query(filePath=output_path + "/states", regionCodes=states, terms = ["nigger + niggers"], convert=True)
    #     #         # end iteraterating over dma set list
    #     #         foundState = True
    #     #         break
    #     # # for each set of 4 dma codes
    #     if (not foundState):
    #         for dmas in dma_mast_list:
    #             # check if a folder exists for the results from the first code in the the set
    #             if (no_file(dmas[0])):
    #                 # if no folder already exists, rune query on set of 4 dma codes
    #                 query( filePath=output_path + "/dmas", regionCodes=dmas, terms = ["nigger + niggers"], convert=True)
    #                 # end iteraterating over dma set list
    #                 break


def test():
    dates = [('2009-01-01', '2009-05-01'), ('2009-01-08', '2009-05-08'), ('2009-01-15', '2009-05-15'),
             ('2009-01-22', '2009-05-22')]
    params = {
        # Can be any number of search terms, using boolean logic. See report methodology for more info.
        'search_term': ['test + terms'],

        # Can be country, state, or DMA. States are US-CA. DMA are a 3 digit code; see Nielsen for info.
        'region': 698,

        # Must be in format YYYY-MM-DD # just testing for a year
        'period_start': '2020-05-04',
        'period_end': '2020-05-10',

        # Options are day, week, month. WARNING: This has been extensively tested with week only.
        'period_length': 'week'
    }

    sample = SearchSampler(apikey, "test_query", params, output_path=output_path + "/test")

    # GETTING DATA

    # rolling set of samples:
    df_results = sample.pull_rolling_window(5)
    print(df_results)
    df_results = reformatFile(df_results)


def weekly_states_names_test():
    pass


def combine(folder):
    df_super_list = []
    names = state_codes
    for name in names:
        df = pd.read_csv(os.path.join(folder, name + ".csv"))

        renames = df.columns[2:]
        renames = {sub: sub.split(":")[0] for sub in renames}

        df = df.rename(columns=renames)

        name_col = [name for i in range(df.shape[0])]
        df['State'] = name_col

        cols = ['Week', 'State', 'George Floyd', 'Breonna Taylor', 'Ahmaud Arbery']
        df = df[cols]

        df_super_list.append(df)

    uber_df = pd.concat(df_super_list)

    return uber_df


def get_val(prefix, week, state, all_dfs_in, fails_in):
    if prefix + " " + state not in all_dfs_in.keys():
        path = "final_data/{}/states/{}/combined/{}-_combined.csv".format(prefix, state, state)
        if os.path.exists(path):
            new_df = pd.read_csv(path)
        else:
            fails_in[path] = True
            return "nan"
        new_df.set_index("period_date", inplace=True)
        all_dfs_in[prefix + " " + state] = new_df

    focus_df = all_dfs_in[prefix + " " + state]
    if week in focus_df.index:
        val = focus_df.loc[week].iloc[0][-1]
        # if math.isnan(val):
        #     return 0
        return val
    else:
        return 0


def combo_table():
    fails = {}
    all_dfs = {}
    df = pd.read_csv("combined.csv")
    rel_cols = df.columns[-3:]
    df = df.rename(columns={col: col + " (trends)" for col in rel_cols})
    new_aa = []
    new_bt = []
    new_gf = []
    focus_df = None
    for index, row in df.iterrows():
        week = row['Week']
        state = row['State']
        new_aa.append(get_val("aa", "Ahmaud Arbery", week, state, all_dfs, fails))
        new_bt.append(get_val("bt", "Breonna Taylor", week, state, all_dfs, fails))
        new_gf.append(get_val("gf", "George Floyd", week, state, all_dfs, fails))

    df['Ahmaud Arbery (health trend)'] = new_aa
    df['Breonna Taylor (health trend)'] = new_bt
    df['George Floyd (health trend)'] = new_gf

    for fail in fails.keys():
        print(fail)
    "Data/George Floyd/states/US-AL/combined/US-AL-_combined.csv"
    "Data/George Floyd/states/US-AL/combined/US-AL-_combined.csv"
    df.to_csv("final_table.csv")
    print(rel_cols)


def combo_from_scratch(weeks, names, states):
    fails = {}
    all_dfs = {}
    n_weeks = len(weeks)
    n_states = len(states)
    new_weeks = [weeks[int(i / n_states)] for i in range(n_states * n_weeks)]
    new_states = [states[i % n_states] for i in range(n_states * n_weeks)]
    dat = {'Week': new_weeks, 'State': new_states}
    df = pd.DataFrame(dat)
    new_dfs = {name: [] for name in names}
    for index, row in df.iterrows():
        week = row['Week']
        state = row['State']
        for name in new_dfs.keys():
            new_dfs[name].append(get_val(name, week, state, all_dfs, fails))

    for name in new_dfs.keys():
        df[name] = new_dfs[name]

    print(df.shape)

    df.to_csv("final_table.csv")

    print(df.shape)

def check_na_in_table(df):
    new_df = df.pivot(index="period_date", columns="variable", values="value")
    for index, row in new_df.iterrows():
        if math.isnan(row['imp_value']):
            if row['avg_value'] != 0:
                return False
    return True


def check_na_is_zero_recur(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isdir(path):
            if check_na_is_zero_recur(path) == False:
                return False
        else:
            if file.split(".")[-1] == "csv":
                if "combined" in file:
                    df = pd.read_csv(path)
                    if check_na_in_table(df) == False:
                        print("failed on {}".format(path))
                        return False
    return True


def get_clean_uber(normal=True, f_name="uberTable2.csv", fillna = False):
    df = pd.read_csv(f_name)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for s in df.columns[2:]:
        if "gt" in s:
            df.drop(s, axis=1, inplace=True)
            continue
        df[s] = df[s].replace("<1", "0")
        df[s] = df[s].astype(float)

        if (normal):
            df[s] = (df[s] - df[s].min()) / (df[s].max() - df[s].min())
    if (fillna):
        df = df.fillna(value=0)
    df.Week = pd.to_datetime(df.Week)
    df = df.sort_values(by='Week')
    return df


def by_region_graph():
    df = get_clean_uber(False, "final_table.csv", True)
    regions = region_codes.keys()
    abv_to_code = {a[0]: a[1] for a in abv_code_name}
    inv_region = {}
    for region in regions:
        for sub in region_codes[region]:
            inv_region[sub.zfill(2)] = region
    colors = {"AA": "blue", "BT": "orange", "GF": "green"}
    mapping = {"AA": "Ahmaud Arbery", "BT": "Breonna Taylor", "GF": "George Floyd"}
    pairs = np.asarray([['AA_gt', 'AA_ght'], ['BT_gt', 'BT_ght'], ['GF_gt', 'GF_ght']])
    types = ["Google Trends", "Google Health Trends"]
    people_names = ["AA", "GF", "BT"]

    df['Region'] = [inv_region[abv_to_code[name.split("-")[1]]] for name in df['State']]
    names = ['South', 'North_East', 'Mid_West', 'West']

    South = df[df.Region == "South"].groupby(['Week']).mean()
    North_East = df[df.Region == "North East"].groupby(['Week']).mean()
    Mid_West = df[df.Region == "Mid West"].groupby(['Week']).mean()
    West = df[df.Region == "West"].groupby(['Week']).mean()

    data = [South, North_East, Mid_West, West]

    for i in range(4):
        data[i].to_csv("{}.csv".format(names[i]))
        data[i].reset_index(level=0, inplace=True)

    deaths = {"Ahmaud Arbery": datetime.datetime(2020, 2, 23), "Breonna Taylor": datetime.datetime(2020, 5, 17),
              "George Floyd": datetime.datetime(2020, 5, 25)}

    deaths_mapping = {"AA": 1 , "BT": 13, "GF": 14}
    week = datetime.timedelta(days=6)
    index = [x.strftime("%m/%d") + "-" + (x + week).strftime("%m/%d") for x in South.Week]
    line_type = {"AA": "dotted", "BT": "solid", "GF": "dashed"}

    a = [[0, 0], [0, 0]]
    dems = (2, 2)
    fig = plt.figure(0)

    a[0][0] = plt.subplot2grid(dems, (0, 0), colspan=1)
    plt.ylabel("Interest")
    plt.xlabel("Week")
    plt.xticks(np.arange(0, 500, 1), rotation=45, ha='right')
    a[0][1] = plt.subplot2grid(dems, (0, 1), colspan=1)
    plt.ylabel("Interest")
    plt.xlabel("Week")
    plt.xticks(np.arange(0, 500, 1), rotation=45, ha='right')
    a[1][0] = plt.subplot2grid(dems, (1, 0), colspan=1)
    plt.ylabel("Interest")
    plt.xlabel("Week")
    plt.xticks(np.arange(0, 500, 1), rotation=45, ha='right')
    a[1][1] = plt.subplot2grid(dems, (1, 1), colspan=1)
    plt.ylabel("Interest")
    plt.xlabel("Week")
    plt.xticks(np.arange(0, 500, 1), rotation=45, ha='right')


    for i in range(2):
        for j in range(2):
            thickness = 2

            sub_i = 2 * i + j
            for p in people_names:
                a[i][j].plot(index, data[sub_i][p], label=mapping[p],
                             color=colors[p], linewidth=thickness, linestyle=line_type[p])
                a[i][j].axvline(x=deaths_mapping[p], linewidth=thickness, color=colors[p],
                                linestyle=line_type[p])
                a[i][j].annotate("Killed On {}".format(deaths[mapping[p]].strftime("%m/%d")),
                                 xy=(deaths_mapping[p], 46652 * 0.4), rotation=-90)
                max = np.argmax(data[sub_i][p].values, axis=0)

                a[i][j].annotate(index[max], xy=(index[max], data[sub_i][p][max] * 0.95))

            a[i][j].set_title(names[sub_i])
            a[i][j].legend(loc="upper left")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.size'] = 12
    plt.show()



def graphs():
    df = get_clean_uber(False)
    colors = {"AA": "blue", "BT": "orange", "GF": "green"}
    mapping = {"AA": "Ahmaud Arbery", "BT": "Breonna Taylor", "GF": "George Floyd"}
    pairs = np.asarray([['AA_gt', 'AA_ght'], ['BT_gt', 'BT_ght'], ['GF_gt', 'GF_ght']])
    types = ["Google Trends", "Google Health Trends"]
    type_mapping = {"gt": types[0], "ght": types[1]}

    by_week = df.groupby(['Week']).mean()

    for a in by_week[by_week.columns[2:]]:
        print(a)
        print(by_week[a].describe())

    by_week.to_csv("by_week.csv")

    by_week.reset_index(level=0, inplace=True)
    week = datetime.timedelta(days=6)
    index = [x.strftime("%m/%d") + "-" + (x + week).strftime("%m/%d") for x in by_week.Week]

    a = [[0, 0, 0], [0, 0]]
    all_min = np.array([by_week[x].min() for x in by_week.columns[2:]]).min()
    all_max = np.array([by_week[x].max() for x in by_week.columns[2:]]).max()
    plt.figure(0)

    for i in range(2):
        for j in range(3):
            if i == 1 and j == 2:
                break
            if i == 0:
                a[i][j] = plt.subplot2grid((2, 6), (i, j * 2), colspan=2)
                plt.ylim(all_min, all_max)  # set the ylim to bottom, top
            else:
                a[i][j] = plt.subplot2grid((2, 6), (i, j * 3), colspan=3)
                if j != 0:
                    plt.ylim(all_min, all_max)  # set the ylim to bottom, top
            plt.ylabel("Interest")
            plt.xlabel("Week of")
            plt.xticks(range(len(index)), index, rotation=45, ha='right')
            a[i][j].xaxis.grid()  # vertical lines

    p_index = 1

    for i in range(3):
        pair = pairs[i]
        data = by_week[pair[p_index]]
        a[0][i].plot(index, data, label=type_mapping[pair[p_index].split("_")[1]],
                     color=colors[pair[p_index].split("_")[0]])
        a[0][i].set_title(mapping[pair[p_index].split("_")[0]])
        max = np.argmax(data.values, axis=0)
        a[0][i].annotate(index[max], xy=(index[max], data[max] * 0.95))

    for i in range(2, 2):
        for j in range(3):
            a[1][i].plot(index, by_week[pairs[j][i]], label=mapping[pairs[j][i].split("_")[0]])
            max = np.argmax(by_week[pairs[j][i]].values, axis=0)
            a[1][i].annotate(index[max], xy=(index[max], by_week[pairs[j][i]][max] * 0.95))
        a[1][i].set_title(type_mapping[pairs[0, i].split("_")[1]])
        a[1][i].legend(loc="upper left")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.show()


if __name__ == "__main__":
    #
    # weeks = ['2020-02-16', '2020-02-23', '2020-03-01', '2020-03-08', '2020-03-15', '2020-03-22', '2020-03-29', '2020-04-05', '2020-04-12', '2020-04-19', '2020-04-26', '2020-05-03', '2020-05-10', '2020-05-17', '2020-05-24', '2020-05-31', '2020-06-07', '2020-06-14', '2020-06-21', '2020-06-28', '2020-07-05', '2020-07-12', '2020-07-19', '2020-07-26', '2020-08-02', '2020-08-09', '2020-08-16', '2020-08-23', '2020-08-30', '2020-09-06', '2020-09-13', '2020-09-20', '2020-09-27']
    # names = ["AA", "GF", "BT"]
    # states = state_codes
    # combo_from_scratch(weeks, names, states)
    #
    # df = get_clean_uber(False, "final_table.csv")







    # run_query_on_next_states("Data/George Floyd/states", terms = "George Floyd", time_start= '2020-02-15', time_end= '2020-08-15')
    # graphs()
    # #test()
    # by_region_graph()
    #
    # df = get_clean_uber(False)
    # states = df.State.unique()
    # by_state = {a: df[df.State == a].groupby(['Week']).mean() for a in df.State.unique()}
    # wide = 10
    # tall = 5
    #
    # all_min = df['GF_ght'].min()
    # all_max = df['GF_ght'].max()
    #
    # grid = [[0 for i in range(tall)] for i in range(wide)]
    # plt.figure(0)
    #
    # for i in range(wide):
    #     for j in range(tall):
    #         grid[i][j] = plt.subplot2grid((wide, tall), (i, j), colspan=1)
    #         if (i + wide * j < len(states)):
    #             data = by_state[states[i + wide * j]]
    #             data.reset_index(level=0, inplace=True)
    #             for k in range(3):
    #                 col = data.columns[k * 2 + 2]
    #                 grid[i][j].plot(data.Week, data[col], label=col)
    #             grid[i][j].set_title(states[i + tall * j])
    #             grid[i][j].xaxis.set_ticklabels([])
    #             plt.ylim(all_min, all_max)  # set the ylim to bottom, top

    # names = {"Ahmaud Arbery","Breonna Taylor","George Floyd"}
    #
    # while (True):
    #     for name in names:
    #         run_query_on_next_states("current_pulls/{}/states".format(name), is_dma=False, terms = [name], time_start= '2020-02-15', time_end= '2020-10-12')

    # params = {
    #     # Can be any number of search terms, using boolean logic. See report methodology for more info.
    #     'search_term': ["George Floyd"],
    #
    #     # Can be country, state, or DMA. States are US-CA. DMA are a 3 digit code; see Nielsen for info.
    #     'region': "replace_me",
    #
    #     # Must be in format YYYY-MM-DD # just testing for a year
    #     'period_start': '2020-02-15',
    #     'period_end': '2020-09-30',
    #
    #     # Options are day, week, month. WARNING: This has been extensively tested with week only.
    #     'period_length': 'week'
    # }
    #
    # query("second run", regionCodes=["US"], i_num_samples=50, params=params)
    # params['search_term'] =  ["Ahmaud Arbery"]
    # query("second run", regionCodes = ["US"], i_num_samples = 50, params = params)
    # params['search_term'] =  ["Breonna Taylor"]
    # query("second run", regionCodes = ["US"],  i_num_samples = 50, params = params)

    df = get_clean_uber(False, "final_table.csv", True)
    df = df.fillna(value=0)

    df = pd.DataFrame()
    names = []
    folder = "newest_data"
    for f in os.listdir(folder):
        if "Google_Data_Appendix_100620" not in f:
            sub_df = pd.read_csv(os.path.join(folder, f))
            df['Week'] = sub_df['period_date']
            df[f.replace(".csv", '').split("_")[0]] = sub_df[sub_df.columns[-1]]

    colors = {"AA": "blue", "BT": "orange", "GF": "green"}
    mapping = {"Ahmaud Arbery": "AA", "Breonna Taylor": "BT", "George Floyd": "GF"}
    line_type = {"Ahmaud Arbery": "dotted", "Breonna Taylor": "solid", "George Floyd": "dashed"}
    fig = plt.figure()
    ax = plt.axes()
    df.Week = pd.to_datetime(df.Week)
    week = datetime.timedelta(days=6)
    index = [x.strftime("%m/%d") + "-" + (x + week).strftime("%m/%d") for x in df.Week]
    plt.xticks(np.arange(0, 500, 1), rotation=45, ha='right')

    deaths = {"Ahmaud Arbery": datetime.datetime(2020, 2, 23), "Breonna Taylor": datetime.datetime(2020, 5, 17),
              "George Floyd": datetime.datetime(2020, 5, 25)}

    deaths_mapping = {"Ahmaud Arbery": 1 , "Breonna Taylor": 13, "George Floyd": 14}

    thickness = 2
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.size'] = 12
    for c in df.columns[1:4]:
        print(df[c].describe())

        col = df[c].fillna(0)
        ax.plot(index, df[c], label="{}".format(c), linestyle=line_type[c], color=colors[mapping[c]], linewidth=thickness)
        max = np.argmax(col.values, axis=0)
        ax.annotate(index[max], xy=(index[max], df[c][max] * 0.95))
        ax.axvline(x=deaths_mapping[c], linewidth=thickness, color=colors[mapping[c]], linestyle=line_type[c])
        ax.annotate("Killed On {}".format(deaths[c].strftime("%m/%d")), xy=(deaths_mapping[c], 46652 * 0.6), rotation=-90)


    plt.legend(loc="upper left")
    plt.ylabel("Interest")
    plt.xlabel("Week of")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    plt.show()



    df = get_clean_uber(False, "final_table.csv", True)
    regions = region_codes.keys()
    abv_to_code = {a[0]: a[1] for a in abv_code_name}
    inv_region = {}
    for region in regions:
        for sub in region_codes[region]:
            inv_region[sub.zfill(2)] = region
    colors = {"AA": "blue", "BT": "orange", "GF": "green"}
    mapping = {"AA": "Ahmaud Arbery", "BT": "Breonna Taylor", "GF": "George Floyd"}
    pairs = np.asarray([['AA_gt', 'AA_ght'], ['BT_gt', 'BT_ght'], ['GF_gt', 'GF_ght']])
    types = ["Google Trends", "Google Health Trends"]
    people_names = ["AA", "GF", "BT"]

    df['Region'] = [inv_region[abv_to_code[name.split("-")[1]]] for name in df['State']]
    names = ['South', 'North_East', 'Mid_West', 'West']

    South = df[df.Region == "South"].groupby(['Week']).mean()
    North_East = df[df.Region == "North East"].groupby(['Week']).mean()
    Mid_West = df[df.Region == "Mid West"].groupby(['Week']).mean()
    West = df[df.Region == "West"].groupby(['Week']).mean()

    data = [South, North_East, Mid_West, West]

    for i in range(4):
        data[i].to_csv("{}.csv".format(names[i]))
        data[i].reset_index(level=0, inplace=True)

    for i in range(4):
        df = data[i]
        fig = plt.figure(figsize=(20, 10))
        ax = plt.axes()
        plt.xticks(np.arange(0, 500, 1), rotation=45, ha='right')
        plt.title(names[i])
        thickness = 2
        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.size'] = 12
        for c in df.columns[1:4]:
            col = df[c].fillna(0)
            ax.plot(index, df[c], label="{}".format(c), linestyle=line_type[mapping[c]], color=colors[c],
                    linewidth=thickness)
            max = np.argmax(col.values, axis=0)
            ax.annotate(index[max], xy=(index[max], df[c][max] * 0.95))
            ax.axvline(x=deaths_mapping[mapping[c]], linewidth=thickness, color=colors[c], linestyle=line_type[mapping[c]])
            ax.annotate("Killed On {}".format(deaths[mapping[c]].strftime("%m/%d")), xy=(deaths_mapping[mapping[c]], 46652 * 0.6),
                        rotation=-90)

        plt.legend(loc="upper left")
        plt.ylabel("Interest")
        plt.xlabel("Week of")
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.tight_layout()
        plt.savefig('{}.png'.format(names[i]))
        # plt.show()
