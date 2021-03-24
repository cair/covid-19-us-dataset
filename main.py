import datetime
from functools import reduce

import gym
import numpy as np
import pandas as pd
from loguru import logger
import us_state_abbrev
import util
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
pd.options.mode.chained_assignment = None

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


# Simulation based of
# https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations
# . Vaccinations began on December 14, 2020.


class BaseEnvironment(gym.Env):

    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        # self.observation_space = gym.spaces.Box()

    def step(self, action):
        pass


class USCountry(BaseEnvironment):

    def __init__(self, date_from=datetime.datetime(2019, 1, 1), date_to=datetime.datetime.now()):
        super().__init__()

        self.date_from = date_from
        self.date_to = date_to

        self.population = self.load_population()
        self.dataset = self.load_data()


    def load_population(self):
        # https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2019/sc-est2019-agesex-civ.pdf
        # https://www2.census.gov/programs-surveys/popest/tables/2010-2019/state/asrh/sc-est2019-agesex-civ.csv
        population = util.download("us-residents", "https://www2.census.gov/programs-surveys/popest/tables/2010-2019/state/asrh/sc-est2019-agesex-civ.csv")
        population = population[population["SUMLEV"] == 40]
        population.drop('REGION', inplace=True, axis=1)
        population.drop('DIVISION', inplace=True, axis=1)
        population.drop('STATE', inplace=True, axis=1)
        population.drop('SUMLEV', inplace=True, axis=1)
        population.drop('ESTBASE2010_CIV', inplace=True, axis=1)

        for x in range(2010, 2019):
            population.drop(f'POPEST{x}_CIV', inplace=True, axis=1)

        #bins = pd.cut(population['AGE'], [-1, 18, 65, 200])
        #9 778 694

        rename_dict = {
            "POPEST2019_CIV": "population",
            "NAME": "state"
        }
        population = population.rename(columns=rename_dict)
        rename_dict.update({k: k.lower() for k in population.columns.values})
        population = population.rename(columns=rename_dict)

        population = population[population.age != 999]
        # example: sum of alabama population with all genders (0)
        # population[(population["name"] == "Alabama") & (population["sex"] == 0)]["population"].sum()
        return population


    def load_data(self):


        vaccinations = util.download("us-vaccinations",
                                     "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv")
        vaccinations = vaccinations.rename(
            columns={
                'location': 'state'
            }
        )

        vaccine_allocations_pfizer = util.download("us-vaccine-allocations-pfizer",
                                                   "https://data.cdc.gov/api/views/saz5-9hgg/rows.csv?accessType=DOWNLOAD")
        vaccine_allocations_pfizer = vaccine_allocations_pfizer.rename(
            columns={
                'Jurisdiction': "state",
                'Week of Allocations': 'date',
                '1st Dose Allocations': 'vaccine_alloc_1_pfizer',
                '2nd Dose Allocations': 'vaccine_alloc_2_pfizer'
            })

        vaccine_allocations_janssen = util.download("us-vaccine-allocations-janssen",
                                                    "https://data.cdc.gov/api/views/w9zu-fywh/rows.csv?accessType=DOWNLOAD")
        vaccine_allocations_janssen = vaccine_allocations_janssen.rename(
            columns={
                'Jurisdiction': "state",
                'Week of Allocations': 'date',
                '1st Dose Allocations': 'vaccine_alloc_1_janssen',
                '2nd Dose Allocations': 'vaccine_alloc_2_jannsen'
            })

        vaccine_allocations_moderna = util.download("us-vaccine-allocations-moderna",
                                                    "https://data.cdc.gov/api/views/b7pe-5nws/rows.csv?accessType=DOWNLOAD")
        vaccine_allocations_moderna = vaccine_allocations_moderna.rename(
            columns={
                'Jurisdiction': "state",
                'Week of Allocations': 'date',
                '1st Dose Allocations': 'vaccine_alloc_1_moderna',
                '2nd Dose Allocations': 'vaccine_alloc_2_moderna'
            })

        death_counts = util.download("us-death-counts-advanced",
                                     "https://data.cdc.gov/api/views/9bhg-hcku/rows.csv?accessType=DOWNLOAD")
        death_counts = death_counts.rename(columns={'State': "state", 'Data As Of': 'date'})

        cases_and_deaths = util.download("us-cases-death",
                                         "https://data.cdc.gov/api/views/9mfq-cb36/rows.csv?accessType=DOWNLOAD")
        cases_and_deaths = cases_and_deaths.rename(columns={'submission_date': "date"})

        cases_and_deaths = cases_and_deaths.replace({"state": us_state_abbrev.abbrev_us_state})
        cases_and_deaths = cases_and_deaths.replace({"state": us_state_abbrev.abbrev_us_state_3})

        # Datasets tend to name stuff differently. Above, we try to normalize state names.
        # A warning is output with states that does not match all dataset sources.
        # You can choose to ignore the warning, or fix the 'problem' (We remove these from the datasets)
        common_states, uncommon_states = util.count_common_states(
            vaccine_allocations_pfizer["state"].unique(),
            vaccine_allocations_janssen["state"].unique(),
            vaccine_allocations_moderna["state"].unique(),
            death_counts["state"].unique(),
            cases_and_deaths["state"].unique(),
            vaccinations["state"].unique()
        )
        logger.warning("The following states are excluded from the dataset: {}", uncommon_states)

        vaccine_allocations_pfizer, \
        vaccine_allocations_janssen, \
        vaccine_allocations_moderna, \
        death_counts, \
        cases_and_deaths, \
        vaccinations = util.remove_uncommon_states(uncommon_states, vaccine_allocations_pfizer,
                                                   vaccine_allocations_janssen,
                                                   vaccine_allocations_moderna,
                                                   death_counts,
                                                   cases_and_deaths,
                                                   vaccinations)
        assert len(vaccine_allocations_janssen["state"].unique()) == len(common_states), "State count not valid!"

        logger.info(vaccine_allocations_pfizer.columns.values)
        logger.info(vaccine_allocations_janssen.columns.values)
        logger.info(vaccine_allocations_moderna.columns.values)
        logger.info(death_counts.columns.values)
        logger.info(vaccinations.columns.values)
        logger.info(cases_and_deaths.columns.values)

        all_data_sources = [
            vaccine_allocations_pfizer,
            vaccine_allocations_janssen,
            vaccine_allocations_moderna,
            vaccinations,
            # death_counts,
            cases_and_deaths
        ]

        util.convert_datestring_to_datetime(*all_data_sources)

        combined_data = reduce(lambda left, right: pd.merge(left, right, on=['date', 'state'],
                                                            how='outer'), all_data_sources).sort_values(by="date")#.interpolate(method='linear', limit_direction='forward', axis=0)

        # Fill specific missing values
        util.fill_na_by_partial_name("vaccine_alloc", combined_data)

        # Interpolate where applicable
        #combined_data = combined_data.loc[combined_data["state"] == "Ohio"]
        combined_data = combined_data.interpolate()

        # Fill rest with 0
        combined_data = combined_data.fillna(0)

        combined_data = combined_data.set_index("date")

        # (Now you should quality check the dataset

        combined_data.to_csv("test.csv")

        print(len(vaccine_allocations_moderna["state"].unique()))
        print(len(cases_and_deaths["state"].unique()))

        # Limit between dates
        #mask_date = (combined_data['date'] > self.date_to) & (combined_data['date'] <= self.date_from)
        combined_data = combined_data.loc[self.date_from:self.date_to]

        return combined_data

    def dump_dataset(self, is_last=False, filenames=[]):
        postfix = "latest" if is_last else str(datetime.datetime.now())
        pop_name = f"population-{postfix}.csv"
        data_name = f"data-{postfix}.csv"
        filenames.extend([pop_name, data_name])

        self.population.to_csv(pop_name)
        self.dataset.to_csv(data_name)

        if not is_last:
            return self.dump_dataset(is_last=True, filenames=filenames)
        return filenames

class QAgent:

    def __init__(self):
        self.q = np.zeros(shape=(200, 200), dtype=np.float32)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-only", type=bool, default=True)
    parser.add_argument("--clean-dataset", type=bool, default=False)
    parser.add_argument('--date-from', default=datetime.datetime(2020, 5, 1), type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'))
    parser.add_argument('--date-to', default=datetime.datetime(2021, 3, 21), type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'))
    args = parser.parse_args()

    if args.clean_dataset:
        # Cleans the current data directory (makes a backup in case external sources goes away)
        util.FileUtils.backup_directory("data", "data")
        util.FileUtils.remove_directory("data")







    env = USCountry(
        date_from=args.date_from,
        date_to=args.date_to
    )

    if args.dump_only:
        from git import Repo
        repo = Repo()
        git = repo.git

        git.checkout(b='dataset')


        # Only dump current dataset'
        files = env.dump_dataset()

        for f in files:
            git.add(f)





        logger.info("Dataset is dumped. Closing.")
        exit(0)


    """
    ['New Jersey' 'Kansas' 'Illinois' 'Montana' 'Florida' 'Alaska' 'Kentucky'
     'Massachusetts' 'Delaware' 'Oklahoma' 'Tennessee' 'Alabama'
     'South Dakota' 'Vermont' 'Arkansas' 'California' 'Utah' 'Michigan'
     'Washington' 'Connecticut' 'Wisconsin' 'Rhode Island' 'Nebraska' 'Idaho'
     'New Mexico' 'Virginia' 'Missouri' 'Iowa' 'Louisiana' 'Minnesota'
     'Oregon' 'New Hampshire' 'North Dakota' 'Colorado' 'Puerto Rico'
     'South Carolina' 'Wyoming' 'West Virginia' 'Ohio' 'Georgia'
     'North Carolina' 'Arizona' 'Texas' 'Pennsylvania' 'District of Columbia'
     'Nevada' 'Hawaii' 'Mississippi' 'Indiana' 'Maryland' 'Maine']
    """
    state_name = "Texas"
    ohio = env.dataset[env.dataset["state"] == state_name]
    state_pop = env.population[(env.population["state"] == state_name) & (env.population["sex"] == 0)]["population"].sum()

    ohio["people_vaccinated_per_hundred"] /= 100.0
    ohio["tot_cases"] /= state_pop
    print(ohio["tot_cases"])
    ohio[["tot_cases", "people_vaccinated_per_hundred"]].plot(title=state_name)
    plt.savefig("test.png")






    #qagent = QAgent(shape=(30, 30))
