import os
import io
import sys
import gzip
import re
import sqlite3
import dbm
from glob import glob
from datetime import datetime, date, timedelta
from pprint import pprint
from math import nan, inf, pi as π, e
import math
from random import seed, choice, randint, sample
from contextlib import contextmanager
from collections import namedtuple
from collections import Counter
from itertools import islice
from textwrap import fill
from dataclasses import dataclass, astuple, asdict, fields
import json
from jsonschema import validate, ValidationError
import simplejson
import requests
import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
import dask
import psycopg2
from sqlalchemy import create_engine

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV, RFE

from pymongo import MongoClient

from IPython.display import Image as Show
import nltk

# Might use tokenization/stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Do not display warnings
import warnings
warnings.simplefilter('ignore')

# Only show 8 rows from large DataFrame
pd.options.display.max_rows = 8
pd.options.display.min_rows = 8

# A bit of setup for monochrome book; not needed for most work
monochrome = (cycler('color', ['k', '0.5']) * 
              cycler('linestyle', ['-', '-.', ':']))
plt.rcParams['axes.prop_cycle'] = monochrome
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600


@contextmanager
def show_more_rows(new=sys.maxsize):
    old_max = pd.options.display.max_rows
    old_min = pd.options.display.min_rows
    try:
        pd.set_option("display.max_rows", new)
        pd.set_option('display.min_rows', new)
        yield old_max
    finally:
        pd.options.display.max_rows = old_max
        pd.options.display.min_rows = old_min


# PostgreSQL configuration
def connect_local():
    user = 'cleaning'
    pwd = 'data'
    host = 'localhost'
    port = '5432'  
    db = 'dirty'
    con = psycopg2.connect(database=db, host=host, user=user, password=pwd)
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{db}')
    return con, engine

# MongoDB conection
def connect_mongo():
    client = MongoClient(port=27017)
    return client


# Utility function
def random_phone(reserved=True):
    digits = '0123456789'
    area = '000'
    while area.startswith('0'):
        area = ''.join(sample(digits, 3))
    # DO NOT REMOVE prefix code
    # it is not used now, but random seed assumes
    # the exact same sequence of calls
    prefix = '000'
    while prefix.startswith('0'):
        prefix = ''.join(sample(digits, 3))
    suffix = ''.join(sample(digits, 4))
    #-------------------------------------
    if reserved:
        prefix = "555"
    return f"+1 {area} {prefix} {suffix}"


# Make the "business" database
def make_mongo_biz(client=connect_mongo()):
    # Remove any existing DB and recreate it
    client.drop_database('business')
    db = client.business

    # Create sample data
    words = [
        'Kitchen', 'Tasty', 'Big', 'City', 'Fish',
        'Delight', 'Goat', 'Salty', 'Sweet']
    title = ['Inc.', 'Restaurant', 'Take-Out']
    cuisine = [
        'Pizza', 'Sandwich', 'Italian', 'Mexican',
        'American', 'Sushi', 'Vegetarian']
    prices = ['cheap', 'reasonable', 'expensive']

    seed(2)
    info = {}
    for n in range(50):
        # Make a random business
        name = (f"{choice(words)} {choice(words)} "
                f"{choice(title)}")
        info[name] = random_phone()
        biz = {
            'name': name,
            'cuisine': choice(cuisine),
            'phone': info[name]
        }
        db.info.insert_one(biz)
        
    for n in range(5000):
        # Make a random review
        name = choice(list(info))
        price = choice(prices)
        review = {'name': name, 'price': price}
        
        # Usually have a rating
        if (n+5) % 100:
            review['rating'] = randint(1, 10)
            
        # Occasionally denormalize
        if not (n+100) % 137:
            review['phone'] = info[name]
            # But sometimes inconsistently
            if not n % 5:
                review['phone'] = random_phone()
            
        # Insert business into MongoDB 
        result = db.reviews.insert_one(review)

    print('Created 50 restaurants; 5000 reviews') 
    

def make_dbm_biz(client=connect_mongo()):
    "We assume that make_mongo_biz() has run"
    biz = client.business
    ratings = dict()
    
    with dbm.open('data/keyval.db', 'n') as db:
        db['DESCRIPTION'] = 'Restaurant information'
        now = datetime.isoformat(datetime.now())
        db['LAST-UPDATE'] = now

        # Add reviews
        q = biz.reviews.find()
        for n, review in enumerate(q):
            key = f"{review['name']}::ratings"
            # Occasionally use unusual delimiter
            if not (n+50) % 100:
                key = key.replace("::", "//")
            # First rating or apppend to list?
            rating = str(review.get('rating', '')) 
            if key in ratings:
                old = ratings[key]
                val = f"{old};{rating}"
            else:
                val = rating 
            db[key] = ratings[key] = val
        
        # Add business info
        for n, info in enumerate(biz.info.find()):
            key1 = f"{info['name']}::info::phone"
            key2 = f"{info['name']}::info::cuisine"
            db[key1] = info['phone']
            db[key2] = info['cuisine']


def pprint_json(jstr):
    from json import dumps, loads
    print(dumps(loads(jstr),indent=2))

    
def print_err(err):
    print(err.__class__.__name__)
    print(fill(str(err)))


def not_valid(instance, schema):
    try:
        return validate(instance, schema)
    except ValidationError as err:
        return str(err) 


def make_missing_pg():
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS missing")
    cur.execute("CREATE TABLE missing (a REAL, b CHAR(10))")
    cur.execute("INSERT INTO missing(a, b) VALUES ('NaN', 'Not number')")
    cur.execute("INSERT INTO missing(a, b) VALUES (1.23, 'A number')")
    cur.execute("INSERT INTO missing(a, b) VALUES (NULL, 'A null')")
    cur.execute("INSERT INTO missing(a, b) VALUES (3.45, 'Santiago')")
    cur.execute("INSERT INTO missing(a, b) VALUES (6.78, '')")    
    cur.execute("INSERT INTO missing(a, b) VALUES (9.01, NULL)")
    con.commit()
    cur.execute("SELECT * FROM missing")
    return cur.fetchall()


def make_dask_data():
    df = dask.datasets.timeseries()
    if not os.path.exists('data/dask'):
        os.mkdir('data/dask')
    df.to_csv('data/dask/*.csv', 
              name_function=lambda i: (
                  str(date(2000, 1, 1) + 
                     i * timedelta(days=1)))
             )


def dask_to_postgres():
    "Put some data into postgres.  Assumes Dask data was generated"
    out = io.StringIO()
    df = pd.read_csv('data/dask/2000-01-02.csv', parse_dates=['timestamp'])
    df = df.loc[3456:5678]
    df.to_sql('dask_sample', engine, if_exists='replace')
    sql = """
        ALTER TABLE dask_sample
        ALTER COLUMN id TYPE smallint,
        ALTER COLUMN name TYPE char(10),
        ALTER COLUMN x TYPE decimal(6, 3),
        ALTER COLUMN y TYPE real,
        ALTER COLUMN index TYPE integer;
        """
    cur = con.cursor()
    cur.execute(sql)
    cur.execute('COMMIT;')
    describe = """
        SELECT column_name, data_type, numeric_precision, character_maximum_length
        FROM information_schema.columns 
        WHERE table_name='dask_sample';"""
    cur.execute(describe)
    for tup in cur:
        print(f"{tup[0]}: {tup[1]} ({tup[2] or tup[3]})", file=out)
    cur.execute("SELECT count(*) FROM dask_sample")
    print("Rows:", cur.fetchone()[0], file=out)
    return out.getvalue()


def make_bad_amtrak():
    "Create deliberately truncated data"
    out = io.StringIO()
    df = pd.read_csv('data/AMTRAK-Stations-Database_2012.csv')
    df = df[['Code', 'StationName', 'City', 'State']]
    df['Visitors'] = np.random.randint(0, 35_000, 973)
    df['StationName'] = df.StationName.str.split(', ', expand=True)[0]
    df['StationName'] = df.StationName.str[:20]
    df['Visitors'] = np.clip(df.Visitors, 0, 2**15-1)
    df.to_sql('bad_amtrak', engine, if_exists='replace', index=False)
    sql = """
        ALTER TABLE bad_amtrak
        ALTER COLUMN "StationName" TYPE char(20),
        ALTER COLUMN "Visitors" TYPE smallint;    
        """
    cur = con.cursor()
    cur.execute(sql)
    cur.execute('COMMIT;')
    describe = """
        SELECT column_name, data_type, numeric_precision, character_maximum_length
        FROM information_schema.columns 
        WHERE table_name='bad_amtrak';"""
    cur.execute(describe)
    for tup in cur:
        print(f"{tup[0]}: {tup[1]} ({tup[2] or tup[3]})", file=out)
    cur.execute("SELECT count(*) FROM bad_amtrak")
    print("Rows:", cur.fetchone()[0], file=out)
    return out.getvalue() 


def make_h5_hierarchy():
    import h5py
    np.random.seed(1)  # Make the notebook generate same random data
    f = h5py.File('data/hierarchy.h5', 'w')
    f.create_dataset('/deeply/nested/group/my_data', 
                     (10, 10, 10, 10), dtype='i')
    f.create_dataset('deeply/path/elsewhere/other', (20,), dtype='i')
    f.create_dataset('deeply/path/that_data', (5, 5), dtype='f')
    dset = f['/deeply/nested/group/my_data']
    np.random.seed(seed=1)
    dset[...] = np.random.randint(-99, 99, (10, 10, 10, 10))
    dset.attrs['author'] = 'David Mertz'
    dset.attrs['citation'] = 'Cleaning Data Book'
    dset.attrs['shape_type'] = '4-D integer array'
    f.close()

    
def show_boxplots(df, cols, whis=1.5):
    # Create as many horizontal plots as we have columns
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2*len(cols)))
    # For each one, plot the non-null data inside it
    for n, col in enumerate(cols):
        data = df[col][df[col].notnull()]
        axes[n].set_title(f'{col} Distribution')
        # Extend whiskers to specified IQR multiplier
        axes[n].boxplot(data, whis=whis, vert=False, sym='x')
        axes[n].set_yticks([])
    # Fix spacing of subplots at the end
    fig.tight_layout()
    plt.savefig(f"img/boxplot-{'_'.join(cols)}.png")
    
    
def make_corrupt_digits():
    from sklearn.datasets import load_digits
    import random as r
    r.seed(1)
    digits = load_digits().images[:50]
    for digit in digits:
        for _ in range(3):
            x, y = r.randint(0, 7), r.randint(0, 7)
            digit[x, y] = -1
    np.save('data/digits.npy', digits.astype(np.int8))

# Load the digit data into namespace
digits = np.load('data/digits.npy')


def show_digits(digits=digits, x=3, y=3, title="Digits"):
    "Display of 'corrupted numerals'"
    if digits.min() >= 0:
        newcm = cm.get_cmap('Greys', 17)
    else:
        gray = cm.get_cmap('Greys', 18)
        newcolors = gray(np.linspace(0, 1, 18))
        newcolors[:1, :] = np.array([1.0, 0.9, 0.9, 1])
        newcm = ListedColormap(newcolors)

    fig, axes = plt.subplots(x, y, figsize=(x*2.5, y*2.5),
                      subplot_kw={'xticks':(), 'yticks': ()})
    
    for ax, img in zip(axes.ravel(), digits):
        ax.imshow(img, cmap=newcm)
        for i in range(8):
            for j in range(8):
                if img[i, j] == -1:
                    s = "╳"
                    c = "k"
                else:
                    s = str(img[i, j])
                    c = "k" if img[i, j] < 8 else "w"
                text = ax.text(j, i, s, color=c,
                               ha="center", va="center")
    fig.suptitle(title, y=0)
    fig.tight_layout()    
    plt.savefig(f'img/{title}.png')
                

kryptonite = pd.read_fwf('data/excited-kryptonite.fwf')

def plot_kryptonite(df=kryptonite, 
                    independent='Wavelength_nm',
                    logx=True,
                    imputed=False):
    fig, ax = plt.subplots(figsize=(10,3))

    (df[df.Kryptonite_type == "Green"]
        .plot(kind='scatter', color="green", marker="o", s=20,
              x=independent, y='candela_per_m2', 
              logx=logx, ax=ax, label="Green"))
    (df[df.Kryptonite_type == "Gold"]
        .plot(kind='scatter', color="goldenrod", marker="s", s=30,
              x=independent, y='candela_per_m2', 
              logx=logx, ax=ax, label="Gold"))
    (df[df.Kryptonite_type == "Red"]
        .plot(kind='scatter', color="darkred", marker="x", s=25,
              x=independent, y='candela_per_m2', 
              logx=logx, ax=ax, label="Red"))
    ax.legend()
    title = f"Luminance response of kryptonite types by {independent}"
    if imputed:
        title = f"{title} (imputed)"
    ax.set_title(title)
    plt.savefig(f"img/{title}.png")

DTI = pd.DatetimeIndex
date_series = pd.Series(
            [-10, 1, 2, np.nan, 4], 
            index=DTI(['2001-01-01',
                       '2001-01-05',
                       '2001-01-10',
                       '2001-02-01',
                       '2001-02-05']))


def plot_filled_trend(s=None):
    n = len(s)
    line = pd.Series(np.linspace(s[0], s[-1], n), 
                     index=s.index)
    filled = s.fillna(pd.Series(line))
    plt.plot(range(n), filled.values, 'o', range(n), line, ":")
    plt.grid(axis='y', color='lightgray', linewidth=0.5)
    plt.xticks(range(n), 
               labels=['Actual', 'Actual',
                       'Actual', 'Imputed (Feb 1)',
                       'Actual'])
    plt.plot([3], [3.69], 'x', 
             label="Time interpolated value")
    plt.legend()
    title = "Global imputation from linear trend"
    plt.title(title)
    plt.savefig(f"img/{title}.png")
    

philly = 'data/philly_houses.json'
def make_philly_missing(fname=philly):
    np.random.seed(1)
    out = fname.replace('houses', 'missing')
    rows = json.load(open(fname))['rows']
    blank = np.random.randint(0, len(rows), 1000)
    for num in blank:
        rows[num]['market_value'] = np.nan
    json.dump(rows, open(out, 'w'))
    return rows


def make_cars(fname='data/cars/car.data'):
    cars = pd.read_csv(fname, header=None, 
                       names=["price_buy",
                              "price_maintain",
                              "doors", 
                              "passengers", 
                              "trunk",
                              "safety", 
                              "rating"])
    price = {'vhigh': 3, 'high': 2, 
             'med': 1, 'low': 0}
    cars['price_buy'] = cars.price_buy.map(price)
    cars['price_maintain'] = cars.price_maintain.map(price)
    cars['doors'] = cars.doors.map(
                        {'2': 2, '3': 3,
                         '4': 4, '5more': 5})
    cars['passengers'] = cars.passengers.map(
                        {'2': 2, '4': 4, 
                         'more': 6})
    cars['trunk'] = cars.trunk.map(
                        {'small': 0, 'med': 1,
                         'big': 2})
    cars['safety'] = cars.safety.map(
                        {'low': 0, 'med': 1, 
                         'high': 2})
    cars['rating'] = cars.rating.map(
                        {'unacc': "Unacceptable",
                         'acc': "Acceptable",
                         'good': "Good", 
                         'vgood': "Very Good"})
    cars = cars.sample(frac=1, random_state=1)
    cars.to_csv('data/cars.csv', index=False)


def make_hair_lengths(df):
    assert len(df) == 25_000
    np.random.seed(1)
    s = np.random.beta(a=1.1, b=5, size=25000) * 147
    negate = np.random.randint(0, 25_000, 100)
    s[negate] *= -1
    neg_one = np.random.randint(0, 25_000, 20)
    s[neg_one] = -1
    zero = np.random.randint(0, 25_000, 500)
    s[zero] = 0
    s = np.round(s, 1)
    df['Hair_Length'] = s
    plt.hist(s, 50, density=True)
    return df


def add_typos(df):
    from random import random, randrange
    fnames = list(df.Name)
    for n, name in enumerate(fnames):
        r = random()
        letters = list(name)
        pos = randrange(0, len(name))
        if r < 0.002:   # Delete a letter
            print("Deleting letter from", name)
            del letters[pos]
            fnames[n] = ''.join(letters)
        elif r < 0.004:  # Add a letter ('e')
            print("Adding letter to", name)
            letters.insert(pos, 'e')
            fnames[n] = ''.join(letters)
        elif r < 0.006:   # Swap adjacent letters
            print("Swapping letters in", name)
            letters[pos], letters[pos-1] = letters[pos-1], letters[pos]
            fnames[n] = ''.join(letters)
            
    df['Name'] = fnames


# Make a synthetic collection of names/ages/other
# based on the actual SSA name frequency
def make_ssa_synthetic(fname='data/Baby-Names-SSA.csv'):
    # Repeatable random
    import random
    random.seed(1)

    # Date for age calculation
    now = 2020

    # Population adjustment by year in 20Ms
    population = np.linspace(5, 16, 100)
    years = np.linspace(1919, 2018, 100, dtype=int)
    year_to_pop = dict(zip(years, population))

    # Rank to count
    rank_to_freq = {'1': 1.0, '2': 0.9, '3': 0.8, 
                    '4': 0.7, '5': 0.6}

    # Read the rank popularity of names by year
    df = pd.read_csv('data/Baby-Names-SSA.csv')
    df = df.set_index('Year').sort_index()
    unstack = df.unstack()

    # Random features
    colors = ('Red', 'Green', 'Blue', 
              'Yellow', 'Purple', 'Black')
    flowers = ('Daisy', 'Orchid', 'Rose', 
               'Violet', 'Lily')

    # Python list-of-lists to construct new data
    rows = []
    for (rank_gender, year), name in unstack.iteritems():
        rank, gender = rank_gender
        age = now - year
        count = int(year_to_pop[year] *
                    rank_to_freq[rank])
        for _ in range(count):
            color = random.choice(colors)
            flower = random.choice(flowers)
            rows.append(
                (age, gender, name, color, flower))

    df = pd.DataFrame(rows)
    df.columns = ('Age', 'Gender', 'Name',
                  'Favorite_Color', 'Favorite_Flower')
    df = df.sample(frac=0.8, random_state=1)
    df.to_parquet('data/usa_names_all.parq', index=False)
    
    # Add age-correlated flower preference
    old = df[df.Age > 70].sample(frac=0.2, random_state=1)
    df.loc[old.index, 'Favorite_Flower'] = 'Orchid'
    young =df[df.Age < 30].sample(frac=0.1, random_state=1)
    df.loc[young.index, 'Favorite_Flower'] = 'Rose'
    
    # Make some data missing selectively by age
    # Missing color for all but forty 30-40 yos
    drop = df[(df.Age > 30) & (df.Age <= 40)].index[:-40]
    df.loc[drop, 'Favorite_Color'] = None

    # Missing flower for all but forty 20-30 yos
    drop = df[(df.Age > 20) & (df.Age <= 30)].index[:-40]
    df.loc[drop, 'Favorite_Flower'] = None
    
    # Jumble the order but keep all rows then write
    df = df.sample(frac=1.0, random_state=1)
    df.to_parquet('data/usa_names.parq', index=False)

month_names = np.array("""
    January February March April May
    June July August September October
    November December""".split())

def make_birth_month(ages):
    "Artificially assign biased birth month"
    np.random.seed(1)
    choice = np.random.choice
    # Just going to use plain Python, not vectorize
    # ... speed irrelevant for few thousand values
    mean_age = np.mean(ages)
    birth = []
    for age in ages:
        # Make Jan either over- or under represented
        bias = (age - mean_age) * 2
        probs = [100] * 12
        probs[0] += bias
        probs /= np.sum(probs)
        birth.append(choice(month_names, 1,
                            p=probs)[0])
    return birth


def make_ssa_states(names='data/usa_names_all.parq',
                    pops='data/state-population.fwf'):
    np.random.seed(1)
    choice = np.random.choice
    normal = np.random.normal
    # Assume make_ssa_synthetic() or equiv has run
    names = pd.read_parquet(names)
    states =  pd.read_fwf(pops)
    
    # Start with log population 
    probs = np.log(states.Population_2010)
    nudge = normal(loc=1, scale=0.2, size=len(probs))
    probs *= nudge
    probs /= probs.sum()
    homes = choice(states.State, len(names), p=probs)
    names['Home'] = homes
    names['Birth_Month'] = make_birth_month(names.Age)
    cols = ['Age', 'Birth_Month','Name', 'Gender', 'Home']
    names = names[cols]
    names.to_parquet('data/usa_names_states.parq',
                     index=False)


# Synthetic data to illustrate scaling
def make_unscaled_features(N=200, j1=1/50, j2=1/10):
    """Create DataFrame of synthetic data
    
    Feature_1 will be:
      * positively correlated with Target
      * numerically small values
    Feature_2 will be:
      * negatively correlatged with Target
      * numerically large values
    
    N  - number of rows to geneate
    j1 - the relative scale of random jitter for F1
    j2 - the relative scale of random jitter for F2
    """
    assert j2 > j1
    # Repeatable randomness
    np.random.seed(1)
    
    # Target points range from 10 to 20
    target = np.linspace(10, 20, N)
    
    # Feature_1 is roughly 1/100th size of Target
    feat1 = target / 100
    feat1 += np.random.normal(
        loc=0, scale=np.max(feat1)*j1, size=N) 

    # Feature_2 is around 20,000
    feat2 = np.linspace(21_000, 19_000, N)
    feat2 += np.random.normal(
        loc=0, scale=np.max(feat2)*j2, size=N)

    df = pd.DataFrame({'Feature_1': feat1, 
                       'Feature_2': feat2, 
                       'Target': target})
    return (df.sample(frac=1)
              .reset_index()
              .drop('index', axis=1))


def plot_univariate_trends(df, Target='Target'):
    df = df.sort_values(Target)
    target = df[Target]
    X = df.drop(Target, axis=1)
    n_feat = len(X.columns)
    fig, axes = plt.subplots(n_feat, 1,
                             figsize=(10, n_feat*2))
    for ax, col in zip(axes, X.columns):
        ax.plot(target, X[col])
        ax.set_title(f"{col} as a function of {Target}")
    fig.tight_layout()
    plt.savefig(f'img/univariate-{"_".join(X.columns)}:{Target}.png')
    

def read_glarp(cleanup=True):
    df = pd.DataFrame()
    # The different thermometers
    places = ['basement', 'lab', 'livingroom', 'outside']
    for therm in places:
        with gzip.open('data/glarp/%s.gz' % therm) as f:
            readings = dict()
            for line in f:
                Y, m, d, H, M, temp = line.split()
                readings[datetime(*map(int, 
                         (Y, m, d, H, M)))] = float(temp)
        df[therm] = pd.Series(readings)

    if cleanup:
        # Add in the relatively few missing times
        df = df.asfreq('3T').interpolate()
        
        # Remove reading with implausible jumps
        diffs = df.diff()
        for therm in places:
            errs = diffs.loc[diffs[therm].abs() > 5,
                             therm].index
            df.loc[errs, therm] = None
            
        # Backfill missing temperatures (at start)
        df = df.interpolate().bfill()

    # Sort by date but remove from index
    df = df.sort_index().reset_index()
    df = df.rename(columns={'index': 'timestamp'})
    
    return df


def get_digits():
    from sklearn.datasets import load_digits
    digits = load_digits()

    fig, axes = plt.subplots(2, 5,
                    figsize=(10, 5),
                    subplot_kw={'xticks':(), 
                                'yticks': ()})
    for ax, img in zip(axes.ravel(),
                       digits.images):
        ax.imshow(img, cmap=plt.get_cmap('Greys'))
    fig.tight_layout()
    fig.savefig("img/first-10-digits.png")
    return digits


grays10 = """
#000000 #DCDCDC #D3D3D3 #C0C0C0 #A9A9A9
#808080 #696969 #778899 #708090 #2F4F4F
""".split()

vivid = """
#476A2A #7851B8 #BD3430 #4A2D4E #875525
#A83683 #4E655E #853541 #3A3120 #535D8E
""".split()

def plot_digits(data, digits, 
                decomp="Unknown", colors=grays10):
    plt.figure(figsize=(8, 8))
    plt.xlim(data[:, 0].min(), 
             data[:, 0].max() + 1)
    plt.ylim(data[:, 1].min(), 
             data[:, 1].max() + 1)
    for i in range(len(digits.data)):
        # plot digits as text not using scatter
        plt.text(data[i, 0], 
                 data[i, 1],
                 str(digits.target[i]),
                 color = colors[
                     digits.target[i]],
                 fontdict={'size': 9})
    plt.title(f"{decomp} Decomposition")
    plt.savefig(f"img/{decomp}-decomposition.png")

