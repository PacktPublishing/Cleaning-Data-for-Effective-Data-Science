from math import pi
import sqlite3
import random
from datetime import datetime
import os
import numpy as np
import pandas as pd

def make_events():
    # For each of 5 distributions, loop through N times, 
    # adding points according to probability at time
    np.random.seed(1)
    N = 3
    nrecs = 0
    day = 60*24
    mins = 366*day
    ts = pd.date_range('2020-01-01', 
                       periods=mins, freq='T')
    dists = {}
    
    x = np.arange(mins)
    dists['A'] = np.sin(x * pi/mins)
    dists['B'] = x/mins
    dists['C'] = np.cos(x*365 * pi/mins)
    dists['D'] = np.ones(x.size) * 0.1
    
    # Create from random distribution, lognormal
    mu, sigma = 1., 0.3 # mean and standard deviation
    s = np.random.lognormal(mu, sigma, mins*20)
    y_e = np.repeat(np.histogram(s, mins//day)[0], day)
    dists['E'] = np.array(y_e, dtype=float) / y_e.max()

    os.remove('data/events.sqlite')
    db = sqlite3.connect('data/events.sqlite')
    cur = db.cursor()
    sql = ('CREATE TABLE event '
           '(timestamp timestamp, instrument CHAR(1))')
    cur.execute(sql)

    for _ in range(N):
        for instrument, dist in dists.items():
            recs = ts[np.random.random(mins) < dist/N]
            for dt in recs:
                sql = ('INSERT INTO event '
                       f'VALUES ("{dt}", "{instrument}")')
                cur.execute(sql)
            cur.execute('COMMIT')
            print(f"Wrote {len(recs)} from {instrument}")
            nrecs += len(recs)
    return nrecs

    
if __name__ == '__main__':
    make_events()