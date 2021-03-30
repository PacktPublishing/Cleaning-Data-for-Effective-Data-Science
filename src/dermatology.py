import pandas as pd
import numpy as np

np.random.seed(1)

# Histopathological Attributes: (values 0, 1, 2, 3)
# Clinical Attributes: (values 0, 1, 2, 3, unless indicated)
features = [
    "erythema",
    "scaling",
    "definite borders",
    "itching",
    "koebner phenomenon",
    "polygonal papules",
    "follicular papules",
    "oral mucosal involvement",
    "knee and elbow involvement",
    "scalp involvement",
    "family history",  # 0 or 1
    "melanin incontinence",
    "eosinophils in the infiltrate",
    "PNL infiltrate",
    "fibrosis of the papillary dermis",
    "exocytosis",
    "acanthosis",
    "hyperkeratosis",
    "parakeratosis",
    "clubbing of the rete ridges",
    "elongation of the rete ridges",
    "thinning of the suprapapillary epidermis",
    "spongiform pustule",
    "munro microabcess",
    "focal hypergranulosis",
    "disappearance of the granular layer",
    "vacuolisation and damage of basal layer",
    "spongiosis",
    "saw-tooth appearance of retes",
    "follicular horn plug",
    "perifollicular parakeratosis",
    "inflammatory monoluclear inflitrate",
    "band-like infiltrate",
    "Age",  # linear; missing marked '?'
    "TARGET"  # See mapping
]

targets = {
    1:"psoriasis",                 # 112 instances
    2:"seboreic dermatitis",       # 61
    3:"lichen planus",             # 72
    4:"pityriasis rosea",          # 49
    5:"cronic dermatitis",         # 52
    6:"pityriasis rubra pilaris",  # 20
}

# SSL problem with UCI!
#base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/'
base = 'data/'
data = base + 'dermatology.data'
metadata = base + 'dermatology.names'
df = pd.read_csv(data, header=None, names=features)
df['TARGET'] = df.TARGET.map(targets)

derm = df.copy()
derm.loc[derm.Age == '?', 'Age'] = None
derm['Age'] = derm.Age.astype(float)
