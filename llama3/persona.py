import csv
import random

import numpy as np
import pycountry

# ─── Configuration ───────────────────────────────────────────────────────────────

# Only these ISO‑2 country codes (from your Table S2: n > 100)
ALLOWED_COUNTRY_CODES = {
    "US","GB","DE","FR","CA","IT","BR","AU","TR","ES",
    "NL","RU","PL","SE","IN","FI","SG","DK","BE","CH",
    "AT","CZ","MX","PT","RO","NZ","VN","HU","NO","JP",
    "AR","CN","GR","IE","UA","PH","HK","TW","CO","IL",
    "ID","MY","ZA","KR","CL","BG","AE","HR","SK","TH",
    "RS","LT","EE","SI","PE","EG","LV","SA","BY","LU",
    "IR","PK","CR","AZ","MA","UY","KZ","DZ","EC","GE",
    "BA","BD","QA","VE","MD","IS","MO","LB","PR"
}

NUM_PROFILES_PER_COUNTRY = 30
CSV_FILENAME            = "donor_persona.csv"

# ─── Demographic & Attitude Distributions from your Tables ────────────────────

# Age [18–75], median=23, IQR=(20–30), Unknown=11.6%
AGE_MEAN      = 23
AGE_SD        = (30 - 20) / 1.34896   # ≈ 7.41
AGE_UNKNOWN_P = 7606 / 65583         # ≈ 0.116

# Gender: M 55.8%, F 31.0%, Other 2.8%, Unknown 10.5%
GENDERS        = ["Male","Female","Other","Unknown"]
GENDER_WEIGHTS = [0.558,0.310,0.028,0.105]

# Income bins (as in Table S1)
INCOME_CATS    = ["<5k","5-10k","10-15k","15-25k","25-35k",
                  "35-50k","50-65k","65-80k","80-100k",">100k","Unknown"]
KNOWN_INC_P    = 1 - 0.176
INCOME_WEIGHTS = [
    0.280*KNOWN_INC_P, 0.110*KNOWN_INC_P, 0.080*KNOWN_INC_P,
    0.092*KNOWN_INC_P, 0.088*KNOWN_INC_P, 0.099*KNOWN_INC_P,
    0.067*KNOWN_INC_P, 0.053*KNOWN_INC_P, 0.046*KNOWN_INC_P,
    0.090*KNOWN_INC_P, 0.176
]

# Education levels (as in Table S1)
EDU_CATS      = ["LessThanHS","HighSchool","VocTraining",
                 "SomeCollege","Bachelors","Graduate","Unknown"]
KNOWN_EDU_P   = 1 - 0.1129
EDU_WEIGHTS   = [
    0.036*KNOWN_EDU_P, 0.20*KNOWN_EDU_P, 0.026*KNOWN_EDU_P,
    0.21*KNOWN_EDU_P, 0.27*KNOWN_EDU_P, 0.26*KNOWN_EDU_P,
    0.1129
]

# Political views (0–1), median=0.71, IQR=(0.50–0.92)
POL_MEAN = 0.71
POL_SD   = (0.92 - 0.50) / 1.34896

# Religious views (0–1), median=0.06, IQR=(0.00–0.50)
REL_MEAN = 0.06
REL_SD   = (0.50 - 0.00) / 1.34896

# Donation history: Yes 70.2%, No 21.3%, Unknown 8.5%
DONATE_CATS    = [1, 0, "Unknown"]
DONATE_WEIGHTS = [0.702, 0.213, 0.085]

# Donation frequency (given donated==1): 
#   More than once a month 20%, 
#   Less than once a month (but > once a year) 37%, 
#   Less than once a year 43%
FREQ_CATS    = ["MoreThanOnceMonth","LessThanOnceMonth","LessThanOnceYear"]
FREQ_WEIGHTS = [0.20, 0.37, 0.43]

# Attitudes (0–1), medians/IQRs from Table S3
TRUST_MEAN, TRUST_SD = 0.50, (0.69 - 0.31) / 1.34896
RATE_MEAN,  RATE_SD  = 0.68, (0.84 - 0.50) / 1.34896
OBJ_MEAN,   OBJ_SD   = 0.66, (0.84 - 0.50) / 1.34896
VIEW_MEAN,  VIEW_SD  = 0.73, (0.92 - 0.50) / 1.34896

# ─── Sampling Helper Functions ────────────────────────────────────────────────

def trunc_norm(mean, sd, lo=0.0, hi=1.0):
    """Sample normal then clip to [lo,hi]."""
    return float(np.clip(np.random.normal(mean, sd), lo, hi))

def random_age():
    if random.random() < AGE_UNKNOWN_P:
        return ""
    a = int(round(np.random.normal(AGE_MEAN, AGE_SD)))
    return max(18, min(75, a))

def random_gender():
    return random.choices(GENDERS, GENDER_WEIGHTS, k=1)[0]

def random_income():
    cat = random.choices(INCOME_CATS, INCOME_WEIGHTS, k=1)[0]
    if cat == "Unknown":
        return ""
    lo, hi = {
        "<5k":    (0,     5000),
        "5-10k":  (5000,  10000),
        "10-15k": (10000, 15000),
        "15-25k": (15000, 25000),
        "25-35k": (25000, 35000),
        "35-50k": (35000, 50000),
        "50-65k": (50000, 65000),
        "65-80k": (65000, 80000),
        "80-100k":(80000,100000),
        ">100k":  (100000,120000),
    }[cat]
    return round(random.uniform(lo, hi), 2)

def random_education():
    cat = random.choices(EDU_CATS, EDU_WEIGHTS, k=1)[0]
    return "" if cat == "Unknown" else cat

def random_political():   return trunc_norm(POL_MEAN, POL_SD)
def random_religious():   return trunc_norm(REL_MEAN, REL_SD)

def random_donated():
    d = random.choices(DONATE_CATS, DONATE_WEIGHTS, k=1)[0]
    return "" if d == "Unknown" else d

def random_frequency(donated):
    if donated == 1:
        return random.choices(FREQ_CATS, FREQ_WEIGHTS, k=1)[0]
    return ""

def random_trust(): return trunc_norm(TRUST_MEAN, TRUST_SD)
def random_rate():  return trunc_norm(RATE_MEAN,  RATE_SD)
def random_obj():   return trunc_norm(OBJ_MEAN,   OBJ_SD)
def random_view():  return trunc_norm(VIEW_MEAN,  VIEW_SD)

# ─── Profile & CSV Writer ────────────────────────────────────────────────────

def generate_profile(country, subdivision):
    donated = random_donated()
    return {
        "age":       random_age(),
        "gender":    random_gender(),
        "income":    random_income(),
        "edu":       random_education(),
        "pol":       random_political(),
        "rel":       random_religious(),
        "donated":   donated,
        "freq":      random_frequency(donated),
        "trust":     random_trust(),
        "bel_rate":  random_rate(),
        "bel_obj":   random_obj(),
        "bel_view":  random_view(),
        "country":   country.name,
        "province":  subdivision.name if subdivision else "",
    }

def save_profiles_to_csv(filename, n_per_country):
    # Filter to only the allowed countries
    country_map  = {c.alpha_2: c for c in pycountry.countries}
    countries    = [country_map[c] for c in ALLOWED_COUNTRY_CODES if c in country_map]
    subdivisions = list(pycountry.subdivisions)

    # Write CSV
    sample      = generate_profile(countries[0], None)
    fieldnames  = list(sample.keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in countries:
            subs = [s for s in subdivisions if s.country_code == c.alpha_2]
            for _ in range(n_per_country):
                sub = random.choice(subs) if subs else None
                writer.writerow(generate_profile(c, sub))

if __name__ == "__main__":
    save_profiles_to_csv(CSV_FILENAME, NUM_PROFILES_PER_COUNTRY)
    total = NUM_PROFILES_PER_COUNTRY * len(ALLOWED_COUNTRY_CODES)
    print(f"✅ Wrote {total} profiles to '{CSV_FILENAME}'")
