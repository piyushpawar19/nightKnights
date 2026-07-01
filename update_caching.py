import os

def update_file(f, replacements):
    c = open(f).read()
    for old, new in replacements:
        c = c.replace(old, new)
    open(f, 'w').write(c)

repl1 = [
    ('from functools import lru_cache', 'from functools import lru_cache\nimport os\nfrom joblib import Memory\n\nCACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "cache", "joblib_cache")\nos.makedirs(CACHE_DIR, exist_ok=True)\nmemory = Memory(CACHE_DIR, verbose=0)'),
    ('@lru_cache(maxsize=1024)', '@memory.cache')
]
update_file('src/preprocessing/feature_extractors.py', repl1)

repl2 = [
    ('from functools import lru_cache', 'from functools import lru_cache\nimport os\nfrom joblib import Memory\n\nCACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "cache", "joblib_cache")\nos.makedirs(CACHE_DIR, exist_ok=True)\nmemory = Memory(CACHE_DIR, verbose=0)'),
    ('@lru_cache(maxsize=256)', '@memory.cache'),
    ('@lru_cache(maxsize=1)', '@memory.cache')
]
update_file('src/ranking/reranking_utils.py', repl2)

repl3 = [
    ('from functools import lru_cache', 'from functools import lru_cache\nimport os\nfrom joblib import Memory\n\nCACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "cache", "joblib_cache")\nos.makedirs(CACHE_DIR, exist_ok=True)\nmemory = Memory(CACHE_DIR, verbose=0)'),
    ('@lru_cache(maxsize=128)', '@memory.cache'),
    ('@lru_cache(maxsize=16)', '@memory.cache')
]
update_file('src/jd_parser/experience_parser.py', repl3)
update_file('src/jd_parser/education_parser.py', repl3)
update_file('src/jd_parser/salary_parser.py', repl3)
update_file('src/ranking/recruiter_parser.py', repl3)
update_file('src/ranking/recruiter_reranker.py', repl3)

print("Updated caching successfully.")