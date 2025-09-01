#!/usr/bin/env python3
import json
from tmo import ModelContext
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Scoring function for evaluating the model performance
def score(context: ModelContext, **kwargs):
    print("Started scoring")
