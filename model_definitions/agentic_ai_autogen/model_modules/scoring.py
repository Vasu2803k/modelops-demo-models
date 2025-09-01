#!/usr/bin/env python3
import json
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Scoring function for evaluating the model performance
def score(context: ModelContext, **kwargs):
    print("Started scoring")
