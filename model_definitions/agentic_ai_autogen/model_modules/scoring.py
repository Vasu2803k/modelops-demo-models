"""
Scoring module for autogen agent.

This module provides scoring capabilities for the conversational agent.
"""
import os
import json
from tmo import ModelContext

# Scoring function for evaluating the model performance
def score(context: ModelContext, **kwargs):
    print("Started scoring")
