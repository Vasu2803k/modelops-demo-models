from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
def score(context: ModelContext, **kwargs):
    aoa_create_context()
    print("Scoring")
