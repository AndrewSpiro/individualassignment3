from feature_transformations import 

saved = ['all', 'assignment 2', 'None']

def load_features(load_saved: str):
    if load_saved not in saved:
        raise ValueError("load_saved must be 'all' (no transformations are rerun-saved results used), 'assignment 2' (new transformations for assignment 3 are rerun, saved assignment 2 results are used), or 'none' (no saved results are used, all transformations rerun. This is not advised because of runtimes >5 hours.)")
