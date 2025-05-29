# coverage_planning/utils.py

# When VERBOSE = True, log() will print.
# When VERBOSE = False, log() does nothing.
VERBOSE = True

def log("*args, **kwargs"):
    if VERBOSE:
        print(*args, **kwargs)

