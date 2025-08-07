# Instructions

The [documentation](https://github.com/pyutils/line_profiler) for `line_profiler` is helpful.
Just run `LINE_PROFILE=1 python run.py` and it will gather data.
You can view the data by running `python -m line_profiler -rtmz profile_output.lprof`.

## The goal
In `run.py` you'll find part of an old version of gw_ac. Identify some bottlenecks with `line_profiler` and try to make them faster. All arrays are real, not complex, to keep this simpler.
Go ahead and use [`pytblis.contract(...)`](https://pytblis.readthedocs.io/en/latest/api.html#pytblis.contract) if you like. Feel free to use anything else at your disposal.

### Tip: save your work using git branches or tags so you're not doing ctrl-z all the time
