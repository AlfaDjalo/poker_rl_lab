import time
import poker_engine.poker_eval as pe
m = lambda *c: sum(1 << x for x in c)
t = time.perf_counter()
r = pe.holdem_matchup_counts(m(12, 25), m(11, 24), 0, 20000, 1)   # AcAd vs KcKd
print(r, time.perf_counter() - t)