import cProfile, pstats
from npMeshMandle import mandlebrotIsolated
from mandelbrot import mandlebrot

cProfile.run('mandlebrot(100)', 'mandlebrot_profile.prof')

cProfile.run('mandlebrotIsolated(100)', 'mandelbrotIsolated_profile.prof')

for name in ('mandlebrot_profile.prof', 'mandelbrotIsolated_profile.prof'):
    stats = pstats.Stats(name)
    stats.sort_stats('cumulative')
    stats.print_stats(10)