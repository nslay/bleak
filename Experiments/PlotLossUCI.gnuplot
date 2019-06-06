#!/usr/bin/gnuplot

set term epslatex standalone color size 3.5in,2in font "arial,8pt"
set output 'LossUCI.tex'

set xtics 10000

set yrange [0.6:1.2]
set xrange [0:40000]
set grid
set key top left

set title 'Varying Tree Depths for madelon'
set xlabel 'Iteration'
set ylabel 'Validation Loss'

set style line 1 lt 4 lc rgb "red" lw 1
set style line 2 lt 6 lc rgb "red" lw 1
set style line 3 lt 8 lc rgb "red" lw 1

set style line 4 lt 5 lc rgb "blue" lw 1
set style line 5 lt 7 lc rgb "blue" lw 1
set style line 6 lt 3 lc rgb "blue" lw 1

plot 'VaryingTrees/madelon/100_1_ValLossVsItr.txt' using 1:2 ls 1 w linespoints title '100/1 Hinge Forest',\
'VaryingTrees/madelon/100_5_ValLossVsItr.txt' using 1:2 ls 2 w linespoints title '100/5',\
'VaryingTrees/madelon/100_10_ValLossVsItr.txt' using 1:2 ls 3 w linespoints title '100/10',\
'VaryingFerns/madelon/100_1_ValLossVsItr.txt' using 1:2 ls 4 w linespoints title '100/1 Hinge Fern',\
'VaryingFerns/madelon/100_5_ValLossVsItr.txt' using 1:2 ls 5 w linespoints title '100/5',\
'VaryingFerns/madelon/100_10_ValLossVsItr.txt' using 1:2 ls 6 w linespoints title '100/10'
