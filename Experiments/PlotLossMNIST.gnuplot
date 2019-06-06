#!/usr/bin/gnuplot

set term epslatex standalone color size 3.5in,2in font "arial,8pt"
set output 'LossMNIST.tex'

set xtics 10000

unset ytics
set y2tics
#set yrange [0.05:0.25]
set y2range [0.04:0.25]
set xrange [0:40000]
set grid x y2
set key bottom right

set title 'Varying Number of Trees for MNIST'
set xlabel 'Iteration'
#set ylabel 'Test Loss'
set y2label 'Test Loss'

set style line 1 lt 4 lc rgb "red" lw 1
set style line 2 lt 6 lc rgb "red" lw 1

set style line 3 lt 5 lc rgb "blue" lw 1
set style line 4 lt 7 lc rgb "blue" lw 1

plot 'CompareTrees/mnist/100_10_TestLossVsItr.txt' using 1:2 axes x1y2 ls 1 w linespoints title '100/10 Hinge Forest',\
'CompareTrees/mnist/1000_10_TestLossVsItr.txt' using 1:2 axes x1y2 ls 2 w linespoints title '1000/10',\
'CompareFerns/mnist/100_10_TestLossVsItr.txt' using 1:2 axes x1y2 ls 3 w linespoints title '100/10 Hinge Fern',\
'CompareFerns/mnist/1000_10_TestLossVsItr.txt' using 1:2 axes x1y2 ls 4 w linespoints title '1000/10'
