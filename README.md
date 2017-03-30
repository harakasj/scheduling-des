Simulates RR scheduling. Have a decent amount of work to do on this to make it run smoother like fixing subclasses, callbacks, better data logging, etc... 

I'd recommend checking out the [SimPy documentation](https://media.readthedocs.org/pdf/simpy/latest/simpy.pdf)

main_queue.py is the main program. The processes are generated as instances of the Job() class, which get saved to a list and pickled.

The other two files are for plotting animation in Qt5, still in the rough.

Typical output looks something like this...
```
Time     PID      Event    Info
0        pid 1    arrive   service time = 75       
0        pid 1    start    remain = 75       
5        pid 1    stop     remain = 70        
5        pid 1    start    remain = 70       
10       pid 2    arrive   service time = 40       
10       pid 1    stop     remain = 65        
10       pid 2    start    remain = 40       
15       pid 3    arrive   service time = 25       
15       pid 2    stop     remain = 35        
```

The test case will generate a summary like below. 
```
================================= Summary ===================================

Time elapsed: 0.009828238999999961
Quantum: 5
Context Switch: 0
Mean turnaround: 114.00
Mean wait time: 73.00
Queue size at finish: 1

job        arrive     service    init wait  finish     wait       turnaround
pid 1      0          75         0          195        120        195       
pid 2      10         40         0          130        80         120       
pid 3      15         25         5          85         45         70        
pid 4      80         20         10         150        50         70        
pid 5      90         45         10         205        70         115       
==============================================================================
```

