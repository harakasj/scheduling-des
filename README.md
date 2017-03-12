# scheduling-des
Simulates RR scheduling. Have a decent amount of work to do on this to make it run smoother like fixing subclasses, callbacks, better data logging, etc... 

I'd recommend checking out the [SimPy documentation](https://media.readthedocs.org/pdf/simpy/latest/simpy.pdf)

queue_sim.py is the main program. The processes are generated as instances of the Job() class, which get saved to a list and pickled.

queue_graphics.py loads the pickled file, so you don't need to keep running the simulation over and over. Then use the instances to make some nice histograms for analysis. 

Typical output looks something like this...
```
Time     PID      Event    Info
4        pid 1    arrive   service time = 9        
	     Queued:  1
4        pid 1    start    time needed = 9        
5        pid 2    arrive   service time = 5        
13       pid 3    arrive   service time = 11       
13       pid 1    finish   trnd time = 9        wait time = 0        
	     Queued:  2
13       pid 2    start    time needed = 5        
18       pid 2    finish   trnd time = 5        wait time = 9        
	     Queued:  1
18       pid 3    start    time needed = 11       
22       pid 4    arrive   service time = 3        
28 	     pid 3  quantum expired  
```

