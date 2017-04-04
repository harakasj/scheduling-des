"""
RR scheduling sim for single-core (or multi) cpu
Requires: Python 3.6
Dependencies: simpy, numpy, 
Optional dep: scipy, pyqt5, matplotlib 2.0

Check SimPy domcumentation to better understand how the sim works:
https://media.readthedocs.org/pdf/simpy/latest/simpy.pdf

If no plotting, only simpy and numpy are required

Only string formatting using f-literals is a compatability issue 
which whas introduced in Python 3.6. Otherwise, it should run 
on < Python 3. Just modify or remove print formats with f-literals.

---- How it works (basically) -----: 
main() creates the simulation environment env(), then passes it to init_sim()

init_sim() creates the CPU() Resource and generates Job() events and 
sends them to job_proc()

job_proc() runs the events. When/if the quantum expires, the job is 
appended to the back of the queue.

---- Other stuff ----
Check the optional arguments for other things.

The Record object records the simulation state, for later analysis
- its a mess right now. But it works.

---- Future ----
allow for preemption
better state recording
better plotting params

"""

import argparse
import pickle
import simpy
import time
import random
from random import randint, expovariate
import numpy as np
from array import array
from PlotQt import *
from numpy.core.fromnumeric import mean
from collections import namedtuple

# Default parameters
RAND_SEED = 50  # rand seed to reproduce
TIME_QUANTUM = 3  # time quantum
SERVICE_TIME = [4, 8]
NUM_CORES = 1  # number of cpu cores
CONTEXT_SWITCH = 0
# INTER_TIME = [30, 300]
INTER_T = (1 / 8)  # 1 arrival every 8 seconds
SIM_TIME = 20000
MAX_PROC = 1000
STOP_SIG = 1000


def plots(arg):
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from scipy import stats
        path = 'figures/'
        plt.close('all')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 15))
        f.tight_layout(pad=2, h_pad=1.0)
        ax1.hist(Record.q_size,
                 histtype='bar',
                 bins=18,
                 normed=1)
        ax1.axis([0, 15, 0, 0.3])
        ax1.set_title("Queue Size (Normalized) (TQ = %d, $\lambda=1/8$, jobs= %d)"
                  % (arg.quantum, len(Record.jobs_completed)))
        ax2.hist(list(x.trnd_t for x in Record.jobs_completed),
                 histtype='bar',
                 bins="auto",
                 normed=0)
        ax2.set_title("Wait Time (TQ = %d, $\lambda=1/8$, jobs= %d)"
                  % (arg.quantum, len(Record.jobs_completed)))
        ax2.axis([0, 100, 0, 250])
        ax3.hist(list(x.wait_t for x in Record.jobs_completed),
                 histtype='bar',
                 bins="auto",
                 normed=0)
        ax3.set_title("Turnaround Time (TQ = %d, $\lambda=1/8$, jobs= %d)"
                  % (arg.quantum, len(Record.jobs_completed)))
        ax3.axis([0, 100, 0, 250])
        f.subplots_adjust(hspace=0.2)
        fname = "TQ-%d.%s" % (arg.quantum, arg.ext)
        f.savefig(path + fname, format=arg.ext)

        plt.figure(figsize=(15, 10))
        plt.step(Record.q_time,
                 stats.zscore(Record.mean_burst),
                 where="mid",
                 label="Average Service Time")

        plt.step(Record.q_time,
                 stats.zscore(Record.q_size),
                 where="mid",
                 label="Queue Size")
        plt.title("Queue Size vs Time (TQ = %d, $\lambda=1/8$, jobs= %d)"
                  % (arg.quantum, arg.jobs))
        plt.legend()

        fname = "TQ-%d-Time-Series.%s" % (arg.quantum, arg.ext)
        plt.savefig(path + fname, format=arg.ext)
        plt.show()
    except ImportError as i:
        print("Missing %s" % i)


def convolve(x, N):
    # helper function, to compute a moving average
    cumsum = np.cumsum(np.insert(x, 0, 0))
    m = np.append(np.zeros(N), (cumsum[N:] - cumsum[:-N]) / N)
    return array("d", m)


def summary(dt,arg):  # Print out your summary data
    # PEP 8, no more lambda assignment, use inline def?
    def avg(x): return sum(x) / len(x)
    print("\n================================= "
          "Summary "
          "===================================\n")
    print("Time elapsed:", dt)
    print("Quantum: %d" % arg.quantum)
    print("Context Switch: %d" % arg.context)
    print('Mean turnaround: %1.2f' % avg(list(x.trnd_t for x in Record.jobs_completed)))
    print('Mean wait time: %1.2f' % avg(list(x.wait_t for x in Record.jobs_completed)))
    print('Queue size at finish: %d' % (Record.q_size[-1]-1)) # bug, queue ending with +1 item
    print("\n{0:<10s} {1:<10s} {2:<10s} {3:<10s} {4:<10s} {5:<10s} {6:<10s}"
          .format("job", "arrive", "service", "init wait", "finish", "wait", "turnaround"))
    print(*(x.__str__(True) for x in sorted(Record.jobs_completed,
                                            key=lambda k: k.arrive_t)), sep='\n')
    print('======================================'
          '======================================')


class Record:
    """
    This is for record-keeping
    Arrays are better on performance when they are being 
    frequently manipulated, as opposed to lists. 

    Pretty much all parameters and simulation state at every
    step is saved in this. Once the simulation terminates, 
    this object is pickled and saved as a binary which can 
    be loaded later on without having to rerun the simulation 

    Some of the fields are redundant and not being used right now.
    """
    # norm = lambda X : [ ((n - min(X)) / (max(X) - min(X))) for n in X]
    sim_t = SIM_TIME  # simulation duration
    inter_t = INTER_T  # param for arrival time
    randseed = RAND_SEED  # random seed used
    quantum_t = TIME_QUANTUM  # time quantum used
    num_cores = NUM_CORES  # number of cores
    cswitch_t = CONTEXT_SWITCH  # context switch size used

    # Gonna keep a list of every job instance that completes
    jobs_completed = []
    total_t = 0  # sum of all jobs burst times
    total_burst = array("d", [])  # sum of burst times at each step
    total_hist_t = array("d", [])  # step at which burst total was recorded
    q_size = array("d", [])  # size of queue at each step
    q_time = array("d", [])  # step at which queue size was recorded
    times = array("d", [])
    mean_burst = array("d", [])
    wait_t = array("d", [])
    trnd_t = array("d", [])
    mean_wait = array("d", [])
    mean_trnd = array("d", [])
    pids = array("i", [])  # pids
    arrivals = array("d", [])  # arrival times of jobs
    finish = array("d", [])  # finish time of jobs
    # IF going to be doing a run-test
    run_test = False
    arrive_times = []
    service_times = []


class Env(simpy.Environment):
    """
    This is the environment in which the simulation is "contained"
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proc_data = []

    def setup(self):
        pass

    def exit(self, value=None):
        return super().exit(value)

    def show_queue(self):
        print(*self._queue, sep='\n')

    def get_queue(self):
        return len(self._queue)

    def step(self):
        return simpy.Environment.step(self)

    @staticmethod
    def save_record():
        filepath = 'bin/Record.p'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print('Writing record to %s' % filepath)
        pickle.dump(Record, open(filepath, 'wb'))
        return


class Job(simpy.events.Event):
    """
    The Job() class is the primary event, where each 
    new job is added to the waiting queue for access to the
    CPU (Resource). 
    Most of this class is just keeping track of specific 
    information like arrival time, turnaround time, wait time, etc..
    """
    def __init__(self, pid, arrive_t, env, rand=randint):
        # If test info is being used, initialize that first
        super().__init__(env)
        if len(Record.service_times) > 0:
            self.service_t = Record.service_times.pop(0)
        else:
            # Otherwise, its going to be a random service time
            self.service_t = rand(*SERVICE_TIME)  # random service time
        self.id = pid
        self.arrive_t = arrive_t
        self.finish_t = 0
        self.trnd_t = 0
        self.wait_t = 0
        self.remain_t = self.service_t
        self.initial_wait = 0

    def get_trnd(self, finish_t):
        # When the job finishes, compute the turnaround/wait time
        self.finish_t = finish_t
        self.trnd_t = self.finish_t - self.arrive_t
        self.wait_t = self.trnd_t - self.service_t
        return self.trnd_t, self.wait_t

    def trigger(self, event):
        return simpy.events.Event.trigger(self, event)

    def __repr__(self):
        return simpy.events.Event.__repr__(self)

    def __str__(self, *args, **kwargs):
        if args:
            return f'{self.id:<10s} {self.arrive_t:<10d} ' \
                   f'{self.service_t:<10d} {self.initial_wait:<10d} ' \
                   f'{self.finish_t:<10d} {self.wait_t:<10d} {self.trnd_t:<10d}'
        else:
            return self.id


class CPU(simpy.Resource):
    # This is an inherited from a generic base class.
    # The CPU is treated as the primary resource
    def __init__(self, quantum,context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = args
        self.data = []
        self.waiting = []
        self.quantum = quantum
        self.context = context

    def request(self, pid, *args, **kwargs):
        if pid:
            self.waiting.append(pid)
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        return super().release(*args, **kwargs)

    def do_job(self, env, pid):
        # Either the job completes or is interrupted by a timeout
        try:
            yield env.timeout(pid.remain_t)
            Record.total_t -= pid.remain_t
        except simpy.Interrupt as i:
            print("{0:<8d} {1:<9s}".format(int(env.now), str(pid)), end='')
            print(i.cause, end='')
            print('{0:>13s} {1:<10.0f}'.format('remain =', pid.remain_t))
            Record.total_t -= self.quantum
        # Some record-keeping
        Record.total_burst.append(Record.total_t)
        Record.mean_burst.append(self.check_queue())
        Record.q_time.append(env.now)
        Record.q_size.append(len(self.waiting))

    def check_queue(self):
        s = 0
        for q in self.waiting:
            s += q.remain_t
        return mean(s)

    def preempted_put(self, event):
        pass

    def _do_put(self, event):
        return simpy.Resource._do_put(self, event)

    def _do_get(self, event):
        return simpy.Resource._do_get(self, event)

# ------------------------------------------------------------------------------
# --------------------------- Schedules jobs -----------------------------------
# ------------------------------------------------------------------------------
def job_sched(env, proc, cpu):
    with cpu.request(pid=proc) as request:
        yield request
        # record the job's first time getting a time slice
        if proc.remain_t == proc.service_t:
            proc.initial_wait = env.now - proc.arrive_t
        # Print out queue size at every step.
        # print("{0:<8d} Queued:{1:<8d}".format(int(env.now), len(cpu.waiting)))
        print(f'{int(env.now):<8d} {str(proc):<8s} '
              f'{"start":<8s} {"remain =":<8s} {proc.remain_t:<8.0f} ')

        # run until job finishes or quantum expires
        terminated = env.process(cpu.do_job(env, proc))
        quantum_exp = env.timeout(cpu.quantum)
        # Whichever is triggered first
        yield terminated | quantum_exp
        # If the job didn't finish, the time quantum expired
        if not terminated.triggered:
            terminated.interrupt("stop")
            # Decrement the job's remaining time
            proc.remain_t -= cpu.quantum
            # Take it out of the wait queue -> reschedule it
            cpu.waiting.remove(proc)
            env.process(job_sched(env, proc, cpu))
        else:
            proc.get_trnd(env.now)  # Compute the turnaround time
            print(f'{int(env.now):<8d} {str(proc):<8s} '
                  f'{"finish":<8s} {"trnd time =":<8s} {proc.trnd_t:<8.0f} '
                  f'{"wait time =":<8s} {proc.wait_t:<8.0f} ')
            # For record-keeping, appends all completed jobs to a list
            Record.jobs_completed.append(proc)
            # Remove the completed job from the waiting queue
            cpu.waiting.remove(proc)
            # ---- Terminate at the Nth job ----
            if str(STOP_SIG) in proc.id:
                print("\nSimulation stopped.\nRemaining jobs: %d" % len(cpu.waiting))
                raise simpy.core.StopSimulation(0)
        # execute a context switch (in this case its 0, so does nothing)
        yield env.timeout(cpu.context)
        if cpu.context > 0:
            print("{0:<8d} context switch".format(int(env.now)))

# ------------------------------------------------------------------------------
# ------------------ Initializes the simulation --------------------------------
# ------------------------------------------------------------------------------
def init_sim(env, arg, rand=expovariate):
    cpu = CPU(arg.quantum, arg.context, env, NUM_CORES, )
    i = 0
    while True:
        if arg.test:
            try:
                # Pop the next arrival time
                next_arrive = Record.arrive_times.pop(0)
                current = env.now
                # Run a timer until it is time for the next job
                yield env.timeout(int(next_arrive - current))
            except IndexError:
                env.exit()
        else:
            if i == arg.jobs:
                env.exit()
            yield env.timeout(int(rand(INTER_T)))

        i += 1  # Increment the pid number
        proc = Job(pid="pid %d" % i, arrive_t=env.now, env=env)
        env.proc_data.append(proc)  # for logging
        print("{0:<8d} {1:<8s} {2:<8s} {3:<8s} {4:<8.0f} ".format(
            int(proc.arrive_t), str(proc), "arrive", "service time =", proc.service_t))
        Record.total_t += proc.service_t
        # throws the job into the waiting queue
        env.process(job_sched(env, proc, cpu))

# ------------------------ End Initialization ----------------------------------
# ------------------------------------------------------------------------------


def main(arg):
    random.seed(RAND_SEED)
    # Create the Environment() object
    env = Env()
    print('\nStarting.\n')
    if arg.test is True:
        Record.arrive_times = [0, 10, 15, 80, 90]
        Record.service_times = [75, 40, 25, 20, 45]
        t = time.process_time()
        env.process(init_sim(env, arg))
        env.run(until=arg.runtime)
        env.save_record()
        dt = time.process_time() - t
        summary(dt,arg)
    else:
        t = time.process_time()
        env.process(init_sim(env, arg))
        env.run(until=arg.runtime)
        env.save_record()
        dt = time.process_time() - t
        print("Time elapsed:", dt)
        if arg.plot:
            plots(arg)

    if arg.qt5:
        # launch animated plot Qt5 GUI
        N = 0
        a = (convolve(Record.total_burst,N))
        qApp = QtWidgets.QApplication(sys.argv)
        aw = ApplicationWindow(data1=Record.q_size, data2=convolve(Record.trnd_t, 8))
        aw.setWindowTitle("%s" % 'rr plot')
        aw.show()
        sys.exit(qApp.exec_())
    return


if __name__ == "__main__":
    """
    Using a named tuple to hold optional arguments due to laziness. 
    That way, all arguments can be passed and referenced as if it
    were a class attribute. i.e arg.jobs, arg.runtime
    """
    Arguments = namedtuple('arg', 'runtime quantum context jobs test plot ext qt5')
    parser = argparse.ArgumentParser(
        description='round-robin scheduling simulation.')
    parser.add_argument('-r', '--runtime', type=int, default=SIM_TIME,
                        help='runtime: default = %s' % SIM_TIME)
    parser.add_argument('-q', '--quantum', type=int, default=TIME_QUANTUM,
                        help='time quantum: default = %s' % TIME_QUANTUM)
    parser.add_argument('-c', '--context', type=int, default=CONTEXT_SWITCH,
                        help='context switch: default = %s' % CONTEXT_SWITCH)
    parser.add_argument('-j', '--jobs', type=int, default=MAX_PROC,
                        help='number of jobs: default = %s' % MAX_PROC)
    parser.add_argument('-t', '--test', type=bool, default=False,
                        help='run test case: default = False')
    parser.add_argument('-p', '--plot', type=bool, default=False,
                        help='enable plotting: default = False')
    parser.add_argument('-e', '--ext', type=str, default='svg',
                        help='save plot format: default = png')
    parser.add_argument('-qt', '--qt5', type=bool, default=False,
                        help='qt5 animated plotting: default = %s' % False)

    a = parser.parse_args()
    main(Arguments(**vars(a)))
