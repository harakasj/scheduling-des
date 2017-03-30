"""
RR scheduling sim for single-core cpu
Requires: Python 3.6
Dependencies: simpy, numpy, 
Optional dep: scipy, pyqt5, matplotlib 2.0

Check SimPy domcumentation:
https://media.readthedocs.org/pdf/simpy/latest/simpy.pdf

If no plotting, only simpy and numpy are required

Only string formatting using f-literals is a compatability issue 
which whas introduced in Python 3.6. Otherwise, it should run 
on < Python 3. Just modify or remove print formats with f-literals.

"""

import pickle
import simpy
import time
import random
from random import randint, expovariate
import numpy as np
from array import array
from PlotQt import *

from numpy.core.fromnumeric import mean

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


def plots(fmt='png'):
    # Do the plotting
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from scipy import stats

        path = 'figures/'

        plt.figure()
        plt.hist(Record.q_size,
                 histtype='bar',
                 bins=18,
                 normed=1)
        plt.axis([0, 15, 0, 0.3])
        plt.title("Queue Size (Normalized) (TQ = %d, $\lambda=1/8$, jobs= %d)"
                  % (TIME_QUANTUM, len(Record.jobs_completed)))
        fname = "TQ-%d-Queue-Size.%s" % (TIME_QUANTUM, fmt)
        plt.savefig(path+fname, format=fmt)

        plt.figure()
        print(len(Record.jobs_completed))
        plt.hist(list(x.trnd_t for x in Record.jobs_completed),
                 histtype='bar',
                 bins="auto",
                 normed=0)
        plt.title("Wait Time (TQ = %d, $\lambda=1/8$, jobs= %d)"
                  % (TIME_QUANTUM, len(Record.jobs_completed)))
        plt.axis([0, 100, 0, 250])
        fname = "TQ-%d-Wait-Time.%s" % (TIME_QUANTUM, fmt)
        plt.savefig(path+fname, format=fmt)

        plt.figure()
        plt.hist(list(x.wait_t for x in Record.jobs_completed),
                 histtype='bar',
                 bins="auto",
                 normed=0)
        plt.title("Turnaround Time (TQ = %d, $\lambda=1/8$, jobs= %d)"
                  % (TIME_QUANTUM, len(Record.jobs_completed)))
        plt.axis([0, 100, 0, 250])
        fname = "TQ-%d-Trnd-Time.%s" % (TIME_QUANTUM, fmt)
        plt.savefig(path+fname, format=fmt)

        plt.figure()
        plt.step(Record.q_time,
                 stats.zscore(Record.mean_burst),
                 where="mid",
                 label="Average Service Time")

        plt.step(Record.q_time,
                 stats.zscore(Record.q_size),
                 where="mid",
                 label="Queue Size")
        plt.title("Queue Size vs Time (TQ = %d, $\lambda=1/8$, runtime= %d)"
                  % (TIME_QUANTUM, MAX_PROC))
        plt.show()
    except ImportError as i:
        print("Missing %s" % i)


def convolve(x, N):
    # Just a helper function, to compute a moving average
    cumsum = np.cumsum(np.insert(x, 0, 0))
    m = np.append(np.zeros(N), (cumsum[N:] - cumsum[:-N]) / N)
    return array("d", m)


def summary(dt):
    # Print out your summary data
    avg = lambda x: sum(x) / len(x)
    print("\n================================= "
          "Summary "
          "===================================\n")
    print("Time elapsed:", dt)
    print("Quantum: %d" % TIME_QUANTUM)
    print("Context Switch: %d" % CONTEXT_SWITCH)
    print('Mean turnaround: %1.2f' % avg(list(x.trnd_t for x in Record.jobs_completed)))
    print('Mean wait time: %1.2f' % avg(list(x.wait_t for x in Record.jobs_completed)))
    print('Queue size at finish: %d' % Record.q_size[-1])
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proc_data = []
        # self.StopSimulation = simpy.core.StopSimulation(0)
        # self.StopSimulation.

    def setup(self):
        pass

    def exit(self, value=None):
        self.save_record()
        return super().exit(value)

    def show_queue(self):
        print(*self._queue, sep='\n')

    def get_queue(self):
        return len(self._queue)

    def step(self):
        return simpy.Environment.step(self)

    @staticmethod
    def save_record():
        print("Writing record...")
        pickle.dump(Record, open("bin/Record.p", "wb"))


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
        if Record.run_test is True and len(Record.service_times) > 0:
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

    def _desc(self):
        return simpy.events.Event._desc(self)

    def trigger(self, event):
        return simpy.events.Event.trigger(self, event)

    def succeed(self, value=None):
        return simpy.events.Event.succeed(self, value)

    def fail(self, exception):
        return simpy.events.Event.fail(self, exception)

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
    def __init__(self, quantum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = args
        self.data = []
        self.waiting = []
        self.quantum = quantum

    def request(self, pid, *args, **kwargs):
        if pid:
            self.waiting.append(pid)
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        return super().release(*args, **kwargs)

    def do_job(self, env, pid):
        # Run out the clock
        # Either the job completes or is interrupted
        try:
            yield env.timeout(pid.remain_t)
            Record.total_t -= pid.remain_t
        except simpy.Interrupt as i:
            print("{0:<8d} {1:<9s}".format(int(env.now), str(pid)), end='')
            print(i.cause, end='')
            print('{0:>13s} {1:<10.0f}'.format('remain =', pid.remain_t))
            Record.total_t -= TIME_QUANTUM
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


def job_sched(env, proc, cpu):
    with cpu.request(pid=proc) as request:
        yield request
        # record the job's first time getting a time slice
        if proc.remain_t == proc.service_t:
            proc.initial_wait = env.now - proc.arrive_t
        # Print out queue size at every step.
        print("{0:<8d} Queued:{1:<8d}".format(int(env.now), len(cpu.waiting)))
        print("{0:<8d} {1:<8s} {2:<8s} {3:<8s} {4:<8.0f} "
              .format(int(env.now), str(proc), "start", "remain =", proc.remain_t))

        # run until job finishes or quantum expires
        terminated = env.process(cpu.do_job(env, proc))
        quantum_exp = env.timeout(TIME_QUANTUM)

        # Whichever is triggered first
        yield terminated | quantum_exp
        # If the job didn't finish, the time quantum expired
        if not terminated.triggered:
            terminated.interrupt("stop")
            # Decrement the job's remaining time
            proc.remain_t -= TIME_QUANTUM
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
                print("\n Simulation stopped.\n Remaining jobs: %d" % len(cpu.waiting))
                env.save_record()
                raise simpy.core.StopSimulation(0)

        # execute a context switch (in this case its 0, so does nothing)
        yield env.timeout(CONTEXT_SWITCH)
        if CONTEXT_SWITCH > 0:
            print("{0:<8d} context switch".format(int(env.now)))
            # print("{0:<8d} total time {1:<8d} ".format(int(env.now), Record.total_t))


# ------------------ Initializes the simulation --------------------------------
def init_sim(env, rand=expovariate, *args):
    cpu = CPU(TIME_QUANTUM, env, NUM_CORES, )
    i = 0
    while True:
        if Record.run_test is True:
            try:
                # Pop the next arrival time
                next_arrive = Record.arrive_times.pop(0)
                current = env.now
                # Run a timer until it is time for the next job
                yield env.timeout(int(next_arrive - current))
            except IndexError:
                env.exit()
        else:
            if i == MAX_PROC:
                env.exit()
            yield env.timeout(int(rand(INTER_T)))

        i += 1  # Increment the pid number
        proc = Job(pid="pid %d" % i, arrive_t=env.now, env=env)

        env.proc_data.append(proc)  # for logging

        print("{0:<8d} {1:<8s} {2:<8s} {3:<8s} {4:<8.0f} ".format(
            int(proc.arrive_t), str(proc), "arrive", "service time =", proc.service_t))

        # Keeps track of the total service time of the queue
        Record.total_t += proc.service_t
        # throws the job into the waiting queue
        env.process(job_sched(env, proc, cpu))
        # print("{0:<8d} total time {1:<8d} ".format(int(env.now), Record.total_t))

# ------------------------------------------------------------------------------
# ------------------------ End Initialization ----------------------------------


def test_run(env):
    # This sets the test case parameters
    Record.run_test = True
    Record.arrive_times = [0, 10, 15, 80, 90]
    Record.service_times = [75, 40, 25, 20, 45]
    t = time.process_time()
    env.process(init_sim(env))
    env.run(until=SIM_TIME)
    dt = time.process_time() - t
    summary(dt)


def timed_run(env):
    env.process(init_sim(env))
    env.run(until=SIM_TIME)
    plots()


def main():
    # TODO: argsparse
    random.seed(RAND_SEED)
    # Create the Environment() object
    env = Env()
    print('\nStarting.\n')
    # test_run(env)
    timed_run(env)

    # Uncomment to launch animated plot Qt5 GUI
    # N = 0
    # a = (convolve(Record.total_burst,N))
    # qApp = QtWidgets.QApplication(sys.argv)
    # aw = ApplicationWindow(data1=Record.q_size, data2=convolve(Record.trnd_t, 8))
    # aw.setWindowTitle("%s" % DES)
    # aw.show()
    # sys.exit(qApp.exec_())

if __name__ == "__main__":
    main()
