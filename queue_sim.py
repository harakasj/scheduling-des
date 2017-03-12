'''
john harakas
RR scheduling sim for single-core cpu

first go at this, and really hacked it together.
'''

import pickle,random, simpy

RAND_SEED       = 50          # rand seed to reproduce 
TIME_QUANTUM    = 10          # time quantum
SERVICE_TIME    = [1, 12]
NUM_CORES       = 1           # number of cpu cores
CONTEXT_TIME    = 1           # NOT IMP, context switch time
INTER_TIME      = [30, 300]  
INTER_T         = 1/8         # 1 arrival every 8 seconds
SIM_TIME        = 10000

class Env(simpy.Environment):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proc_data = []

# Custom subclass, handles some things         
class Resource(simpy.Resource):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.data = []
            self.waiting = []
            self.queue = []
            
        def request(self, pid, *args, **kwargs):
            if pid:
                self.waiting.append(pid)
            self.data.append((self._env.now, len(self.queue)))
            return super().request(*args, **kwargs)
   
        def release(self ,*args, **kwargs):
            self.data.append((self._env.now, len(self.queue)))
            return super().release(*args, **kwargs)
        
#         def __str__(self, *args, **kwargs):
#             return simpy.Resource.__str__(self, *args, **kwargs)

# Create job/process instance to arrive in queue  
class Job(object):
    def __init__(self,pid,arrive_t):
        self.id         = pid
        self.service_t = random.randint(*SERVICE_TIME) # random service time
        self.arrive_t   = arrive_t
        self.finish_t   = 0
        self.trnd_t     = 0
        self.wait_t     = 0
        self.remain_t  = self.service_t
    
    def get_trnd(self,finish_t):
        self.finish_t = finish_t
        self.trnd_t = self.finish_t - self.arrive_t
        self.wait_t = self.trnd_t - self.service_t
        return self.trnd_t, self.wait_t
    
    def __str__(self, *args, **kwargs):
        return (self.id)

class CPU(object):
    def __init__(self, env, cores, quantum):
        self.env = env
        self.service = Resource(env, cores)
        self.quantum = quantum
        
    def do_job(self,env,pid):
        # the process finishes or the time quantum expires
        try: yield self.env.timeout(pid.remain_t)
        
        except simpy.Interrupt as i:
            print("%d \t %s %s" %(env.now,pid, i.cause))

def job_sched(env, proc, cpu):
    with cpu.service.request(pid=proc.id) as request:
        yield request
        
        print("\t Queued: ",len(cpu.service.waiting))
        print("{0:<8d} {1:<8s} {2:<8s} {3:<8s} {4:<8.0f} ".format( 
            int(env.now), str(proc), "start", 
            "time needed =", proc.remain_t ) 
        )
        
        terminated = env.process(cpu.do_job(env,proc))
        quantum_exp = env.timeout(TIME_QUANTUM)
        
        # Whichever one of these is triggered first
        yield terminated | quantum_exp
        
        if not(terminated.triggered):
            terminated.interrupt(" quantum expired")
            proc.remain_t -= TIME_QUANTUM
            cpu.service.waiting.remove(proc.id)
            env.process(job_sched(env, proc, cpu))
            
        else:
            proc.get_trnd(env.now)
            print("{0:<8d} {1:<8s} {2:<8s} {3:<8s} {4:<8.0f} {5:<8s} {6:<8.0f} ".format( 
                int(env.now), str(proc), "finish", 
                "trnd time =",proc.remain_t, 
                "wait time =",proc.wait_t )
            )
            cpu.service.waiting.remove(proc.id)

def init_sim(env,*args):
    cpu = CPU(env, NUM_CORES, TIME_QUANTUM)
    i = 0
    while True:
        # arrival times are exponentially distributed
        yield env.timeout(random.expovariate(1/8))
#         yield env.timeout(random.randint(*INTER_TIME))
        i += 1
        
        # create a Job() instance
        proc = Job(pid = "pid %d"%i, arrive_t=env.now)
        # for logging
        env.proc_data.append(proc)
        
        print("{0:<8d} {1:<8s} {2:<8s} {3:<8s} {4:<8.0f} ".format( 
            int(env.now), str(proc), "arrive", 
            "service time =", proc.service_t ) 
        )
        env.process(job_sched(env,proc,cpu))

def main():
    random.seed(RAND_SEED)
    env = Env()
    env.process(init_sim(env))
    env.run(until=SIM_TIME)
    
    # this will dump the list of job instances as a binary.
    # can be later unpacked and analyzed without re-running the entire sim
    pickle.dump( env.proc_data, open( "process_data.p", "wb" ) )

if __name__ == "__main__":
    main()

