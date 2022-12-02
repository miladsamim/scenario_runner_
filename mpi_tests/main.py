from mpi4py import MPI
import sys
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
status = MPI.Status()

script = 'child.py'

icomms = []

d = {'sensor data': np.zeros((4,32,16,16,3)),
     'stats': 'stats',
     'other': [1,2,3,4]}
for i in range(100):
    try:
        print('Trying to spawn child process...')
        icomm = MPI.COMM_SELF.Spawn(sys.executable, args=[script], maxprocs=1, root=0)
        # time.sleep(10)
        icomm.send(d, dest=0, tag=11)
        print("Sleeping for 10 secs")
        icomms.append(icomm)
        req = icomm.recv(source=0, tag=MPI.ANY_TAG)
        data = req
        print("Parent received from child", data)
        # time.sleep(1)
        icomm.Disconnect()
        print('Spawned a child.')
    except: 
        ValueError('Spawn failed to start.')

data = None#np.empty((4,32,16,16,4), dtype=np.float64)
# data = {}
# data = req.wait()
# data = req.wait()
print(MPI.Status)
MPI.Finalize()