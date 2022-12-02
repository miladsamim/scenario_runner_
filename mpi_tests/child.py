from mpi4py import MPI
import sys
import time 

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
print("waiting for message")
msg = comm.recv(source=0, tag=MPI.ANY_TAG)
with open('test.txt', mode='w') as fp:
    fp.write('Message received 2')
print(f"Child {rank} msg received ", msg['stats'])
# comm.Send([msg['sensor data']+1, MPI.DOUBLE], dest=0, tag=0)
comm.send({'hi':'ho'}, dest=0, tag=0)

# comm.Disconnect()
MPI.Finalize()
