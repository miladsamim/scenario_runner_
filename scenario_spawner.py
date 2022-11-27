import traceback
from srunner.tools import dotdict

from scenario_runner import ScenarioRunner
from mpi4py import MPI

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

print("Waiting for specification")
scenario_args = comm.recv(source=0, tag=MPI.ANY_TAG)
scenario_args = dotdict(scenario_args)
try:
    scenario_runner = ScenarioRunner(scenario_args)
    result = scenario_runner.run()
except Exception:   # pylint: disable=broad-except
    raise Exception(traceback.print_exc())
finally:
    if scenario_runner is not None:
        scenario_runner.destroy()
        del scenario_runner
        # if scenario_runner.manager.use_mpi: # when errors happens in scenario side
        comm.send({'done':True,
                    'cleaned up': True}, dest=0, tag=1)#(source=0, tag=MPI.ANY_TAG)
    # comm.Disconnect()
