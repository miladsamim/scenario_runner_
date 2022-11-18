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
    traceback.print_exc()
finally:
    if scenario_runner is not None:
        print("destroyed")
        scenario_runner.destroy()
        del scenario_runner
    comm.Disconnect()
