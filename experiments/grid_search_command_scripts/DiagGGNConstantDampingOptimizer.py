# imports provided by import_statements
import sys
from bpoptim import DiagGGNConstantDampingOptimizer
from bp_dops_integration.runners import BPOptimRunner

# imports above have to import runner and optimizer
runner_cls = BPOptimRunner
optim_cls = DiagGGNConstantDampingOptimizer
runner_hyperparams = {'damping': {'type': float}, 'lr': {'type': float}}

# build runner
runner = runner_cls(optim_cls, runner_hyperparams)

# arguments from command line
runner.run()

# Write command to 'finished.txt'
command = 'python3 {}\n'.format(' '.join(sys.argv))
with open('finished.txt', 'a') as f:
	f.write(command)
