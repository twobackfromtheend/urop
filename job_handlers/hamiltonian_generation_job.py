import os

from ifttt_webhook import trigger_event
from job_handlers.hamiltonian import SpinHamiltonian

job_id = os.getenv("PBS_JOBID")
N = int(os.getenv("N"))

trigger_event("job_progress", value1=f"Hamiltonian generation for N={N} started", value2=job_id)

SpinHamiltonian.create_and_save_hamiltonian(8)

trigger_event("job_progress", value1=f"Hamiltonian generation completed", value2=job_id)
