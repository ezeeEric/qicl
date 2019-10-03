nodeSetupTemplate="""\
lsetup \'views LCG_95apython3 x86_64-centos7-gcc7-opt\'; \
cd $HOME/projects/ctb-stelzer/edrechsl/qicl/; \
source ./venv_qicl/bin/activate;\
echo $VIRTUAL_ENV
"""

#lsetup \"views LCG_95apython3 x86_64-centos7-gcc7-opt\"; \
SLURM_JOB_TEMPLATE ="""\
#!/bin/bash
#SBATCH --time={time}
#SBATCH --cpus-per-task={cores}
#SBATCH --mem={memory}
#SBATCH --account={project}
#SBATCH --job-name={jobname}
#SBATCH --error={logsdir}/%x.e%A
#SBATCH --output={logsdir}/%x.o%A
source /project/atlas/Tier3/AtlasUserSiteSetup.sh

#directly executed after container setup
export ALRB_CONT_POSTSETUP="pwd; whoami; date; hostname -f; date -u; {setup}"

#export variables needed within Container environment - SINGULARITYENV_ affix needed
export SINGULARITYENV_SLURM_SUBMIT_DIR=${{SLURM_SUBMIT_DIR}}
export SINGULARITYENV_SLURM_JOB_NAME=${{SLURM_JOB_NAME}}
export SINGULARITYENV_SLURM_JOB_USER=${{SLURM_JOB_USER}}
export SINGULARITYENV_SLURM_JOB_ID=${{SLURM_JOB_ID}}
export SINGULARITYENV_HOSTNAME=${{HOSTNAME}}

export SINGULARITYENV_TMPDIR=/{local_scratch}/${{SLURM_JOB_USER}}/${{SLURM_JOB_ID}}
export SINGULARITYENV_LOGSDIR=${{SLURM_SUBMIT_DIR}}/{outdir}/{logsdir}
export SINGULARITYENV_OUTDIR=${{SLURM_SUBMIT_DIR}}/{outdir}

#job
export ALRB_CONT_RUNPAYLOAD=\"{payload}\"

#execute (by setting up container)
setupATLAS -c centos7
"""
