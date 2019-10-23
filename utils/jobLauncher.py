import itertools 
import time
import os,sys

from slurmTemplate import SLURM_JOB_TEMPLATE,nodeSetupTemplate

test=False
#jobSettings
datetag=time.strftime("%y%m%d")
taskTemplate="python qicl.py"
batchDir="/project/6024950/edrechsl/qicl/outfiles/batch/"
outDir="/project/6024950/edrechsl/qicl/outfiles/191010_test/"
batchCommandTemplate="sbatch {submitscript}"

parameterDict={
        'numberEvents'  :[100,500,1000],
        'numberShots'   :[100,1000],
        'maxTrials'     :[10,50,100,500],
        'varFormDepth'  :[2,3],
    }

paramPermutation=[]

for key, valList in parameterDict.items():
    permList=[]
    for val in valList:
        permList.append((key,val))
    paramPermutation.append(permList)

allJobs=[]
for jobConf in itertools.product(*paramPermutation):
    argList=[]
    idList=[]
    for idxSet in range(0,len(jobConf)):
        if not len(jobConf[idxSet])==2:
            logger.error("Something went wrong when unpacking batchJob config.")
            raise Exception()
        configArg=jobConf[idxSet][0]
        configVal=jobConf[idxSet][1]
        argList.append("--{0} {1}".format(configArg,configVal))
        idList.append(str(jobConf[idxSet][1]))

    jobID = "_".join(idList)
    print('Setting up job {0}'.format(jobID))
    
    runCommandArguments=" ".join(argList)
    #assembling runscript per node
    runCommand=' '.join([taskTemplate,runCommandArguments])
    runCommand+=" --steerOutDir {0}".format(outDir)
    if not os.path.exists(outDir):
        print("Creating path {0}".format(outDir))
        os.mkdir(outDir)
    
    runScript='_'.join([datetag,jobID,'.sh'])
    runScript=os.path.join(batchDir,runScript)
    runScript=open(runScript,'w')
#    
    runScript.write(nodeSetupTemplate)
    runScript.write(runCommand)
    runScript.close()

    #build submit script and log files
    submitScript="_".join([datetag,jobID,"submit.sh"])
    submitScript=os.path.join(batchDir,submitScript)
    logSubmit=submitScript.replace('.sh','.log')
    #create logfiles
    logExecute="_".join([datetag,jobID,"run.log"])
    logExecute=os.path.join(outDir,logExecute)
    logError=logExecute
    
    submitFile=open(submitScript,'w')
    slurmJob=SLURM_JOB_TEMPLATE.format(
            runscript=runScript,
            inputfiles='',
            logsubmit=logSubmit,
            logerror=logError,
            logexecute=logExecute,
            memory="4GB",
            time="8:00:00",
            cores="10",
            project="ctb-stelzer",
            setup=nodeSetupTemplate,
            logsdir=batchDir,
            jobname="_".join(["job",datetag,jobID]),
            local_scratch="/scratch/",
            outdir=outDir,
            payload=runCommand,
            )
    submitFile.write(slurmJob)
    submitFile.close()
    
    batchCommand=batchCommandTemplate.format(
            submitscript=submitScript
            )
    allJobs.append(batchCommand)
    if test: break

print("Sending {0} jobs".format(len(allJobs)))
for job in allJobs:
    print('{0}'.format(job))
    if not test: os.system(job)
    time.sleep(2)

print("All jobs submitted")
