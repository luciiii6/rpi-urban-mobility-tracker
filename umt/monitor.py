import psutil
import subprocess
import time
import sys

print("Start monitoring: ")

command = ["python", "app.py"] + sys.argv[1:]

app_process = subprocess.Popen(command,
                                 stdout = subprocess.PIPE,
                                 stderr = subprocess.PIPE)
stdout, stderr = app_process.communicate()
times_restarted = 0

#a separate function that monitors in the background:
while True:
  time.sleep(1)

  app_status = 0
  
  try:
    app_status = psutil.Process(app_process.pid).status()
  except psutil.NoSuchProcess:
    pass

  if(app_status == psutil.STATUS_STOPPED or
     app_status == psutil.STATUS_DEAD or
     app_status == psutil.STATUS_ZOMBIE or
     app_status == 0):
    
    times_restarted = times_restarted + 1
    print(f"Application died; restarting for {times_restarted} time")
    exit_code = app_process.returncode
    if exit_code == 2 or exit_code == 3:
      break
    
    app_process = subprocess.Popen(command,
                                       stdout = subprocess.DEVNULL,
                                       stderr = subprocess.STDOUT)
    stdout, stderr = app_process.communicate()
    
