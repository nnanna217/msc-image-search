import os
import subprocess
import sys

# os.system("nohup sh -c '" +
#           sys.executable + " model.py --lr 0.01 >res1.txt && " +
#           sys.executable + " model.py --lr 0.03 >res2.txt && " +
#           sys.executable + " model.py --lr 0.09 >res3.txt" +
#           "' &")
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)
#
# # Add additional paths to Python path
# content_path = os.path.join(script_dir, 'content')
# source_path = os.path.join(script_dir, 'offline_training.py')
#
# sys.path.append(content_path)
# sys.path.append(source_path)
#
# # Define the nohup command
# nohup_command = "nohup python offline_training.py > output.log 2>&1 &"
#
# # Execute the nohup command using subprocess
# subprocess.Popen(nohup_command, shell=True)

os.system("nohup sh -c python '" + sys.executable + "cp_fashion-classifier.py > res1.txt"+"' &")
