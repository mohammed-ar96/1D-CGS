# # -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 03:40:57 2024

@author: لينوفو
"""

import time

start_time = None
# log_file_path = 'algorithm_times.txt'

log_file_path2 = 'training_times.txt'

def start():
    """
    Record the start time for measuring elapsed time.
    """
    global start_time
    start_time = time.time()

def elapsedTime(algorithm_name,g):
        # file.write(log_entry)
        elapsed_time = time.time() - start_time
        log_entry = f"\n Algorithm: {algorithm_name}, Elapsed time: {elapsed_time:.4f} seconds \n"
        # print(log_entry)
        
        # Store the time in a .txt file
        with open(log_file_path, 'a') as file:
            file.write('\n'+g+'\n')
            file.write(log_entry)
            file.write('-'*70)
        
        return elapsed_time


def elapsedTime2(algorithm_name,g):
        # file.write(log_entry)
        elapsed_time = time.time() - start_time
        log_entry = f"\n Algorithm: {algorithm_name}, Elapsed time: {elapsed_time:.4f} seconds \n"
        # print(log_entry)
        
        # Store the time in a .txt file
        with open(log_file_path2, 'a') as file:
            file.write('\n'+g+'\n')
            file.write(log_entry)
            file.write('-'*70)
        
        return elapsed_time


# def secondsToStr(t):
#     return str(timedelta(seconds=t))

# line = "="*40
# def log(s, elapsed=None):
#     print(line)
#     print(secondsToStr(time.perf_counter()), '-', s)
#     if elapsed:
#         print("Elapsed time:", elapsed)
#         with open(graph.GraphName+"_Time.txt", 'a') as convert_file: 
#             convert_file.write("\n ============================================================== \n")
#             convert_file.write("\n =====*****************************************************==== \n")
#             convert_file.write("elapsed time: ")
#             convert_file.write(elapsed)
#             convert_file.write("\n ============================================================== \n")
#             convert_file.write("\n =====*****************************************************==== \n")
#     print(line)
#     print()

# def endlog():
#     end = time.perf_counter()
#     elapsed = end-start
#     with open(graph.GraphName+"_Time.txt", 'a') as convert_file: 
#         convert_file.write("\n ============================================================== \n")
#         convert_file.write("\n =====*****************************************************==== \n")
#         convert_file.write(" End Program:: ")
#         convert_file.write(secondsToStr(elapsed))
#         convert_file.write("\n ============================================================== \n")
#         convert_file.write("\n =====*****************************************************==== \n")
#     log("End Program", secondsToStr(elapsed))

# def now():
#     return secondsToStr(time.time())

# start = time.perf_counter()
# atexit.register(endlog)
# log("Start Program")
