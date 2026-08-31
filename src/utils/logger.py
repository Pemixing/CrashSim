# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os, re, datetime, shutil
import logging, sys


# class Logger(object):
#     '''
#     "Static" class to handle logging.
#     '''
#     log_file = None

#     @staticmethod
#     def init(log_path):
#         Logger.log_file = log_path

#     @staticmethod
#     def log(write_str):
#         print(write_str)
#         if not Logger.log_file:
#             print('Logger must be initialized before logging!')
#             return
#         time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#         with open(Logger.log_file, 'a') as f:
#             f.write(time_str + '  ')
#             f.write(str(write_str) + '\n')

class Logger(object):
    '''
    "Static" class to handle logging.
    '''
    log_file = None
    logger = None

    @staticmethod
    def init(log_path):
        Logger.log_file = log_path

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%Y-%m-%d_%H:%M:%S',
                            filename=Logger.log_file,
                            filemode='w')
        
        Logger.logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        Logger.logger.addHandler(handler)

    @staticmethod
    def log(write_str):
        if not Logger.log_file:
            print('Logger must be initialized before logging!')
            return
        Logger.logger.info(write_str)

    def setSilent():
        Logger.logger = logging.getLogger(name=__name__)
        Logger.logger.propagate = False

    def log_root(write_str):
        logging.info(write_str)

def throw_err(err_str):
    '''
    Logs and throws a runtime error.
    '''
    Logger.log('ERROR: %s' % (err_str))
    raise RuntimeError(err_str)