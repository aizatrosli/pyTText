import time,sys,os



class Logger(object):

    def __init__(self,logpath):
        self.logpath = logpath

    def redirect(self, msg):
        return msg

