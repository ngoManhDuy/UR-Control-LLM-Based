from src import UR
import socket
import threading
import select
import re
import numpy as np
import time

DEFAULT_TIMEOUT = 1.0
#%%
class ConnectionState:
    ERROR = 0
    DISCONNECTED = 1
    CONNECTED = 2
    PAUSED = 3
    STARTED = 4
#%%
class RealTimeClient(object):

    def __init__(self, robotModel):
        '''
        Constructor see class description for more info.
        '''
        if(False):
            assert isinstance(robotModel, UR.robotModel.RobotModel)
        self.__robotModel = robotModel

        logger = UR.dataLogging.DataLogging()
        name = logger.AddEventLogging(__name__, log2Consol=False,level = UR.logging.WARNING)
        self.__logger = logger.__dict__[name]
        self.__robotModel.rtcConnectionState = ConnectionState.DISCONNECTED
        self.__reconnectTimeout = 60
        self.__sock = None
        self.__thread = None
        if self.__connect():
            self.__logger.info('RT_CLient constructor done')
        else:
            self.__logger.info('RT_CLient constructor done but not connected')
#%%
    def __connect(self):
        '''
        Initialize RT Client connection to host .
        
        Return value:
        success (boolean)
        
        Example:
        rob = URBasic.realTimeClient.RT_CLient('192.168.56.101')
        rob.connect()
        '''       
        if self.__sock:
            return True

        t0 = time.time()
        while (time.time()-t0<self.__reconnectTimeout) and self.__robotModel.rtcConnectionState < ConnectionState.CONNECTED:
            try:
                self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)            
                self.__sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)         
                self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.__sock.settimeout(DEFAULT_TIMEOUT)
                self.__sock.connect((self.__robotModel.ipAddress, 30003))
                self.__robotModel.rtcConnectionState = ConnectionState.CONNECTED
                time.sleep(0.5)
                self.__logger.info('Connected')
                return True
            except (socket.timeout, socket.error):
                self.__sock = None
                self.__logger.error('RTC connecting')

        return False
#%%                
    def Disconnect(self):
        '''
        Disconnect the RT Client connection.
        '''        
        if self.__sock:
            self.__sock.close()
            self.__sock = None
            self.__logger.info('Disconnected')
        self.__robotModel.rtcConnectionState = ConnectionState.DISCONNECTED
        return True
#%%
    def IsRtcConnected(self):
        '''
        Returns True if the connection is open.

        Return value:
        status (boolean): True if connected and False of not connected.

        Example:
        rob = URBasic.realTimeClient.RT_CLient('192.168.56.101')
        rob.connect()
        print(rob.is_connected())
        rob.disconnect()
        '''
        return self.__robotModel.rtcConnectionState > ConnectionState.DISCONNECTED
#%%       
    def SendProgram(self,prg=''):
        '''
        Send a new command or program (string) to the UR controller. 
        The command or program will be executed as soon as it's received by the UR controller. 
        Sending a new command or program while stop and existing running command or program and start the new one.
        The program or command will also bee modified to include some control signals to be used
        for monitoring if a program execution is successful and finished.  

        Input parameters:
        prg (string): A string containing a single command or a whole program.

        Example:
        rob = URBasic.realTimeClient.RT_CLient('192.168.56.101',logger=logger)
        rob.connect()
        rob.send_srt('set_digital_out(0, True)')
        rob.disconnect()        
        '''
        if not self.IsRtcConnected():
            if not self.__connect():
                self.__logger.error('SendProgram: Not connected to robot')
 
        if self.__robotModel.stopRunningFlag:
            self.__logger.info('SendProgram: Send program aborted due to stopRunningFlag')
            return
 
        #Close down previous thread 
        if self.__thread is not None:
            if self.__robotModel.rtcProgramRunning:
                self.__robotModel.stopRunningFlag = True
                while self.__robotModel.rtcProgramRunning: time.sleep(0.1)
                self.__robotModel.stopRunningFlag = False
            self.__thread.join()
            
        
        #Rest status bits
        self.__robotModel.rtcProgramRunning = True
        self.__robotModel.rtcProgramExecutionError = False
        
        #Send and wait from program
        self.__sendPrg(self.__AddStatusBit2Prog(prg))        
        self.__thread = threading.Thread(target=self.__waitForProgram2Finish, kwargs={'prg': prg})
        self.__thread.start()
        #self.__waitForProgram2Finish(prg)
#%%            
    def Send(self,prg=''):
        '''
        Send a new command (string) to the UR controller. 
        The command or program will be executed as soon as it's received by the UR controller. 
        Sending a new command or program while stop and existing running command or program and start the new one.
        The program or command will also bee modified to include some control signals to be used
        for monitoring if a program execution is successful and finished.  

        Input parameters:
        prg (string): A string containing a single command or a whole program.


        Example:
        rob = URBasic.realTimeClient.RT_CLient('192.168.56.101',logger=logger)
        rob.connect()
        rob.send_srt('set_digital_out(0, True)')
        rob.disconnect()        
        '''
        if not self.IsRtcConnected():
            if not self.__connect():
                self.__logger.error('SendProgram: Not connected to robot')
        if self.__robotModel.stopRunningFlag:
            self.__logger.info('SendProgram: Send command aborted due to stopRunningFlag')
            return
    
        #Rest status bits
        self.__robotModel.rtcProgramRunning = True
        self.__robotModel.rtcProgramExecutionError = False
        
        #Send
        self.__sendPrg(prg)      
        self.__robotModel.rtcProgramRunning = False
#%%
    def __AddStatusBit2Prog(self,prg):
        '''
        Modifying program to include status bit's in beginning and end of program
        '''
        def1 = prg.find('def ')
        if def1>=0:
            prglen = len(prg)
            prg = prg.replace('):\n', '):\n  write_output_boolean_register(0, True)\n',1)
            if len(prg) == prglen:
                self.__logger.warning('Send_program: Syntax error in program')
                return False
                
            if (len(re.findall('def ', prg)))>1:
                mainprg = prg[0:prg[def1+4:].find('def ')+def1+4]
                mainPrgEnd = (np.max([mainprg.rfind('end '), mainprg.rfind('end\n')]))
                prg = prg.replace(prg[0:mainPrgEnd], prg[0:mainPrgEnd] + '\n  write_output_boolean_register(1, True)\n',1)
            else:
                mainPrgEnd = prg.rfind('end')
                prg = prg.replace(prg[0:mainPrgEnd], prg[0:mainPrgEnd] + '\n  write_output_boolean_register(1, True)\n',1)
                
        else:
            prg = 'def script():\n  write_output_boolean_register(0, True)\n  ' + prg + '\n  write_output_boolean_register(1, True)\nend\n'
        return prg
#%%       
    def __sendPrg(self,prg):
        '''
        Sending program str via socket
        '''
        programSend = False      
        self.__robotModel.forceRemoteActiveFlag = False
        while not self.__robotModel.stopRunningFlag and not programSend:
            try:
                (_, writable, _) = select.select([], [self.__sock], [], DEFAULT_TIMEOUT)
                if len(writable):
                    self.__sock.send(prg.encode('utf-8'))
                    self.__logger.info('Program send to Robot:\n' + prg)
                    programSend = True
            except:
                self.__sock = None
                self.__robotModel.rtcConnectionState = ConnectionState.ERROR
                self.__logger.warning('Could not send program!')
                self.__connect()                
        if not programSend:
            self.__robotModel.rtcProgramRunning = False
            self.__logger.error('Program re-sending timed out - Could not send program!')
        time.sleep(0.1)
#%%
    def __waitForProgram2Finish(self,prg):
        '''
        waiting for program to finish
        '''
        waitForProgramStart = len(prg)/50
        notrun = 0
        prgRest = 'def resetRegister():\n  write_output_boolean_register(0, False)\n  write_output_boolean_register(1, False)\nend\n'
        while not self.__robotModel.stopRunningFlag and self.__robotModel.rtcProgramRunning:            
            if self.__robotModel.SafetyStatus().StoppedDueToSafety:
                self.__robotModel.rtcProgramRunning = False
                self.__robotModel.rtcProgramExecutionError = True
                self.__logger.error('SendProgram: Safety Stop')
            elif self.__robotModel.OutputBitRegister()[0] == False:
                self.__logger.debug('sendProgram: Program not started')
                notrun += 1
                if notrun > waitForProgramStart:
                    self.__robotModel.rtcProgramRunning = False
                    self.__logger.error('sendProgram: Program not able to run')
            elif self.__robotModel.OutputBitRegister()[0] == True and self.__robotModel.OutputBitRegister()[1] == True:
                self.__robotModel.rtcProgramRunning = False
                self.__logger.info('sendProgram: Finished')
            elif self.__robotModel.OutputBitRegister()[0] == True:
                if self.__robotModel.RobotStatus().ProgramRunning:
                    self.__logger.debug('sendProgram: UR running')
                    notrun = 0
                else:
                    notrun += 1
                    if notrun>10:
                        self.__robotModel.rtcProgramRunning = False
                        self.__robotModel.rtcProgramExecutionError = True
                        self.__logger.error('SendProgram: Program Stopped but not finiched!!!')    
            else:
                self.__robotModel.rtcProgramRunning = False
                self.__logger.error('SendProgram: Unknown error')
            time.sleep(0.05)
        self.__sendPrg(prgRest)
        self.__robotModel.rtcProgramRunning = False
        