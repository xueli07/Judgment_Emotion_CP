#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 十一月 12, 2025, at 14:11
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'Control'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'age': '',
    'handness': ['right', 'left'],
    'gender': ['female', 'male'],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 720]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath=__file__,
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('instr1_key_resp') is None:
        # initialise instr1_key_resp
        instr1_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr1_key_resp',
        )
    if deviceManager.getDevice('instr2_key_resp') is None:
        # initialise instr2_key_resp
        instr2_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr2_key_resp',
        )
    if deviceManager.getDevice('instr3_key_resp') is None:
        # initialise instr3_key_resp
        instr3_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr3_key_resp',
        )
    if deviceManager.getDevice('instr4_key_resp') is None:
        # initialise instr4_key_resp
        instr4_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr4_key_resp',
        )
    if deviceManager.getDevice('instr5_key_resp') is None:
        # initialise instr5_key_resp
        instr5_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr5_key_resp',
        )
    if deviceManager.getDevice('prac_1back_face_key_resp') is None:
        # initialise prac_1back_face_key_resp
        prac_1back_face_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_1back_face_key_resp',
        )
    if deviceManager.getDevice('prac_1back_end_key_resp') is None:
        # initialise prac_1back_end_key_resp
        prac_1back_end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_1back_end_key_resp',
        )
    if deviceManager.getDevice('form_1back_begin_key_resp') is None:
        # initialise form_1back_begin_key_resp
        form_1back_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='form_1back_begin_key_resp',
        )
    if deviceManager.getDevice('form_1back_face_key_resp') is None:
        # initialise form_1back_face_key_resp
        form_1back_face_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='form_1back_face_key_resp',
        )
    if deviceManager.getDevice('form_1back_break_key_resp') is None:
        # initialise form_1back_break_key_resp
        form_1back_break_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='form_1back_break_key_resp',
        )
    if deviceManager.getDevice('xab_instr1_key_resp') is None:
        # initialise xab_instr1_key_resp
        xab_instr1_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='xab_instr1_key_resp',
        )
    if deviceManager.getDevice('xab_instr2_key_resp') is None:
        # initialise xab_instr2_key_resp
        xab_instr2_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='xab_instr2_key_resp',
        )
    if deviceManager.getDevice('xab_instr3_key_resp') is None:
        # initialise xab_instr3_key_resp
        xab_instr3_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='xab_instr3_key_resp',
        )
    if deviceManager.getDevice('prac_xab_ab_key_resp') is None:
        # initialise prac_xab_ab_key_resp
        prac_xab_ab_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_xab_ab_key_resp',
        )
    if deviceManager.getDevice('prac_xab_end_key_resp') is None:
        # initialise prac_xab_end_key_resp
        prac_xab_end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_xab_end_key_resp',
        )
    if deviceManager.getDevice('xab_ab_key_resp') is None:
        # initialise xab_ab_key_resp
        xab_ab_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='xab_ab_key_resp',
        )
    if deviceManager.getDevice('xab_break_key_resp') is None:
        # initialise xab_break_key_resp
        xab_break_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='xab_break_key_resp',
        )
    if deviceManager.getDevice('end_key_resp') is None:
        # initialise end_key_resp
        end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instr1" ---
    instr1_top_text = visual.TextStim(win=win, name='instr1_top_text',
        text='实验指导语\n',
        font='Arial',
        pos=(0, 0.18), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr1_center_text = visual.TextStim(win=win, name='instr1_center_text',
        text='您好，欢迎参加本实验！\n\n在本实验中，您一共需要完成2个任务，\n\n整个实验过程大约需要35分钟，\n\n请您仔细阅读后面的指导语。\n\n按“空格”键查看更多说明\n',
        font='Arial',
        pos=(0, -0.05), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr1_key_resp = keyboard.Keyboard(deviceName='instr1_key_resp')
    
    # --- Initialize components for Routine "instr2" ---
    instr2_center_text = visual.TextStim(win=win, name='instr2_center_text',
        text='在第一个任务中，您会看到一系列面孔图片，\n\n这些面孔来自两个不同的人（一位女性和一位男性）。\n\n每个人都有多张不同表情的面孔，\n\n也就是说，同一个人的面孔会以不同的表情多次出现。\n\n您的任务是判断：当前呈现的面孔，是否与前一张面孔属于同一个人，\n\n如果是同一个人，请按"m"键；如果是不同的人，请按"x"键。\n\n\n按“空格”键继续',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr2_key_resp = keyboard.Keyboard(deviceName='instr2_key_resp')
    
    # --- Initialize components for Routine "instr3" ---
    instr3_top_text = visual.TextStim(win=win, name='instr3_top_text',
        text='【任务流程】\n\n每个试次开始时，屏幕中央会出现一个 ‘+’，接着，该 ‘+’会消失，\n\n然后，屏幕中央会呈现一张面孔图片（如下所示）。\n\n您可以在面孔呈现期间以及面孔消失后的1秒内按键反应，\n\n如果您超过1秒未按键，系统将记录为"缺失反应"，随后继续下一个试次。\n\n请您尽量避免缺失反应。',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr3_face_image = visual.ImageStim(
        win=win,
        name='instr3_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    instr3_key_resp = keyboard.Keyboard(deviceName='instr3_key_resp')
    instr3_bottom_text = visual.TextStim(win=win, name='instr3_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "instr4" ---
    instr4_center_text = visual.TextStim(win=win, name='instr4_center_text',
        text='【请注意】：\n\n1.由于第一张面孔没有“前一张”可以比较，所以在第一个试次中，您可以任意按"m"或"x"键。\n\n2.即使同一个人的表情不同，只要是同一个人，就应该判断为“相同”。\n',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr4_key_resp = keyboard.Keyboard(deviceName='instr4_key_resp')
    instr4_bottom_text = visual.TextStim(win=win, name='instr4_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "instr5" ---
    instr5_top_text = visual.TextStim(win=win, name='instr5_top_text',
        text='【练习实验说明】\n\n正式实验开始前，您将先进行22次练习，以熟悉任务流程。\n\n如果还有任何疑问，请联系主试。\n\n如果您已完全理解任务要求，按“空格”键开始练习。',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr5_bottom_text = visual.TextStim(win=win, name='instr5_bottom_text',
        text='按“空格”键开始练习实验',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr5_key_resp = keyboard.Keyboard(deviceName='instr5_key_resp')
    
    # --- Initialize components for Routine "prac_1back_face" ---
    prac_1back_face_text = visual.TextStim(win=win, name='prac_1back_face_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_1back_face_image = visual.ImageStim(
        win=win,
        name='prac_1back_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    prac_1back_face_ISI_text = visual.TextStim(win=win, name='prac_1back_face_ISI_text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_1back_face_x_text = visual.TextStim(win=win, name='prac_1back_face_x_text',
        text='不相同（x）',
        font='Arial',
        pos=(-0.25, -0.3), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_1back_face_m_text = visual.TextStim(win=win, name='prac_1back_face_m_text',
        text='相同（m）',
        font='Arial',
        pos=(0.25, -0.3), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    prac_1back_face_key_resp = keyboard.Keyboard(deviceName='prac_1back_face_key_resp')
    # Run 'Begin Experiment' code from prac_1back_face_code
    #########
    # 以下代码为了保存数据
    prac_1back_block = 1  # block计数器（每次重新开始练习时+1）
    prac_1back_trial = 0  # trial计数器（每个试次+1）
    
    # --- Initialize components for Routine "prac_1back_end" ---
    prac_1back_end_center_text = visual.TextStim(win=win, name='prac_1back_end_center_text',
        text='恭喜！您已经完成所有练习试次，接下来将开始正式实验。\n\n该任务总共分为3个部分，每完成一部分后会有休息时间。您可以休息1分钟，感觉准备好了就继续。\n\n整个任务大约需要20分钟。\n\n【请注意】：\n\n实验结束后，我们会检查您实验的认真程度，其与您的被试费挂钩，请您务必认真作答。\n\n如果仍需练习，请按“P”键返回练习。如果仍有疑问，请联系主试。\n\n如果您已完全理解任务要求，请“空格‘键开始正式实验。',
        font='Arial',
        pos=(-0.21, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_1back_end_bottom_text = visual.TextStim(win=win, name='prac_1back_end_bottom_text',
        text='按“空格”键开始正式实验',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_1back_end_key_resp = keyboard.Keyboard(deviceName='prac_1back_end_key_resp')
    # Run 'Begin Experiment' code from prac_1back_end_code
    # 初始化变量，只在实验开始时运行一次
    needPractice = True
    
    # --- Initialize components for Routine "form_1back_begin" ---
    form_1back_begin_text = visual.TextStim(win=win, name='form_1back_begin_text',
        text='按“空格”键开始',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    form_1back_begin_key_resp = keyboard.Keyboard(deviceName='form_1back_begin_key_resp')
    
    # --- Initialize components for Routine "form_1back_face" ---
    form_1back_face_text = visual.TextStim(win=win, name='form_1back_face_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    form_1back_face_image = visual.ImageStim(
        win=win,
        name='form_1back_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    form_1back_face_ISI = visual.TextStim(win=win, name='form_1back_face_ISI',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    form_1back_face_x_text = visual.TextStim(win=win, name='form_1back_face_x_text',
        text='不相同（x）',
        font='Arial',
        pos=(-0.25, -0.3), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    form_1back_face_m_text = visual.TextStim(win=win, name='form_1back_face_m_text',
        text='相同（m）',
        font='Arial',
        pos=(0.25, -0.3), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    form_1back_face_key_resp = keyboard.Keyboard(deviceName='form_1back_face_key_resp')
    # Run 'Begin Experiment' code from form_1back_face_code
    # 以下代码为了保存数据
    form_1back_block = 1  # block计数器
    form_1back_trial = 0  # trial计数器
    
    # 初始化行范围（第一个block读取0-88行，Python索引从0开始）
    rowRange = '0:88'
    
    # --- Initialize components for Routine "form_1back_break" ---
    form_1back_break_center_text = visual.TextStim(win=win, name='form_1back_break_center_text',
        text='现在休息1分钟\n\n1分钟后，如果您感觉准备好了，您需要按“空格”键继续实验',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    form_1back_break_bottom_text = visual.TextStim(win=win, name='form_1back_break_bottom_text',
        text='按“空格”键继续实验',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    form_1back_break_key_resp = keyboard.Keyboard(deviceName='form_1back_break_key_resp')
    
    # --- Initialize components for Routine "xab_instr1" ---
    xab_instr1_top_text = visual.TextStim(win=win, name='xab_instr1_top_text',
        text='恭喜！您已完成本实验的第一个任务，\n\n接下来我们继续进行情绪识别相关的第二个任务。\n\n在本任务中，首先，你会先看到一张面孔（如下所示），',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    xab_instr1_X_image = visual.ImageStim(
        win=win,
        name='xab_instr1_X_image', 
        image='prac_control_faces/m_0_f1.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    xab_instr1_bottom_text = visual.TextStim(win=win, name='xab_instr1_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    xab_instr1_key_resp = keyboard.Keyboard(deviceName='xab_instr1_key_resp')
    
    # --- Initialize components for Routine "xab_instr2" ---
    xab_instr2_top_text = visual.TextStim(win=win, name='xab_instr2_top_text',
        text='接着，面孔图片消失后，您会同时看到两张面孔图片（如下所示），\n\n您需要判断这两张面孔中的哪一张与第一张面孔更相似\n\n如果你认为左边面孔与第一张面孔更相似，按“F”键，\n\n如果你认为右边面孔与第一张面孔更相似，按“J”键。',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    xab_instr2_bottom_text = visual.TextStim(win=win, name='xab_instr2_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    xab_instr2_face_a_image = visual.ImageStim(
        win=win,
        name='xab_instr2_face_a_image', 
        image='prac_control_faces/m_10_f1.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.15, -0.1), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    xab_instr2_face_b_image = visual.ImageStim(
        win=win,
        name='xab_instr2_face_b_image', 
        image='prac_control_faces/m_0_f1.png', mask=None, anchor='center',
        ori=0.0, pos=(0.15, -0.1), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    xab_instr2_key_resp = keyboard.Keyboard(deviceName='xab_instr2_key_resp')
    
    # --- Initialize components for Routine "xab_instr3" ---
    xab_instr3_center_text = visual.TextStim(win=win, name='xab_instr3_center_text',
        text='【练习实验说明】\n\n\n【重要】：请您凭第一直觉【又快又准】地做出每一次判断。\n\n正式实验开始前，您将先进行14次练习，以熟悉任务流程。\n\n如果还有任何疑问，请联系主试。\n\n如果您已完全理解任务要求，按"空格"键开始练习。',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    xab_instr3_bottom_text = visual.TextStim(win=win, name='xab_instr3_bottom_text',
        text='按“空格”键开始练习实验',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    xab_instr3_key_resp = keyboard.Keyboard(deviceName='xab_instr3_key_resp')
    
    # --- Initialize components for Routine "prac_xab_x" ---
    prac_xab_x_cross_text = visual.TextStim(win=win, name='prac_xab_x_cross_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_xab_x_image = visual.ImageStim(
        win=win,
        name='prac_xab_x_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "prac_xab_blank1" ---
    prac_xab_blank1_text = visual.TextStim(win=win, name='prac_xab_blank1_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_xab_ab" ---
    prac_xab_ab_a_image = visual.ImageStim(
        win=win,
        name='prac_xab_ab_a_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.15, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac_xab_ab_b_image = visual.ImageStim(
        win=win,
        name='prac_xab_ab_b_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.15, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    prac_xab_ab_key_resp = keyboard.Keyboard(deviceName='prac_xab_ab_key_resp')
    # Run 'Begin Experiment' code from prac_xab_ab_code
    #########
    # 以下代码为了保存数据
    prac_xab_block = 1  # block计数器（每次重新开始练习时+1）
    prac_xab_trial = 0  # trial计数器（每个试次+1）
    
    # --- Initialize components for Routine "prac_xab_blank2" ---
    pracl_xab_blank2_text = visual.TextStim(win=win, name='pracl_xab_blank2_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_xab_end" ---
    prac_xab_end_center_text = visual.TextStim(win=win, name='prac_xab_end_center_text',
        text='恭喜！您已经完成所有练习试次，接下来将开始正式实验。\n\n该任务总共分为4个部分，每完成一部分后会有休息时间。您可以休息1分钟，感觉准备好了就继续。\n\n整个任务大约需要16分钟。\n\n\n【请注意】：\n\n1.请您凭第一直觉【又快又准】地做出每一次判断。\n\n2.实验结束后，我们会检查您实验的认真程度，其与您的被试费挂钩，请您务必认真作答。\n\n\n如果仍需练习，请按“P”键返回练习。如果仍有疑问，请联系主试。\n\n如果您已完全理解任务要求，请空格键开始正式实验。',
        font='Arial',
        pos=(-0.21, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_xab_end_bottom_text = visual.TextStim(win=win, name='prac_xab_end_bottom_text',
        text='按“空格”键开始正式实验',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_xab_end_key_resp = keyboard.Keyboard(deviceName='prac_xab_end_key_resp')
    # Run 'Begin Experiment' code from prac_xan_end_code
    # 初始化变量，只在实验开始时运行一次
    needPractice = True
    
    # --- Initialize components for Routine "xab_x" ---
    xab_x_cross_text = visual.TextStim(win=win, name='xab_x_cross_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    xab_x_image = visual.ImageStim(
        win=win,
        name='xab_x_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "xab_blank1" ---
    xab_blank1_text = visual.TextStim(win=win, name='xab_blank1_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "xab_ab" ---
    xab_ab_a_image = visual.ImageStim(
        win=win,
        name='xab_ab_a_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.15, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    xab_ab_b_image = visual.ImageStim(
        win=win,
        name='xab_ab_b_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.15, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    xab_ab_key_resp = keyboard.Keyboard(deviceName='xab_ab_key_resp')
    # Run 'Begin Experiment' code from xab_ab_code
    #########
    # 以下代码为了保存数据
    formal_xab_block = 1  # block计数器
    formal_xab_trial = 0  # trial计数器
    
    # --- Initialize components for Routine "xab_blank2" ---
    xab_blank2_text_2 = visual.TextStim(win=win, name='xab_blank2_text_2',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "xab_break" ---
    xab_break_center_text = visual.TextStim(win=win, name='xab_break_center_text',
        text='现在休息1分钟\n\n1分钟后，如果您感觉准备好了，您需要按“空格”键继续实验',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    xab_break_bottom_text = visual.TextStim(win=win, name='xab_break_bottom_text',
        text='按“空格”键继续实验',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    xab_break_key_resp = keyboard.Keyboard(deviceName='xab_break_key_resp')
    
    # --- Initialize components for Routine "end" ---
    end_center_text = visual.TextStim(win=win, name='end_center_text',
        text='您已经完成全部实验！\n\n请按“空格”键退出实验，并联系主试。',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_key_resp = keyboard.Keyboard(deviceName='end_key_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instr1" ---
    # create an object to store info about Routine instr1
    instr1 = data.Routine(
        name='instr1',
        components=[instr1_top_text, instr1_center_text, instr1_key_resp],
    )
    instr1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr1_key_resp
    instr1_key_resp.keys = []
    instr1_key_resp.rt = []
    _instr1_key_resp_allKeys = []
    # store start times for instr1
    instr1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr1.tStart = globalClock.getTime(format='float')
    instr1.status = STARTED
    thisExp.addData('instr1.started', instr1.tStart)
    instr1.maxDuration = None
    # keep track of which components have finished
    instr1Components = instr1.components
    for thisComponent in instr1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr1" ---
    instr1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr1_top_text* updates
        
        # if instr1_top_text is starting this frame...
        if instr1_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr1_top_text.frameNStart = frameN  # exact frame index
            instr1_top_text.tStart = t  # local t and not account for scr refresh
            instr1_top_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr1_top_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr1_top_text.started')
            # update status
            instr1_top_text.status = STARTED
            instr1_top_text.setAutoDraw(True)
        
        # if instr1_top_text is active this frame...
        if instr1_top_text.status == STARTED:
            # update params
            pass
        
        # *instr1_center_text* updates
        
        # if instr1_center_text is starting this frame...
        if instr1_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr1_center_text.frameNStart = frameN  # exact frame index
            instr1_center_text.tStart = t  # local t and not account for scr refresh
            instr1_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr1_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr1_center_text.started')
            # update status
            instr1_center_text.status = STARTED
            instr1_center_text.setAutoDraw(True)
        
        # if instr1_center_text is active this frame...
        if instr1_center_text.status == STARTED:
            # update params
            pass
        
        # *instr1_key_resp* updates
        waitOnFlip = False
        
        # if instr1_key_resp is starting this frame...
        if instr1_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr1_key_resp.frameNStart = frameN  # exact frame index
            instr1_key_resp.tStart = t  # local t and not account for scr refresh
            instr1_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr1_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr1_key_resp.started')
            # update status
            instr1_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr1_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr1_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr1_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr1_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr1_key_resp_allKeys.extend(theseKeys)
            if len(_instr1_key_resp_allKeys):
                instr1_key_resp.keys = _instr1_key_resp_allKeys[-1].name  # just the last key pressed
                instr1_key_resp.rt = _instr1_key_resp_allKeys[-1].rt
                instr1_key_resp.duration = _instr1_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr1" ---
    for thisComponent in instr1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr1
    instr1.tStop = globalClock.getTime(format='float')
    instr1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr1.stopped', instr1.tStop)
    # check responses
    if instr1_key_resp.keys in ['', [], None]:  # No response was made
        instr1_key_resp.keys = None
    thisExp.addData('instr1_key_resp.keys',instr1_key_resp.keys)
    if instr1_key_resp.keys != None:  # we had a response
        thisExp.addData('instr1_key_resp.rt', instr1_key_resp.rt)
        thisExp.addData('instr1_key_resp.duration', instr1_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr2" ---
    # create an object to store info about Routine instr2
    instr2 = data.Routine(
        name='instr2',
        components=[instr2_center_text, instr2_key_resp],
    )
    instr2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr2_key_resp
    instr2_key_resp.keys = []
    instr2_key_resp.rt = []
    _instr2_key_resp_allKeys = []
    # store start times for instr2
    instr2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr2.tStart = globalClock.getTime(format='float')
    instr2.status = STARTED
    thisExp.addData('instr2.started', instr2.tStart)
    instr2.maxDuration = None
    # keep track of which components have finished
    instr2Components = instr2.components
    for thisComponent in instr2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr2" ---
    instr2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr2_center_text* updates
        
        # if instr2_center_text is starting this frame...
        if instr2_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr2_center_text.frameNStart = frameN  # exact frame index
            instr2_center_text.tStart = t  # local t and not account for scr refresh
            instr2_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr2_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr2_center_text.started')
            # update status
            instr2_center_text.status = STARTED
            instr2_center_text.setAutoDraw(True)
        
        # if instr2_center_text is active this frame...
        if instr2_center_text.status == STARTED:
            # update params
            pass
        
        # *instr2_key_resp* updates
        waitOnFlip = False
        
        # if instr2_key_resp is starting this frame...
        if instr2_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr2_key_resp.frameNStart = frameN  # exact frame index
            instr2_key_resp.tStart = t  # local t and not account for scr refresh
            instr2_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr2_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr2_key_resp.started')
            # update status
            instr2_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr2_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr2_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr2_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr2_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr2_key_resp_allKeys.extend(theseKeys)
            if len(_instr2_key_resp_allKeys):
                instr2_key_resp.keys = _instr2_key_resp_allKeys[-1].name  # just the last key pressed
                instr2_key_resp.rt = _instr2_key_resp_allKeys[-1].rt
                instr2_key_resp.duration = _instr2_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr2" ---
    for thisComponent in instr2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr2
    instr2.tStop = globalClock.getTime(format='float')
    instr2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr2.stopped', instr2.tStop)
    # check responses
    if instr2_key_resp.keys in ['', [], None]:  # No response was made
        instr2_key_resp.keys = None
    thisExp.addData('instr2_key_resp.keys',instr2_key_resp.keys)
    if instr2_key_resp.keys != None:  # we had a response
        thisExp.addData('instr2_key_resp.rt', instr2_key_resp.rt)
        thisExp.addData('instr2_key_resp.duration', instr2_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr3" ---
    # create an object to store info about Routine instr3
    instr3 = data.Routine(
        name='instr3',
        components=[instr3_top_text, instr3_face_image, instr3_key_resp, instr3_bottom_text],
    )
    instr3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instr3_face_image.setImage('prac_control_faces/m_0_m1.png')
    # create starting attributes for instr3_key_resp
    instr3_key_resp.keys = []
    instr3_key_resp.rt = []
    _instr3_key_resp_allKeys = []
    # store start times for instr3
    instr3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr3.tStart = globalClock.getTime(format='float')
    instr3.status = STARTED
    thisExp.addData('instr3.started', instr3.tStart)
    instr3.maxDuration = None
    # keep track of which components have finished
    instr3Components = instr3.components
    for thisComponent in instr3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr3" ---
    instr3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr3_top_text* updates
        
        # if instr3_top_text is starting this frame...
        if instr3_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr3_top_text.frameNStart = frameN  # exact frame index
            instr3_top_text.tStart = t  # local t and not account for scr refresh
            instr3_top_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr3_top_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr3_top_text.started')
            # update status
            instr3_top_text.status = STARTED
            instr3_top_text.setAutoDraw(True)
        
        # if instr3_top_text is active this frame...
        if instr3_top_text.status == STARTED:
            # update params
            pass
        
        # *instr3_face_image* updates
        
        # if instr3_face_image is starting this frame...
        if instr3_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            instr3_face_image.frameNStart = frameN  # exact frame index
            instr3_face_image.tStart = t  # local t and not account for scr refresh
            instr3_face_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr3_face_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr3_face_image.started')
            # update status
            instr3_face_image.status = STARTED
            instr3_face_image.setAutoDraw(True)
        
        # if instr3_face_image is active this frame...
        if instr3_face_image.status == STARTED:
            # update params
            pass
        
        # *instr3_key_resp* updates
        waitOnFlip = False
        
        # if instr3_key_resp is starting this frame...
        if instr3_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr3_key_resp.frameNStart = frameN  # exact frame index
            instr3_key_resp.tStart = t  # local t and not account for scr refresh
            instr3_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr3_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr3_key_resp.started')
            # update status
            instr3_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr3_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr3_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr3_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr3_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr3_key_resp_allKeys.extend(theseKeys)
            if len(_instr3_key_resp_allKeys):
                instr3_key_resp.keys = _instr3_key_resp_allKeys[-1].name  # just the last key pressed
                instr3_key_resp.rt = _instr3_key_resp_allKeys[-1].rt
                instr3_key_resp.duration = _instr3_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr3_bottom_text* updates
        
        # if instr3_bottom_text is starting this frame...
        if instr3_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr3_bottom_text.frameNStart = frameN  # exact frame index
            instr3_bottom_text.tStart = t  # local t and not account for scr refresh
            instr3_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr3_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr3_bottom_text.started')
            # update status
            instr3_bottom_text.status = STARTED
            instr3_bottom_text.setAutoDraw(True)
        
        # if instr3_bottom_text is active this frame...
        if instr3_bottom_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr3" ---
    for thisComponent in instr3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr3
    instr3.tStop = globalClock.getTime(format='float')
    instr3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr3.stopped', instr3.tStop)
    # check responses
    if instr3_key_resp.keys in ['', [], None]:  # No response was made
        instr3_key_resp.keys = None
    thisExp.addData('instr3_key_resp.keys',instr3_key_resp.keys)
    if instr3_key_resp.keys != None:  # we had a response
        thisExp.addData('instr3_key_resp.rt', instr3_key_resp.rt)
        thisExp.addData('instr3_key_resp.duration', instr3_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr4" ---
    # create an object to store info about Routine instr4
    instr4 = data.Routine(
        name='instr4',
        components=[instr4_center_text, instr4_key_resp, instr4_bottom_text],
    )
    instr4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr4_key_resp
    instr4_key_resp.keys = []
    instr4_key_resp.rt = []
    _instr4_key_resp_allKeys = []
    # Run 'Begin Routine' code from instr4_code
    #########
    # 控制文字的左对齐
    instr4_center_text.alignText = 'left'
    # store start times for instr4
    instr4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr4.tStart = globalClock.getTime(format='float')
    instr4.status = STARTED
    thisExp.addData('instr4.started', instr4.tStart)
    instr4.maxDuration = None
    # keep track of which components have finished
    instr4Components = instr4.components
    for thisComponent in instr4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr4" ---
    instr4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr4_center_text* updates
        
        # if instr4_center_text is starting this frame...
        if instr4_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr4_center_text.frameNStart = frameN  # exact frame index
            instr4_center_text.tStart = t  # local t and not account for scr refresh
            instr4_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr4_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr4_center_text.started')
            # update status
            instr4_center_text.status = STARTED
            instr4_center_text.setAutoDraw(True)
        
        # if instr4_center_text is active this frame...
        if instr4_center_text.status == STARTED:
            # update params
            pass
        
        # *instr4_key_resp* updates
        waitOnFlip = False
        
        # if instr4_key_resp is starting this frame...
        if instr4_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr4_key_resp.frameNStart = frameN  # exact frame index
            instr4_key_resp.tStart = t  # local t and not account for scr refresh
            instr4_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr4_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr4_key_resp.started')
            # update status
            instr4_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr4_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr4_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr4_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr4_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr4_key_resp_allKeys.extend(theseKeys)
            if len(_instr4_key_resp_allKeys):
                instr4_key_resp.keys = _instr4_key_resp_allKeys[-1].name  # just the last key pressed
                instr4_key_resp.rt = _instr4_key_resp_allKeys[-1].rt
                instr4_key_resp.duration = _instr4_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr4_bottom_text* updates
        
        # if instr4_bottom_text is starting this frame...
        if instr4_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr4_bottom_text.frameNStart = frameN  # exact frame index
            instr4_bottom_text.tStart = t  # local t and not account for scr refresh
            instr4_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr4_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr4_bottom_text.started')
            # update status
            instr4_bottom_text.status = STARTED
            instr4_bottom_text.setAutoDraw(True)
        
        # if instr4_bottom_text is active this frame...
        if instr4_bottom_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr4.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr4" ---
    for thisComponent in instr4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr4
    instr4.tStop = globalClock.getTime(format='float')
    instr4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr4.stopped', instr4.tStop)
    # check responses
    if instr4_key_resp.keys in ['', [], None]:  # No response was made
        instr4_key_resp.keys = None
    thisExp.addData('instr4_key_resp.keys',instr4_key_resp.keys)
    if instr4_key_resp.keys != None:  # we had a response
        thisExp.addData('instr4_key_resp.rt', instr4_key_resp.rt)
        thisExp.addData('instr4_key_resp.duration', instr4_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    continue_prac_1back = data.TrialHandler2(
        name='continue_prac_1back',
        nReps=999.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(continue_prac_1back)  # add the loop to the experiment
    thisContinue_prac_1back = continue_prac_1back.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisContinue_prac_1back.rgb)
    if thisContinue_prac_1back != None:
        for paramName in thisContinue_prac_1back:
            globals()[paramName] = thisContinue_prac_1back[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisContinue_prac_1back in continue_prac_1back:
        currentLoop = continue_prac_1back
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisContinue_prac_1back.rgb)
        if thisContinue_prac_1back != None:
            for paramName in thisContinue_prac_1back:
                globals()[paramName] = thisContinue_prac_1back[paramName]
        
        # --- Prepare to start Routine "instr5" ---
        # create an object to store info about Routine instr5
        instr5 = data.Routine(
            name='instr5',
            components=[instr5_top_text, instr5_bottom_text, instr5_key_resp],
        )
        instr5.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for instr5_key_resp
        instr5_key_resp.keys = []
        instr5_key_resp.rt = []
        _instr5_key_resp_allKeys = []
        # store start times for instr5
        instr5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instr5.tStart = globalClock.getTime(format='float')
        instr5.status = STARTED
        thisExp.addData('instr5.started', instr5.tStart)
        instr5.maxDuration = None
        # keep track of which components have finished
        instr5Components = instr5.components
        for thisComponent in instr5.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instr5" ---
        # if trial has changed, end Routine now
        if isinstance(continue_prac_1back, data.TrialHandler2) and thisContinue_prac_1back.thisN != continue_prac_1back.thisTrial.thisN:
            continueRoutine = False
        instr5.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *instr5_top_text* updates
            
            # if instr5_top_text is starting this frame...
            if instr5_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr5_top_text.frameNStart = frameN  # exact frame index
                instr5_top_text.tStart = t  # local t and not account for scr refresh
                instr5_top_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr5_top_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr5_top_text.started')
                # update status
                instr5_top_text.status = STARTED
                instr5_top_text.setAutoDraw(True)
            
            # if instr5_top_text is active this frame...
            if instr5_top_text.status == STARTED:
                # update params
                pass
            
            # *instr5_bottom_text* updates
            
            # if instr5_bottom_text is starting this frame...
            if instr5_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr5_bottom_text.frameNStart = frameN  # exact frame index
                instr5_bottom_text.tStart = t  # local t and not account for scr refresh
                instr5_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr5_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr5_bottom_text.started')
                # update status
                instr5_bottom_text.status = STARTED
                instr5_bottom_text.setAutoDraw(True)
            
            # if instr5_bottom_text is active this frame...
            if instr5_bottom_text.status == STARTED:
                # update params
                pass
            
            # *instr5_key_resp* updates
            waitOnFlip = False
            
            # if instr5_key_resp is starting this frame...
            if instr5_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr5_key_resp.frameNStart = frameN  # exact frame index
                instr5_key_resp.tStart = t  # local t and not account for scr refresh
                instr5_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr5_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr5_key_resp.started')
                # update status
                instr5_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(instr5_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(instr5_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if instr5_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = instr5_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _instr5_key_resp_allKeys.extend(theseKeys)
                if len(_instr5_key_resp_allKeys):
                    instr5_key_resp.keys = _instr5_key_resp_allKeys[-1].name  # just the last key pressed
                    instr5_key_resp.rt = _instr5_key_resp_allKeys[-1].rt
                    instr5_key_resp.duration = _instr5_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instr5.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instr5.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instr5" ---
        for thisComponent in instr5.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instr5
        instr5.tStop = globalClock.getTime(format='float')
        instr5.tStopRefresh = tThisFlipGlobal
        thisExp.addData('instr5.stopped', instr5.tStop)
        # check responses
        if instr5_key_resp.keys in ['', [], None]:  # No response was made
            instr5_key_resp.keys = None
        continue_prac_1back.addData('instr5_key_resp.keys',instr5_key_resp.keys)
        if instr5_key_resp.keys != None:  # we had a response
            continue_prac_1back.addData('instr5_key_resp.rt', instr5_key_resp.rt)
            continue_prac_1back.addData('instr5_key_resp.duration', instr5_key_resp.duration)
        # the Routine "instr5" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        prac_1back = data.TrialHandler2(
            name='prac_1back',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('prac_control_oneback.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(prac_1back)  # add the loop to the experiment
        thisPrac_1back = prac_1back.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_1back.rgb)
        if thisPrac_1back != None:
            for paramName in thisPrac_1back:
                globals()[paramName] = thisPrac_1back[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPrac_1back in prac_1back:
            currentLoop = prac_1back
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPrac_1back.rgb)
            if thisPrac_1back != None:
                for paramName in thisPrac_1back:
                    globals()[paramName] = thisPrac_1back[paramName]
            
            # --- Prepare to start Routine "prac_1back_face" ---
            # create an object to store info about Routine prac_1back_face
            prac_1back_face = data.Routine(
                name='prac_1back_face',
                components=[prac_1back_face_text, prac_1back_face_image, prac_1back_face_ISI_text, prac_1back_face_x_text, prac_1back_face_m_text, prac_1back_face_key_resp],
            )
            prac_1back_face.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac_1back_face_image.setImage(prac_control_face)
            # create starting attributes for prac_1back_face_key_resp
            prac_1back_face_key_resp.keys = []
            prac_1back_face_key_resp.rt = []
            _prac_1back_face_key_resp_allKeys = []
            # Run 'Begin Routine' code from prac_1back_face_code
            # 初始化反馈状态
            prac_1back_face_x_text.color = 'black'
            prac_1back_face_m_text.color = 'black'
            feedback_given = False
            # store start times for prac_1back_face
            prac_1back_face.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_1back_face.tStart = globalClock.getTime(format='float')
            prac_1back_face.status = STARTED
            thisExp.addData('prac_1back_face.started', prac_1back_face.tStart)
            prac_1back_face.maxDuration = None
            # keep track of which components have finished
            prac_1back_faceComponents = prac_1back_face.components
            for thisComponent in prac_1back_face.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_1back_face" ---
            # if trial has changed, end Routine now
            if isinstance(prac_1back, data.TrialHandler2) and thisPrac_1back.thisN != prac_1back.thisTrial.thisN:
                continueRoutine = False
            prac_1back_face.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_1back_face_text* updates
                
                # if prac_1back_face_text is starting this frame...
                if prac_1back_face_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_1back_face_text.frameNStart = frameN  # exact frame index
                    prac_1back_face_text.tStart = t  # local t and not account for scr refresh
                    prac_1back_face_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_1back_face_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_1back_face_text.started')
                    # update status
                    prac_1back_face_text.status = STARTED
                    prac_1back_face_text.setAutoDraw(True)
                
                # if prac_1back_face_text is active this frame...
                if prac_1back_face_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_1back_face_text is stopping this frame...
                if prac_1back_face_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_1back_face_text.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_1back_face_text.tStop = t  # not accounting for scr refresh
                        prac_1back_face_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_1back_face_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_1back_face_text.stopped')
                        # update status
                        prac_1back_face_text.status = FINISHED
                        prac_1back_face_text.setAutoDraw(False)
                
                # *prac_1back_face_image* updates
                
                # if prac_1back_face_image is starting this frame...
                if prac_1back_face_image.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    prac_1back_face_image.frameNStart = frameN  # exact frame index
                    prac_1back_face_image.tStart = t  # local t and not account for scr refresh
                    prac_1back_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_1back_face_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_1back_face_image.started')
                    # update status
                    prac_1back_face_image.status = STARTED
                    prac_1back_face_image.setAutoDraw(True)
                
                # if prac_1back_face_image is active this frame...
                if prac_1back_face_image.status == STARTED:
                    # update params
                    pass
                
                # if prac_1back_face_image is stopping this frame...
                if prac_1back_face_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_1back_face_image.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_1back_face_image.tStop = t  # not accounting for scr refresh
                        prac_1back_face_image.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_1back_face_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_1back_face_image.stopped')
                        # update status
                        prac_1back_face_image.status = FINISHED
                        prac_1back_face_image.setAutoDraw(False)
                
                # *prac_1back_face_ISI_text* updates
                
                # if prac_1back_face_ISI_text is starting this frame...
                if prac_1back_face_ISI_text.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                    # keep track of start time/frame for later
                    prac_1back_face_ISI_text.frameNStart = frameN  # exact frame index
                    prac_1back_face_ISI_text.tStart = t  # local t and not account for scr refresh
                    prac_1back_face_ISI_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_1back_face_ISI_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_1back_face_ISI_text.started')
                    # update status
                    prac_1back_face_ISI_text.status = STARTED
                    prac_1back_face_ISI_text.setAutoDraw(True)
                
                # if prac_1back_face_ISI_text is active this frame...
                if prac_1back_face_ISI_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_1back_face_ISI_text is stopping this frame...
                if prac_1back_face_ISI_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_1back_face_ISI_text.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_1back_face_ISI_text.tStop = t  # not accounting for scr refresh
                        prac_1back_face_ISI_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_1back_face_ISI_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_1back_face_ISI_text.stopped')
                        # update status
                        prac_1back_face_ISI_text.status = FINISHED
                        prac_1back_face_ISI_text.setAutoDraw(False)
                
                # *prac_1back_face_x_text* updates
                
                # if prac_1back_face_x_text is starting this frame...
                if prac_1back_face_x_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    prac_1back_face_x_text.frameNStart = frameN  # exact frame index
                    prac_1back_face_x_text.tStart = t  # local t and not account for scr refresh
                    prac_1back_face_x_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_1back_face_x_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_1back_face_x_text.started')
                    # update status
                    prac_1back_face_x_text.status = STARTED
                    prac_1back_face_x_text.setAutoDraw(True)
                
                # if prac_1back_face_x_text is active this frame...
                if prac_1back_face_x_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_1back_face_x_text is stopping this frame...
                if prac_1back_face_x_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_1back_face_x_text.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_1back_face_x_text.tStop = t  # not accounting for scr refresh
                        prac_1back_face_x_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_1back_face_x_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_1back_face_x_text.stopped')
                        # update status
                        prac_1back_face_x_text.status = FINISHED
                        prac_1back_face_x_text.setAutoDraw(False)
                
                # *prac_1back_face_m_text* updates
                
                # if prac_1back_face_m_text is starting this frame...
                if prac_1back_face_m_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    prac_1back_face_m_text.frameNStart = frameN  # exact frame index
                    prac_1back_face_m_text.tStart = t  # local t and not account for scr refresh
                    prac_1back_face_m_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_1back_face_m_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_1back_face_m_text.started')
                    # update status
                    prac_1back_face_m_text.status = STARTED
                    prac_1back_face_m_text.setAutoDraw(True)
                
                # if prac_1back_face_m_text is active this frame...
                if prac_1back_face_m_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_1back_face_m_text is stopping this frame...
                if prac_1back_face_m_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_1back_face_m_text.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_1back_face_m_text.tStop = t  # not accounting for scr refresh
                        prac_1back_face_m_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_1back_face_m_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_1back_face_m_text.stopped')
                        # update status
                        prac_1back_face_m_text.status = FINISHED
                        prac_1back_face_m_text.setAutoDraw(False)
                
                # *prac_1back_face_key_resp* updates
                waitOnFlip = False
                
                # if prac_1back_face_key_resp is starting this frame...
                if prac_1back_face_key_resp.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    prac_1back_face_key_resp.frameNStart = frameN  # exact frame index
                    prac_1back_face_key_resp.tStart = t  # local t and not account for scr refresh
                    prac_1back_face_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_1back_face_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_1back_face_key_resp.started')
                    # update status
                    prac_1back_face_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_1back_face_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_1back_face_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if prac_1back_face_key_resp is stopping this frame...
                if prac_1back_face_key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_1back_face_key_resp.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_1back_face_key_resp.tStop = t  # not accounting for scr refresh
                        prac_1back_face_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_1back_face_key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_1back_face_key_resp.stopped')
                        # update status
                        prac_1back_face_key_resp.status = FINISHED
                        prac_1back_face_key_resp.status = FINISHED
                if prac_1back_face_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = prac_1back_face_key_resp.getKeys(keyList=['m','x'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_1back_face_key_resp_allKeys.extend(theseKeys)
                    if len(_prac_1back_face_key_resp_allKeys):
                        prac_1back_face_key_resp.keys = _prac_1back_face_key_resp_allKeys[0].name  # just the first key pressed
                        prac_1back_face_key_resp.rt = _prac_1back_face_key_resp_allKeys[0].rt
                        prac_1back_face_key_resp.duration = _prac_1back_face_key_resp_allKeys[0].duration
                # Run 'Each Frame' code from prac_1back_face_code
                # 检查是否有按键反应且还未给过反馈
                if prac_1back_face_key_resp.keys and not feedback_given:
                    if prac_1back_face_key_resp.keys[0] == 'x':
                        prac_1back_face_x_text.color = 'yellow'
                    elif prac_1back_face_key_resp.keys[0] == 'm':
                        prac_1back_face_m_text.color = 'yellow'
                    feedback_given = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_1back_face.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_1back_face.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_1back_face" ---
            for thisComponent in prac_1back_face.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_1back_face
            prac_1back_face.tStop = globalClock.getTime(format='float')
            prac_1back_face.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_1back_face.stopped', prac_1back_face.tStop)
            # check responses
            if prac_1back_face_key_resp.keys in ['', [], None]:  # No response was made
                prac_1back_face_key_resp.keys = None
            prac_1back.addData('prac_1back_face_key_resp.keys',prac_1back_face_key_resp.keys)
            if prac_1back_face_key_resp.keys != None:  # we had a response
                prac_1back.addData('prac_1back_face_key_resp.rt', prac_1back_face_key_resp.rt)
                prac_1back.addData('prac_1back_face_key_resp.duration', prac_1back_face_key_resp.duration)
            # Run 'End Routine' code from prac_1back_face_code
            # 增加试次计数器
            prac_1back_trial += 1
            
            # 提取被试的反应
            if prac_1back_face_key_resp.keys:  # 如果被试按了键
                # 获取第一个按键（因为设置的是first key）
                prac_control_resp = prac_1back_face_key_resp.keys[0]
                # 获取反应时
                prac_control_rt = prac_1back_face_key_resp.rt
                
                # 判断反应是否正确
                # 首先获取当前试次的正确答案
                correct_answer = prac_control_resp_correct  # 这个变量来自Excel文件
                
                # 如果正确答案是'NA'（第一个试次），任何反应都算对
                if correct_answer == 'NA':
                    prac_control_correct = 1  # 标记为正确
                # 否则比较被试的反应和正确答案
                elif prac_control_resp == correct_answer:
                    prac_control_correct = 1  # 正确
                else:
                    prac_control_correct = 0  # 错误
                    
            else:  # 如果被试没有按键（缺失反应）
                prac_control_resp = None
                prac_control_rt = None
                prac_control_correct = 'NA'  
            
            # 将这些变量添加到thisExp，这样它们会被保存到数据文件中
            thisExp.addData('prac_1back_block', prac_1back_block)
            thisExp.addData('prac_1back_trial', prac_1back_trial)
            thisExp.addData('prac_control_resp', prac_control_resp)
            thisExp.addData('prac_control_correct', prac_control_correct)
            thisExp.addData('prac_control_rt', prac_control_rt)
            
            # 可选：打印到控制台，方便调试时查看
            print(f"Trial {prac_1back_trial}: Response={prac_control_resp}, Correct={prac_control_correct}, RT={prac_control_rt}")
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_1back_face.maxDurationReached:
                routineTimer.addTime(-prac_1back_face.maxDuration)
            elif prac_1back_face.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'prac_1back'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "prac_1back_end" ---
        # create an object to store info about Routine prac_1back_end
        prac_1back_end = data.Routine(
            name='prac_1back_end',
            components=[prac_1back_end_center_text, prac_1back_end_bottom_text, prac_1back_end_key_resp],
        )
        prac_1back_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for prac_1back_end_key_resp
        prac_1back_end_key_resp.keys = []
        prac_1back_end_key_resp.rt = []
        _prac_1back_end_key_resp_allKeys = []
        # Run 'Begin Routine' code from prac_1back_end_code
        ###########
        #以下代码是为了按“p”键能够返回练习
        # 清空之前的按键缓存，准备接收新按键
        event.clearEvents()
        
        ##########
        # 如果按了P键重新开始练习，block计数器+1
        prac_1back_block += 1
        
        #########
        # 控制文字的左对齐
        prac_1back_end_center_text.alignText = 'left'
        # store start times for prac_1back_end
        prac_1back_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_1back_end.tStart = globalClock.getTime(format='float')
        prac_1back_end.status = STARTED
        thisExp.addData('prac_1back_end.started', prac_1back_end.tStart)
        prac_1back_end.maxDuration = None
        # keep track of which components have finished
        prac_1back_endComponents = prac_1back_end.components
        for thisComponent in prac_1back_end.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_1back_end" ---
        # if trial has changed, end Routine now
        if isinstance(continue_prac_1back, data.TrialHandler2) and thisContinue_prac_1back.thisN != continue_prac_1back.thisTrial.thisN:
            continueRoutine = False
        prac_1back_end.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_1back_end_center_text* updates
            
            # if prac_1back_end_center_text is starting this frame...
            if prac_1back_end_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_1back_end_center_text.frameNStart = frameN  # exact frame index
                prac_1back_end_center_text.tStart = t  # local t and not account for scr refresh
                prac_1back_end_center_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_1back_end_center_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_1back_end_center_text.started')
                # update status
                prac_1back_end_center_text.status = STARTED
                prac_1back_end_center_text.setAutoDraw(True)
            
            # if prac_1back_end_center_text is active this frame...
            if prac_1back_end_center_text.status == STARTED:
                # update params
                pass
            
            # *prac_1back_end_bottom_text* updates
            
            # if prac_1back_end_bottom_text is starting this frame...
            if prac_1back_end_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_1back_end_bottom_text.frameNStart = frameN  # exact frame index
                prac_1back_end_bottom_text.tStart = t  # local t and not account for scr refresh
                prac_1back_end_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_1back_end_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_1back_end_bottom_text.started')
                # update status
                prac_1back_end_bottom_text.status = STARTED
                prac_1back_end_bottom_text.setAutoDraw(True)
            
            # if prac_1back_end_bottom_text is active this frame...
            if prac_1back_end_bottom_text.status == STARTED:
                # update params
                pass
            
            # *prac_1back_end_key_resp* updates
            waitOnFlip = False
            
            # if prac_1back_end_key_resp is starting this frame...
            if prac_1back_end_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_1back_end_key_resp.frameNStart = frameN  # exact frame index
                prac_1back_end_key_resp.tStart = t  # local t and not account for scr refresh
                prac_1back_end_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_1back_end_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_1back_end_key_resp.started')
                # update status
                prac_1back_end_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_1back_end_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_1back_end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_1back_end_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_1back_end_key_resp.getKeys(keyList=['space','p'], ignoreKeys=["escape"], waitRelease=False)
                _prac_1back_end_key_resp_allKeys.extend(theseKeys)
                if len(_prac_1back_end_key_resp_allKeys):
                    prac_1back_end_key_resp.keys = _prac_1back_end_key_resp_allKeys[-1].name  # just the last key pressed
                    prac_1back_end_key_resp.rt = _prac_1back_end_key_resp_allKeys[-1].rt
                    prac_1back_end_key_resp.duration = _prac_1back_end_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_1back_end.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_1back_end.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_1back_end" ---
        for thisComponent in prac_1back_end.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_1back_end
        prac_1back_end.tStop = globalClock.getTime(format='float')
        prac_1back_end.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_1back_end.stopped', prac_1back_end.tStop)
        # check responses
        if prac_1back_end_key_resp.keys in ['', [], None]:  # No response was made
            prac_1back_end_key_resp.keys = None
        continue_prac_1back.addData('prac_1back_end_key_resp.keys',prac_1back_end_key_resp.keys)
        if prac_1back_end_key_resp.keys != None:  # we had a response
            continue_prac_1back.addData('prac_1back_end_key_resp.rt', prac_1back_end_key_resp.rt)
            continue_prac_1back.addData('prac_1back_end_key_resp.duration', prac_1back_end_key_resp.duration)
        # Run 'End Routine' code from prac_1back_end_code
        # 等待任意按键
        keys = event.waitKeys()  
        
        if 'p' in keys:
            # 继续练习，什么都不做
            pass
        else:
            # 结束练习
            continue_prac_1back.finished = True
        # the Routine "prac_1back_end" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 999.0 repeats of 'continue_prac_1back'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "form_1back_begin" ---
    # create an object to store info about Routine form_1back_begin
    form_1back_begin = data.Routine(
        name='form_1back_begin',
        components=[form_1back_begin_text, form_1back_begin_key_resp],
    )
    form_1back_begin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for form_1back_begin_key_resp
    form_1back_begin_key_resp.keys = []
    form_1back_begin_key_resp.rt = []
    _form_1back_begin_key_resp_allKeys = []
    # store start times for form_1back_begin
    form_1back_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    form_1back_begin.tStart = globalClock.getTime(format='float')
    form_1back_begin.status = STARTED
    thisExp.addData('form_1back_begin.started', form_1back_begin.tStart)
    form_1back_begin.maxDuration = None
    # keep track of which components have finished
    form_1back_beginComponents = form_1back_begin.components
    for thisComponent in form_1back_begin.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "form_1back_begin" ---
    form_1back_begin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *form_1back_begin_text* updates
        
        # if form_1back_begin_text is starting this frame...
        if form_1back_begin_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            form_1back_begin_text.frameNStart = frameN  # exact frame index
            form_1back_begin_text.tStart = t  # local t and not account for scr refresh
            form_1back_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(form_1back_begin_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'form_1back_begin_text.started')
            # update status
            form_1back_begin_text.status = STARTED
            form_1back_begin_text.setAutoDraw(True)
        
        # if form_1back_begin_text is active this frame...
        if form_1back_begin_text.status == STARTED:
            # update params
            pass
        
        # *form_1back_begin_key_resp* updates
        waitOnFlip = False
        
        # if form_1back_begin_key_resp is starting this frame...
        if form_1back_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            form_1back_begin_key_resp.frameNStart = frameN  # exact frame index
            form_1back_begin_key_resp.tStart = t  # local t and not account for scr refresh
            form_1back_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(form_1back_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'form_1back_begin_key_resp.started')
            # update status
            form_1back_begin_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(form_1back_begin_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(form_1back_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if form_1back_begin_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = form_1back_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _form_1back_begin_key_resp_allKeys.extend(theseKeys)
            if len(_form_1back_begin_key_resp_allKeys):
                form_1back_begin_key_resp.keys = _form_1back_begin_key_resp_allKeys[-1].name  # just the last key pressed
                form_1back_begin_key_resp.rt = _form_1back_begin_key_resp_allKeys[-1].rt
                form_1back_begin_key_resp.duration = _form_1back_begin_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            form_1back_begin.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in form_1back_begin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "form_1back_begin" ---
    for thisComponent in form_1back_begin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for form_1back_begin
    form_1back_begin.tStop = globalClock.getTime(format='float')
    form_1back_begin.tStopRefresh = tThisFlipGlobal
    thisExp.addData('form_1back_begin.stopped', form_1back_begin.tStop)
    # check responses
    if form_1back_begin_key_resp.keys in ['', [], None]:  # No response was made
        form_1back_begin_key_resp.keys = None
    thisExp.addData('form_1back_begin_key_resp.keys',form_1back_begin_key_resp.keys)
    if form_1back_begin_key_resp.keys != None:  # we had a response
        thisExp.addData('form_1back_begin_key_resp.rt', form_1back_begin_key_resp.rt)
        thisExp.addData('form_1back_begin_key_resp.duration', form_1back_begin_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "form_1back_begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    form_1back_outer = data.TrialHandler2(
        name='form_1back_outer',
        nReps=3.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(form_1back_outer)  # add the loop to the experiment
    thisForm_1back_outer = form_1back_outer.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisForm_1back_outer.rgb)
    if thisForm_1back_outer != None:
        for paramName in thisForm_1back_outer:
            globals()[paramName] = thisForm_1back_outer[paramName]
    
    for thisForm_1back_outer in form_1back_outer:
        currentLoop = form_1back_outer
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisForm_1back_outer.rgb)
        if thisForm_1back_outer != None:
            for paramName in thisForm_1back_outer:
                globals()[paramName] = thisForm_1back_outer[paramName]
        
        # set up handler to look after randomisation of conditions etc
        form_1back_inner = data.TrialHandler2(
            name='form_1back_inner',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            'formal_control_oneback.xlsx', 
            selection=rowRange
        )
        , 
            seed=None, 
        )
        thisExp.addLoop(form_1back_inner)  # add the loop to the experiment
        thisForm_1back_inner = form_1back_inner.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisForm_1back_inner.rgb)
        if thisForm_1back_inner != None:
            for paramName in thisForm_1back_inner:
                globals()[paramName] = thisForm_1back_inner[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisForm_1back_inner in form_1back_inner:
            currentLoop = form_1back_inner
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisForm_1back_inner.rgb)
            if thisForm_1back_inner != None:
                for paramName in thisForm_1back_inner:
                    globals()[paramName] = thisForm_1back_inner[paramName]
            
            # --- Prepare to start Routine "form_1back_face" ---
            # create an object to store info about Routine form_1back_face
            form_1back_face = data.Routine(
                name='form_1back_face',
                components=[form_1back_face_text, form_1back_face_image, form_1back_face_ISI, form_1back_face_x_text, form_1back_face_m_text, form_1back_face_key_resp],
            )
            form_1back_face.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            form_1back_face_image.setImage(formal_control_face)
            # create starting attributes for form_1back_face_key_resp
            form_1back_face_key_resp.keys = []
            form_1back_face_key_resp.rt = []
            _form_1back_face_key_resp_allKeys = []
            # Run 'Begin Routine' code from form_1back_face_code
            # 初始化反馈状态
            form_1back_face_x_text.color = 'black'
            form_1back_face_m_text.color = 'black'
            feedback_given = False
            # store start times for form_1back_face
            form_1back_face.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            form_1back_face.tStart = globalClock.getTime(format='float')
            form_1back_face.status = STARTED
            thisExp.addData('form_1back_face.started', form_1back_face.tStart)
            form_1back_face.maxDuration = None
            # keep track of which components have finished
            form_1back_faceComponents = form_1back_face.components
            for thisComponent in form_1back_face.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "form_1back_face" ---
            # if trial has changed, end Routine now
            if isinstance(form_1back_inner, data.TrialHandler2) and thisForm_1back_inner.thisN != form_1back_inner.thisTrial.thisN:
                continueRoutine = False
            form_1back_face.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *form_1back_face_text* updates
                
                # if form_1back_face_text is starting this frame...
                if form_1back_face_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    form_1back_face_text.frameNStart = frameN  # exact frame index
                    form_1back_face_text.tStart = t  # local t and not account for scr refresh
                    form_1back_face_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(form_1back_face_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_1back_face_text.started')
                    # update status
                    form_1back_face_text.status = STARTED
                    form_1back_face_text.setAutoDraw(True)
                
                # if form_1back_face_text is active this frame...
                if form_1back_face_text.status == STARTED:
                    # update params
                    pass
                
                # if form_1back_face_text is stopping this frame...
                if form_1back_face_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > form_1back_face_text.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        form_1back_face_text.tStop = t  # not accounting for scr refresh
                        form_1back_face_text.tStopRefresh = tThisFlipGlobal  # on global time
                        form_1back_face_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_1back_face_text.stopped')
                        # update status
                        form_1back_face_text.status = FINISHED
                        form_1back_face_text.setAutoDraw(False)
                
                # *form_1back_face_image* updates
                
                # if form_1back_face_image is starting this frame...
                if form_1back_face_image.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    form_1back_face_image.frameNStart = frameN  # exact frame index
                    form_1back_face_image.tStart = t  # local t and not account for scr refresh
                    form_1back_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(form_1back_face_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_1back_face_image.started')
                    # update status
                    form_1back_face_image.status = STARTED
                    form_1back_face_image.setAutoDraw(True)
                
                # if form_1back_face_image is active this frame...
                if form_1back_face_image.status == STARTED:
                    # update params
                    pass
                
                # if form_1back_face_image is stopping this frame...
                if form_1back_face_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > form_1back_face_image.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        form_1back_face_image.tStop = t  # not accounting for scr refresh
                        form_1back_face_image.tStopRefresh = tThisFlipGlobal  # on global time
                        form_1back_face_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_1back_face_image.stopped')
                        # update status
                        form_1back_face_image.status = FINISHED
                        form_1back_face_image.setAutoDraw(False)
                
                # *form_1back_face_ISI* updates
                
                # if form_1back_face_ISI is starting this frame...
                if form_1back_face_ISI.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                    # keep track of start time/frame for later
                    form_1back_face_ISI.frameNStart = frameN  # exact frame index
                    form_1back_face_ISI.tStart = t  # local t and not account for scr refresh
                    form_1back_face_ISI.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(form_1back_face_ISI, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_1back_face_ISI.started')
                    # update status
                    form_1back_face_ISI.status = STARTED
                    form_1back_face_ISI.setAutoDraw(True)
                
                # if form_1back_face_ISI is active this frame...
                if form_1back_face_ISI.status == STARTED:
                    # update params
                    pass
                
                # if form_1back_face_ISI is stopping this frame...
                if form_1back_face_ISI.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > form_1back_face_ISI.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        form_1back_face_ISI.tStop = t  # not accounting for scr refresh
                        form_1back_face_ISI.tStopRefresh = tThisFlipGlobal  # on global time
                        form_1back_face_ISI.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_1back_face_ISI.stopped')
                        # update status
                        form_1back_face_ISI.status = FINISHED
                        form_1back_face_ISI.setAutoDraw(False)
                
                # *form_1back_face_x_text* updates
                
                # if form_1back_face_x_text is starting this frame...
                if form_1back_face_x_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    form_1back_face_x_text.frameNStart = frameN  # exact frame index
                    form_1back_face_x_text.tStart = t  # local t and not account for scr refresh
                    form_1back_face_x_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(form_1back_face_x_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_1back_face_x_text.started')
                    # update status
                    form_1back_face_x_text.status = STARTED
                    form_1back_face_x_text.setAutoDraw(True)
                
                # if form_1back_face_x_text is active this frame...
                if form_1back_face_x_text.status == STARTED:
                    # update params
                    pass
                
                # if form_1back_face_x_text is stopping this frame...
                if form_1back_face_x_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > form_1back_face_x_text.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        form_1back_face_x_text.tStop = t  # not accounting for scr refresh
                        form_1back_face_x_text.tStopRefresh = tThisFlipGlobal  # on global time
                        form_1back_face_x_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_1back_face_x_text.stopped')
                        # update status
                        form_1back_face_x_text.status = FINISHED
                        form_1back_face_x_text.setAutoDraw(False)
                
                # *form_1back_face_m_text* updates
                
                # if form_1back_face_m_text is starting this frame...
                if form_1back_face_m_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    form_1back_face_m_text.frameNStart = frameN  # exact frame index
                    form_1back_face_m_text.tStart = t  # local t and not account for scr refresh
                    form_1back_face_m_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(form_1back_face_m_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_1back_face_m_text.started')
                    # update status
                    form_1back_face_m_text.status = STARTED
                    form_1back_face_m_text.setAutoDraw(True)
                
                # if form_1back_face_m_text is active this frame...
                if form_1back_face_m_text.status == STARTED:
                    # update params
                    pass
                
                # if form_1back_face_m_text is stopping this frame...
                if form_1back_face_m_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > form_1back_face_m_text.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        form_1back_face_m_text.tStop = t  # not accounting for scr refresh
                        form_1back_face_m_text.tStopRefresh = tThisFlipGlobal  # on global time
                        form_1back_face_m_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_1back_face_m_text.stopped')
                        # update status
                        form_1back_face_m_text.status = FINISHED
                        form_1back_face_m_text.setAutoDraw(False)
                
                # *form_1back_face_key_resp* updates
                waitOnFlip = False
                
                # if form_1back_face_key_resp is starting this frame...
                if form_1back_face_key_resp.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    form_1back_face_key_resp.frameNStart = frameN  # exact frame index
                    form_1back_face_key_resp.tStart = t  # local t and not account for scr refresh
                    form_1back_face_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(form_1back_face_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_1back_face_key_resp.started')
                    # update status
                    form_1back_face_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(form_1back_face_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(form_1back_face_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if form_1back_face_key_resp is stopping this frame...
                if form_1back_face_key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > form_1back_face_key_resp.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        form_1back_face_key_resp.tStop = t  # not accounting for scr refresh
                        form_1back_face_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        form_1back_face_key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_1back_face_key_resp.stopped')
                        # update status
                        form_1back_face_key_resp.status = FINISHED
                        form_1back_face_key_resp.status = FINISHED
                if form_1back_face_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = form_1back_face_key_resp.getKeys(keyList=['m','x'], ignoreKeys=["escape"], waitRelease=False)
                    _form_1back_face_key_resp_allKeys.extend(theseKeys)
                    if len(_form_1back_face_key_resp_allKeys):
                        form_1back_face_key_resp.keys = _form_1back_face_key_resp_allKeys[0].name  # just the first key pressed
                        form_1back_face_key_resp.rt = _form_1back_face_key_resp_allKeys[0].rt
                        form_1back_face_key_resp.duration = _form_1back_face_key_resp_allKeys[0].duration
                # Run 'Each Frame' code from form_1back_face_code
                # 检查是否有按键反应且还未给过反馈
                if form_1back_face_key_resp.keys and not feedback_given:
                    if form_1back_face_key_resp.keys[0] == 'x':
                        form_1back_face_x_text.color = 'yellow'
                    elif form_1back_face_key_resp.keys[0] == 'm':
                        form_1back_face_m_text.color = 'yellow'
                    feedback_given = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    form_1back_face.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in form_1back_face.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "form_1back_face" ---
            for thisComponent in form_1back_face.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for form_1back_face
            form_1back_face.tStop = globalClock.getTime(format='float')
            form_1back_face.tStopRefresh = tThisFlipGlobal
            thisExp.addData('form_1back_face.stopped', form_1back_face.tStop)
            # check responses
            if form_1back_face_key_resp.keys in ['', [], None]:  # No response was made
                form_1back_face_key_resp.keys = None
            form_1back_inner.addData('form_1back_face_key_resp.keys',form_1back_face_key_resp.keys)
            if form_1back_face_key_resp.keys != None:  # we had a response
                form_1back_inner.addData('form_1back_face_key_resp.rt', form_1back_face_key_resp.rt)
                form_1back_inner.addData('form_1back_face_key_resp.duration', form_1back_face_key_resp.duration)
            # Run 'End Routine' code from form_1back_face_code
            # 增加试次计数器
            form_1back_trial += 1
            # 提取被试的反应
            if form_1back_face_key_resp.keys:  # 如果被试按了键
                # 获取第一个按键（因为设置的是first key）
                form_control_resp = form_1back_face_key_resp.keys[0]
                # 获取反应时
                form_control_rt = form_1back_face_key_resp.rt
            
                # 判断反应是否正确
                # 首先获取当前试次的正确答案
                correct_answer = formal_control_resp_correct  # 这个变量来自Excel文件
            
                # 如果正确答案是'NA'（第一个试次），任何反应都算对
                if correct_answer == 'NA':
                    form_control_correct = 1  # 标记为正确
                # 否则比较被试的反应和正确答案
                elif form_control_resp == correct_answer:
                    form_control_correct = 1  # 正确
                else:
                    form_control_correct = 0  # 错误
            
            else:  # 如果被试没有按键（缺失反应）
                form_control_resp = None
                form_control_rt = None
                form_control_correct = 'NA'
            
            # 将这些变量添加到thisExp，这样它们会被保存到数据文件中
            thisExp.addData('form_1back_block', form_1back_block)
            thisExp.addData('form_1back_trial', form_1back_trial)
            thisExp.addData('form_control_resp', form_control_resp)
            thisExp.addData('form_control_correct', form_control_correct)
            thisExp.addData('form_control_rt', form_control_rt)
            
            # 可选：打印到控制台，方便调试时查看
            print(f"Trial {form_1back_trial}: Response={form_control_resp}, Correct={form_control_correct}, RT={form_control_rt}")
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if form_1back_face.maxDurationReached:
                routineTimer.addTime(-form_1back_face.maxDuration)
            elif form_1back_face.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'form_1back_inner'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "form_1back_break" ---
        # create an object to store info about Routine form_1back_break
        form_1back_break = data.Routine(
            name='form_1back_break',
            components=[form_1back_break_center_text, form_1back_break_bottom_text, form_1back_break_key_resp],
        )
        form_1back_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for form_1back_break_key_resp
        form_1back_break_key_resp.keys = []
        form_1back_break_key_resp.rt = []
        _form_1back_break_key_resp_allKeys = []
        # Run 'Begin Routine' code from form_1back_break_code
        # 每次休息，block计数器+1
        form_1back_block += 1
        
        # 根据当前block设置下一个block要读取的行范围
        if form_1back_block == 2:
            rowRange = '88:176'  # 第二个block读取第89到176行（索引88-175）
        elif form_1back_block == 3:
            rowRange = '176:264'  # 第三个block读取第177到264行（索引176-263）
        
        # 显示实验进度
        progress_text = visual.TextStim(
            win=win,
            text=f'已完成 {form_1back_block-1}/3 部分',font='Arial',
            pos=[0, 0.1],
            height=0.03,
            color='black'
        )
        
        # 记录文本开始显示的时间
        text_start_time = core.getTime()
        # store start times for form_1back_break
        form_1back_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        form_1back_break.tStart = globalClock.getTime(format='float')
        form_1back_break.status = STARTED
        thisExp.addData('form_1back_break.started', form_1back_break.tStart)
        form_1back_break.maxDuration = None
        # keep track of which components have finished
        form_1back_breakComponents = form_1back_break.components
        for thisComponent in form_1back_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "form_1back_break" ---
        # if trial has changed, end Routine now
        if isinstance(form_1back_outer, data.TrialHandler2) and thisForm_1back_outer.thisN != form_1back_outer.thisTrial.thisN:
            continueRoutine = False
        form_1back_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *form_1back_break_center_text* updates
            
            # if form_1back_break_center_text is starting this frame...
            if form_1back_break_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                form_1back_break_center_text.frameNStart = frameN  # exact frame index
                form_1back_break_center_text.tStart = t  # local t and not account for scr refresh
                form_1back_break_center_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(form_1back_break_center_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'form_1back_break_center_text.started')
                # update status
                form_1back_break_center_text.status = STARTED
                form_1back_break_center_text.setAutoDraw(True)
            
            # if form_1back_break_center_text is active this frame...
            if form_1back_break_center_text.status == STARTED:
                # update params
                pass
            
            # if form_1back_break_center_text is stopping this frame...
            if form_1back_break_center_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > form_1back_break_center_text.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    form_1back_break_center_text.tStop = t  # not accounting for scr refresh
                    form_1back_break_center_text.tStopRefresh = tThisFlipGlobal  # on global time
                    form_1back_break_center_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_1back_break_center_text.stopped')
                    # update status
                    form_1back_break_center_text.status = FINISHED
                    form_1back_break_center_text.setAutoDraw(False)
            
            # *form_1back_break_bottom_text* updates
            
            # if form_1back_break_bottom_text is starting this frame...
            if form_1back_break_bottom_text.status == NOT_STARTED and tThisFlip >= 60-frameTolerance:
                # keep track of start time/frame for later
                form_1back_break_bottom_text.frameNStart = frameN  # exact frame index
                form_1back_break_bottom_text.tStart = t  # local t and not account for scr refresh
                form_1back_break_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(form_1back_break_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'form_1back_break_bottom_text.started')
                # update status
                form_1back_break_bottom_text.status = STARTED
                form_1back_break_bottom_text.setAutoDraw(True)
            
            # if form_1back_break_bottom_text is active this frame...
            if form_1back_break_bottom_text.status == STARTED:
                # update params
                pass
            
            # *form_1back_break_key_resp* updates
            waitOnFlip = False
            
            # if form_1back_break_key_resp is starting this frame...
            if form_1back_break_key_resp.status == NOT_STARTED and tThisFlip >= 60-frameTolerance:
                # keep track of start time/frame for later
                form_1back_break_key_resp.frameNStart = frameN  # exact frame index
                form_1back_break_key_resp.tStart = t  # local t and not account for scr refresh
                form_1back_break_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(form_1back_break_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'form_1back_break_key_resp.started')
                # update status
                form_1back_break_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(form_1back_break_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(form_1back_break_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if form_1back_break_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = form_1back_break_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _form_1back_break_key_resp_allKeys.extend(theseKeys)
                if len(_form_1back_break_key_resp_allKeys):
                    form_1back_break_key_resp.keys = _form_1back_break_key_resp_allKeys[-1].name  # just the last key pressed
                    form_1back_break_key_resp.rt = _form_1back_break_key_resp_allKeys[-1].rt
                    form_1back_break_key_resp.duration = _form_1back_break_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from form_1back_break_code
            # 控制文本只显示60秒
            if core.getTime() - text_start_time <= 60.0:  
                progress_text.draw()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                form_1back_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in form_1back_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "form_1back_break" ---
        for thisComponent in form_1back_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for form_1back_break
        form_1back_break.tStop = globalClock.getTime(format='float')
        form_1back_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('form_1back_break.stopped', form_1back_break.tStop)
        # check responses
        if form_1back_break_key_resp.keys in ['', [], None]:  # No response was made
            form_1back_break_key_resp.keys = None
        form_1back_outer.addData('form_1back_break_key_resp.keys',form_1back_break_key_resp.keys)
        if form_1back_break_key_resp.keys != None:  # we had a response
            form_1back_outer.addData('form_1back_break_key_resp.rt', form_1back_break_key_resp.rt)
            form_1back_outer.addData('form_1back_break_key_resp.duration', form_1back_break_key_resp.duration)
        # the Routine "form_1back_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 3.0 repeats of 'form_1back_outer'
    
    
    # --- Prepare to start Routine "xab_instr1" ---
    # create an object to store info about Routine xab_instr1
    xab_instr1 = data.Routine(
        name='xab_instr1',
        components=[xab_instr1_top_text, xab_instr1_X_image, xab_instr1_bottom_text, xab_instr1_key_resp],
    )
    xab_instr1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for xab_instr1_key_resp
    xab_instr1_key_resp.keys = []
    xab_instr1_key_resp.rt = []
    _xab_instr1_key_resp_allKeys = []
    # store start times for xab_instr1
    xab_instr1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    xab_instr1.tStart = globalClock.getTime(format='float')
    xab_instr1.status = STARTED
    thisExp.addData('xab_instr1.started', xab_instr1.tStart)
    xab_instr1.maxDuration = None
    # keep track of which components have finished
    xab_instr1Components = xab_instr1.components
    for thisComponent in xab_instr1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "xab_instr1" ---
    xab_instr1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *xab_instr1_top_text* updates
        
        # if xab_instr1_top_text is starting this frame...
        if xab_instr1_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr1_top_text.frameNStart = frameN  # exact frame index
            xab_instr1_top_text.tStart = t  # local t and not account for scr refresh
            xab_instr1_top_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr1_top_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr1_top_text.started')
            # update status
            xab_instr1_top_text.status = STARTED
            xab_instr1_top_text.setAutoDraw(True)
        
        # if xab_instr1_top_text is active this frame...
        if xab_instr1_top_text.status == STARTED:
            # update params
            pass
        
        # *xab_instr1_X_image* updates
        
        # if xab_instr1_X_image is starting this frame...
        if xab_instr1_X_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr1_X_image.frameNStart = frameN  # exact frame index
            xab_instr1_X_image.tStart = t  # local t and not account for scr refresh
            xab_instr1_X_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr1_X_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr1_X_image.started')
            # update status
            xab_instr1_X_image.status = STARTED
            xab_instr1_X_image.setAutoDraw(True)
        
        # if xab_instr1_X_image is active this frame...
        if xab_instr1_X_image.status == STARTED:
            # update params
            pass
        
        # *xab_instr1_bottom_text* updates
        
        # if xab_instr1_bottom_text is starting this frame...
        if xab_instr1_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr1_bottom_text.frameNStart = frameN  # exact frame index
            xab_instr1_bottom_text.tStart = t  # local t and not account for scr refresh
            xab_instr1_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr1_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr1_bottom_text.started')
            # update status
            xab_instr1_bottom_text.status = STARTED
            xab_instr1_bottom_text.setAutoDraw(True)
        
        # if xab_instr1_bottom_text is active this frame...
        if xab_instr1_bottom_text.status == STARTED:
            # update params
            pass
        
        # *xab_instr1_key_resp* updates
        waitOnFlip = False
        
        # if xab_instr1_key_resp is starting this frame...
        if xab_instr1_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr1_key_resp.frameNStart = frameN  # exact frame index
            xab_instr1_key_resp.tStart = t  # local t and not account for scr refresh
            xab_instr1_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr1_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr1_key_resp.started')
            # update status
            xab_instr1_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(xab_instr1_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(xab_instr1_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if xab_instr1_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = xab_instr1_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _xab_instr1_key_resp_allKeys.extend(theseKeys)
            if len(_xab_instr1_key_resp_allKeys):
                xab_instr1_key_resp.keys = _xab_instr1_key_resp_allKeys[-1].name  # just the last key pressed
                xab_instr1_key_resp.rt = _xab_instr1_key_resp_allKeys[-1].rt
                xab_instr1_key_resp.duration = _xab_instr1_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            xab_instr1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in xab_instr1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "xab_instr1" ---
    for thisComponent in xab_instr1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for xab_instr1
    xab_instr1.tStop = globalClock.getTime(format='float')
    xab_instr1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('xab_instr1.stopped', xab_instr1.tStop)
    # check responses
    if xab_instr1_key_resp.keys in ['', [], None]:  # No response was made
        xab_instr1_key_resp.keys = None
    thisExp.addData('xab_instr1_key_resp.keys',xab_instr1_key_resp.keys)
    if xab_instr1_key_resp.keys != None:  # we had a response
        thisExp.addData('xab_instr1_key_resp.rt', xab_instr1_key_resp.rt)
        thisExp.addData('xab_instr1_key_resp.duration', xab_instr1_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "xab_instr1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "xab_instr2" ---
    # create an object to store info about Routine xab_instr2
    xab_instr2 = data.Routine(
        name='xab_instr2',
        components=[xab_instr2_top_text, xab_instr2_bottom_text, xab_instr2_face_a_image, xab_instr2_face_b_image, xab_instr2_key_resp],
    )
    xab_instr2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for xab_instr2_key_resp
    xab_instr2_key_resp.keys = []
    xab_instr2_key_resp.rt = []
    _xab_instr2_key_resp_allKeys = []
    # store start times for xab_instr2
    xab_instr2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    xab_instr2.tStart = globalClock.getTime(format='float')
    xab_instr2.status = STARTED
    thisExp.addData('xab_instr2.started', xab_instr2.tStart)
    xab_instr2.maxDuration = None
    # keep track of which components have finished
    xab_instr2Components = xab_instr2.components
    for thisComponent in xab_instr2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "xab_instr2" ---
    xab_instr2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *xab_instr2_top_text* updates
        
        # if xab_instr2_top_text is starting this frame...
        if xab_instr2_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr2_top_text.frameNStart = frameN  # exact frame index
            xab_instr2_top_text.tStart = t  # local t and not account for scr refresh
            xab_instr2_top_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr2_top_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr2_top_text.started')
            # update status
            xab_instr2_top_text.status = STARTED
            xab_instr2_top_text.setAutoDraw(True)
        
        # if xab_instr2_top_text is active this frame...
        if xab_instr2_top_text.status == STARTED:
            # update params
            pass
        
        # *xab_instr2_bottom_text* updates
        
        # if xab_instr2_bottom_text is starting this frame...
        if xab_instr2_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr2_bottom_text.frameNStart = frameN  # exact frame index
            xab_instr2_bottom_text.tStart = t  # local t and not account for scr refresh
            xab_instr2_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr2_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr2_bottom_text.started')
            # update status
            xab_instr2_bottom_text.status = STARTED
            xab_instr2_bottom_text.setAutoDraw(True)
        
        # if xab_instr2_bottom_text is active this frame...
        if xab_instr2_bottom_text.status == STARTED:
            # update params
            pass
        
        # *xab_instr2_face_a_image* updates
        
        # if xab_instr2_face_a_image is starting this frame...
        if xab_instr2_face_a_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr2_face_a_image.frameNStart = frameN  # exact frame index
            xab_instr2_face_a_image.tStart = t  # local t and not account for scr refresh
            xab_instr2_face_a_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr2_face_a_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr2_face_a_image.started')
            # update status
            xab_instr2_face_a_image.status = STARTED
            xab_instr2_face_a_image.setAutoDraw(True)
        
        # if xab_instr2_face_a_image is active this frame...
        if xab_instr2_face_a_image.status == STARTED:
            # update params
            pass
        
        # *xab_instr2_face_b_image* updates
        
        # if xab_instr2_face_b_image is starting this frame...
        if xab_instr2_face_b_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr2_face_b_image.frameNStart = frameN  # exact frame index
            xab_instr2_face_b_image.tStart = t  # local t and not account for scr refresh
            xab_instr2_face_b_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr2_face_b_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr2_face_b_image.started')
            # update status
            xab_instr2_face_b_image.status = STARTED
            xab_instr2_face_b_image.setAutoDraw(True)
        
        # if xab_instr2_face_b_image is active this frame...
        if xab_instr2_face_b_image.status == STARTED:
            # update params
            pass
        
        # *xab_instr2_key_resp* updates
        waitOnFlip = False
        
        # if xab_instr2_key_resp is starting this frame...
        if xab_instr2_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            xab_instr2_key_resp.frameNStart = frameN  # exact frame index
            xab_instr2_key_resp.tStart = t  # local t and not account for scr refresh
            xab_instr2_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(xab_instr2_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'xab_instr2_key_resp.started')
            # update status
            xab_instr2_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(xab_instr2_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(xab_instr2_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if xab_instr2_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = xab_instr2_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _xab_instr2_key_resp_allKeys.extend(theseKeys)
            if len(_xab_instr2_key_resp_allKeys):
                xab_instr2_key_resp.keys = _xab_instr2_key_resp_allKeys[-1].name  # just the last key pressed
                xab_instr2_key_resp.rt = _xab_instr2_key_resp_allKeys[-1].rt
                xab_instr2_key_resp.duration = _xab_instr2_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            xab_instr2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in xab_instr2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "xab_instr2" ---
    for thisComponent in xab_instr2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for xab_instr2
    xab_instr2.tStop = globalClock.getTime(format='float')
    xab_instr2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('xab_instr2.stopped', xab_instr2.tStop)
    # check responses
    if xab_instr2_key_resp.keys in ['', [], None]:  # No response was made
        xab_instr2_key_resp.keys = None
    thisExp.addData('xab_instr2_key_resp.keys',xab_instr2_key_resp.keys)
    if xab_instr2_key_resp.keys != None:  # we had a response
        thisExp.addData('xab_instr2_key_resp.rt', xab_instr2_key_resp.rt)
        thisExp.addData('xab_instr2_key_resp.duration', xab_instr2_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "xab_instr2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    continue_prac_xab = data.TrialHandler2(
        name='continue_prac_xab',
        nReps=999.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(continue_prac_xab)  # add the loop to the experiment
    thisContinue_prac_xab = continue_prac_xab.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisContinue_prac_xab.rgb)
    if thisContinue_prac_xab != None:
        for paramName in thisContinue_prac_xab:
            globals()[paramName] = thisContinue_prac_xab[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisContinue_prac_xab in continue_prac_xab:
        currentLoop = continue_prac_xab
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisContinue_prac_xab.rgb)
        if thisContinue_prac_xab != None:
            for paramName in thisContinue_prac_xab:
                globals()[paramName] = thisContinue_prac_xab[paramName]
        
        # --- Prepare to start Routine "xab_instr3" ---
        # create an object to store info about Routine xab_instr3
        xab_instr3 = data.Routine(
            name='xab_instr3',
            components=[xab_instr3_center_text, xab_instr3_bottom_text, xab_instr3_key_resp],
        )
        xab_instr3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for xab_instr3_key_resp
        xab_instr3_key_resp.keys = []
        xab_instr3_key_resp.rt = []
        _xab_instr3_key_resp_allKeys = []
        # store start times for xab_instr3
        xab_instr3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        xab_instr3.tStart = globalClock.getTime(format='float')
        xab_instr3.status = STARTED
        thisExp.addData('xab_instr3.started', xab_instr3.tStart)
        xab_instr3.maxDuration = None
        # keep track of which components have finished
        xab_instr3Components = xab_instr3.components
        for thisComponent in xab_instr3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "xab_instr3" ---
        # if trial has changed, end Routine now
        if isinstance(continue_prac_xab, data.TrialHandler2) and thisContinue_prac_xab.thisN != continue_prac_xab.thisTrial.thisN:
            continueRoutine = False
        xab_instr3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *xab_instr3_center_text* updates
            
            # if xab_instr3_center_text is starting this frame...
            if xab_instr3_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                xab_instr3_center_text.frameNStart = frameN  # exact frame index
                xab_instr3_center_text.tStart = t  # local t and not account for scr refresh
                xab_instr3_center_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(xab_instr3_center_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xab_instr3_center_text.started')
                # update status
                xab_instr3_center_text.status = STARTED
                xab_instr3_center_text.setAutoDraw(True)
            
            # if xab_instr3_center_text is active this frame...
            if xab_instr3_center_text.status == STARTED:
                # update params
                pass
            
            # *xab_instr3_bottom_text* updates
            
            # if xab_instr3_bottom_text is starting this frame...
            if xab_instr3_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                xab_instr3_bottom_text.frameNStart = frameN  # exact frame index
                xab_instr3_bottom_text.tStart = t  # local t and not account for scr refresh
                xab_instr3_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(xab_instr3_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xab_instr3_bottom_text.started')
                # update status
                xab_instr3_bottom_text.status = STARTED
                xab_instr3_bottom_text.setAutoDraw(True)
            
            # if xab_instr3_bottom_text is active this frame...
            if xab_instr3_bottom_text.status == STARTED:
                # update params
                pass
            
            # *xab_instr3_key_resp* updates
            waitOnFlip = False
            
            # if xab_instr3_key_resp is starting this frame...
            if xab_instr3_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                xab_instr3_key_resp.frameNStart = frameN  # exact frame index
                xab_instr3_key_resp.tStart = t  # local t and not account for scr refresh
                xab_instr3_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(xab_instr3_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xab_instr3_key_resp.started')
                # update status
                xab_instr3_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(xab_instr3_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(xab_instr3_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if xab_instr3_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = xab_instr3_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _xab_instr3_key_resp_allKeys.extend(theseKeys)
                if len(_xab_instr3_key_resp_allKeys):
                    xab_instr3_key_resp.keys = _xab_instr3_key_resp_allKeys[-1].name  # just the last key pressed
                    xab_instr3_key_resp.rt = _xab_instr3_key_resp_allKeys[-1].rt
                    xab_instr3_key_resp.duration = _xab_instr3_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                xab_instr3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in xab_instr3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "xab_instr3" ---
        for thisComponent in xab_instr3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for xab_instr3
        xab_instr3.tStop = globalClock.getTime(format='float')
        xab_instr3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('xab_instr3.stopped', xab_instr3.tStop)
        # check responses
        if xab_instr3_key_resp.keys in ['', [], None]:  # No response was made
            xab_instr3_key_resp.keys = None
        continue_prac_xab.addData('xab_instr3_key_resp.keys',xab_instr3_key_resp.keys)
        if xab_instr3_key_resp.keys != None:  # we had a response
            continue_prac_xab.addData('xab_instr3_key_resp.rt', xab_instr3_key_resp.rt)
            continue_prac_xab.addData('xab_instr3_key_resp.duration', xab_instr3_key_resp.duration)
        # the Routine "xab_instr3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        prac_xab = data.TrialHandler2(
            name='prac_xab',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('prac_xab_faces.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(prac_xab)  # add the loop to the experiment
        thisPrac_xab = prac_xab.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_xab.rgb)
        if thisPrac_xab != None:
            for paramName in thisPrac_xab:
                globals()[paramName] = thisPrac_xab[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPrac_xab in prac_xab:
            currentLoop = prac_xab
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPrac_xab.rgb)
            if thisPrac_xab != None:
                for paramName in thisPrac_xab:
                    globals()[paramName] = thisPrac_xab[paramName]
            
            # --- Prepare to start Routine "prac_xab_x" ---
            # create an object to store info about Routine prac_xab_x
            prac_xab_x = data.Routine(
                name='prac_xab_x',
                components=[prac_xab_x_cross_text, prac_xab_x_image],
            )
            prac_xab_x.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac_xab_x_image.setImage(prac_x)
            # store start times for prac_xab_x
            prac_xab_x.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_xab_x.tStart = globalClock.getTime(format='float')
            prac_xab_x.status = STARTED
            thisExp.addData('prac_xab_x.started', prac_xab_x.tStart)
            prac_xab_x.maxDuration = None
            # keep track of which components have finished
            prac_xab_xComponents = prac_xab_x.components
            for thisComponent in prac_xab_x.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_xab_x" ---
            # if trial has changed, end Routine now
            if isinstance(prac_xab, data.TrialHandler2) and thisPrac_xab.thisN != prac_xab.thisTrial.thisN:
                continueRoutine = False
            prac_xab_x.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.25:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_xab_x_cross_text* updates
                
                # if prac_xab_x_cross_text is starting this frame...
                if prac_xab_x_cross_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_xab_x_cross_text.frameNStart = frameN  # exact frame index
                    prac_xab_x_cross_text.tStart = t  # local t and not account for scr refresh
                    prac_xab_x_cross_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_xab_x_cross_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_xab_x_cross_text.started')
                    # update status
                    prac_xab_x_cross_text.status = STARTED
                    prac_xab_x_cross_text.setAutoDraw(True)
                
                # if prac_xab_x_cross_text is active this frame...
                if prac_xab_x_cross_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_xab_x_cross_text is stopping this frame...
                if prac_xab_x_cross_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_xab_x_cross_text.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_xab_x_cross_text.tStop = t  # not accounting for scr refresh
                        prac_xab_x_cross_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_xab_x_cross_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_xab_x_cross_text.stopped')
                        # update status
                        prac_xab_x_cross_text.status = FINISHED
                        prac_xab_x_cross_text.setAutoDraw(False)
                
                # *prac_xab_x_image* updates
                
                # if prac_xab_x_image is starting this frame...
                if prac_xab_x_image.status == NOT_STARTED and tThisFlip >= 0.25-frameTolerance:
                    # keep track of start time/frame for later
                    prac_xab_x_image.frameNStart = frameN  # exact frame index
                    prac_xab_x_image.tStart = t  # local t and not account for scr refresh
                    prac_xab_x_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_xab_x_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_xab_x_image.started')
                    # update status
                    prac_xab_x_image.status = STARTED
                    prac_xab_x_image.setAutoDraw(True)
                
                # if prac_xab_x_image is active this frame...
                if prac_xab_x_image.status == STARTED:
                    # update params
                    pass
                
                # if prac_xab_x_image is stopping this frame...
                if prac_xab_x_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_xab_x_image.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_xab_x_image.tStop = t  # not accounting for scr refresh
                        prac_xab_x_image.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_xab_x_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_xab_x_image.stopped')
                        # update status
                        prac_xab_x_image.status = FINISHED
                        prac_xab_x_image.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_xab_x.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_xab_x.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_xab_x" ---
            for thisComponent in prac_xab_x.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_xab_x
            prac_xab_x.tStop = globalClock.getTime(format='float')
            prac_xab_x.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_xab_x.stopped', prac_xab_x.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_xab_x.maxDurationReached:
                routineTimer.addTime(-prac_xab_x.maxDuration)
            elif prac_xab_x.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.250000)
            
            # --- Prepare to start Routine "prac_xab_blank1" ---
            # create an object to store info about Routine prac_xab_blank1
            prac_xab_blank1 = data.Routine(
                name='prac_xab_blank1',
                components=[prac_xab_blank1_text],
            )
            prac_xab_blank1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for prac_xab_blank1
            prac_xab_blank1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_xab_blank1.tStart = globalClock.getTime(format='float')
            prac_xab_blank1.status = STARTED
            thisExp.addData('prac_xab_blank1.started', prac_xab_blank1.tStart)
            prac_xab_blank1.maxDuration = None
            # keep track of which components have finished
            prac_xab_blank1Components = prac_xab_blank1.components
            for thisComponent in prac_xab_blank1.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_xab_blank1" ---
            # if trial has changed, end Routine now
            if isinstance(prac_xab, data.TrialHandler2) and thisPrac_xab.thisN != prac_xab.thisTrial.thisN:
                continueRoutine = False
            prac_xab_blank1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_xab_blank1_text* updates
                
                # if prac_xab_blank1_text is starting this frame...
                if prac_xab_blank1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_xab_blank1_text.frameNStart = frameN  # exact frame index
                    prac_xab_blank1_text.tStart = t  # local t and not account for scr refresh
                    prac_xab_blank1_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_xab_blank1_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_xab_blank1_text.started')
                    # update status
                    prac_xab_blank1_text.status = STARTED
                    prac_xab_blank1_text.setAutoDraw(True)
                
                # if prac_xab_blank1_text is active this frame...
                if prac_xab_blank1_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_xab_blank1_text is stopping this frame...
                if prac_xab_blank1_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_xab_blank1_text.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_xab_blank1_text.tStop = t  # not accounting for scr refresh
                        prac_xab_blank1_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_xab_blank1_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_xab_blank1_text.stopped')
                        # update status
                        prac_xab_blank1_text.status = FINISHED
                        prac_xab_blank1_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_xab_blank1.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_xab_blank1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_xab_blank1" ---
            for thisComponent in prac_xab_blank1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_xab_blank1
            prac_xab_blank1.tStop = globalClock.getTime(format='float')
            prac_xab_blank1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_xab_blank1.stopped', prac_xab_blank1.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_xab_blank1.maxDurationReached:
                routineTimer.addTime(-prac_xab_blank1.maxDuration)
            elif prac_xab_blank1.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "prac_xab_ab" ---
            # create an object to store info about Routine prac_xab_ab
            prac_xab_ab = data.Routine(
                name='prac_xab_ab',
                components=[prac_xab_ab_a_image, prac_xab_ab_b_image, prac_xab_ab_key_resp],
            )
            prac_xab_ab.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac_xab_ab_a_image.setImage(prac_a)
            prac_xab_ab_b_image.setImage(prac_b)
            # create starting attributes for prac_xab_ab_key_resp
            prac_xab_ab_key_resp.keys = []
            prac_xab_ab_key_resp.rt = []
            _prac_xab_ab_key_resp_allKeys = []
            # Run 'Begin Routine' code from prac_xab_ab_code
            # 防止被试在刺激呈现之前按键反应
            # 完全重置keyboard组件
            prac_xab_ab_key_resp.status = NOT_STARTED
            prac_xab_ab_key_resp.keys = []
            prac_xab_ab_key_resp.rt = []
            prac_xab_ab_key_resp.clock.reset()
            prac_xab_ab_key_resp.clearEvents()
            event.clearEvents(eventType='keyboard')
            
            # 记录routine开始时间（用于计算反应时）
            prac_start_time = core.getTime()
            
            # 试次计数+1
            prac_xab_trial += 1
            
            # 初始化反应相关变量
            prac_xab_resp = None  # 反应正确性（1=正确，0=错误）
            prac_xab_rt = None    # 反应时
            prac_user_key = None  # 被试按下的键
            
            # store start times for prac_xab_ab
            prac_xab_ab.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_xab_ab.tStart = globalClock.getTime(format='float')
            prac_xab_ab.status = STARTED
            thisExp.addData('prac_xab_ab.started', prac_xab_ab.tStart)
            prac_xab_ab.maxDuration = None
            # keep track of which components have finished
            prac_xab_abComponents = prac_xab_ab.components
            for thisComponent in prac_xab_ab.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_xab_ab" ---
            # if trial has changed, end Routine now
            if isinstance(prac_xab, data.TrialHandler2) and thisPrac_xab.thisN != prac_xab.thisTrial.thisN:
                continueRoutine = False
            prac_xab_ab.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_xab_ab_a_image* updates
                
                # if prac_xab_ab_a_image is starting this frame...
                if prac_xab_ab_a_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_xab_ab_a_image.frameNStart = frameN  # exact frame index
                    prac_xab_ab_a_image.tStart = t  # local t and not account for scr refresh
                    prac_xab_ab_a_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_xab_ab_a_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_xab_ab_a_image.started')
                    # update status
                    prac_xab_ab_a_image.status = STARTED
                    prac_xab_ab_a_image.setAutoDraw(True)
                
                # if prac_xab_ab_a_image is active this frame...
                if prac_xab_ab_a_image.status == STARTED:
                    # update params
                    pass
                
                # *prac_xab_ab_b_image* updates
                
                # if prac_xab_ab_b_image is starting this frame...
                if prac_xab_ab_b_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_xab_ab_b_image.frameNStart = frameN  # exact frame index
                    prac_xab_ab_b_image.tStart = t  # local t and not account for scr refresh
                    prac_xab_ab_b_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_xab_ab_b_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_xab_ab_b_image.started')
                    # update status
                    prac_xab_ab_b_image.status = STARTED
                    prac_xab_ab_b_image.setAutoDraw(True)
                
                # if prac_xab_ab_b_image is active this frame...
                if prac_xab_ab_b_image.status == STARTED:
                    # update params
                    pass
                
                # *prac_xab_ab_key_resp* updates
                waitOnFlip = False
                
                # if prac_xab_ab_key_resp is starting this frame...
                if prac_xab_ab_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_xab_ab_key_resp.frameNStart = frameN  # exact frame index
                    prac_xab_ab_key_resp.tStart = t  # local t and not account for scr refresh
                    prac_xab_ab_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_xab_ab_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_xab_ab_key_resp.started')
                    # update status
                    prac_xab_ab_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_xab_ab_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_xab_ab_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac_xab_ab_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = prac_xab_ab_key_resp.getKeys(keyList=['f','j'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_xab_ab_key_resp_allKeys.extend(theseKeys)
                    if len(_prac_xab_ab_key_resp_allKeys):
                        prac_xab_ab_key_resp.keys = _prac_xab_ab_key_resp_allKeys[-1].name  # just the last key pressed
                        prac_xab_ab_key_resp.rt = _prac_xab_ab_key_resp_allKeys[-1].rt
                        prac_xab_ab_key_resp.duration = _prac_xab_ab_key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from prac_xab_ab_code
                # 强制控制routine不能自动结束
                continueRoutine = True  # 每帧都重新设置为True
                
                # 按键检测代码
                keys = event.getKeys(keyList=['f', 'j'])
                if keys and prac_user_key is None:  # 只记录第一次按键
                    prac_user_key = keys[0]  # 记录按下的键
                    prac_xab_rt = round(core.getTime() - prac_start_time, 3)  # 计算反应时
                    
                    # 判断反应是否正确
                    if prac_user_key == prac_correct_resp:  # 与条件文件中的正确答案比较
                        prac_xab_resp = 1  # 正确
                    else:
                        prac_xab_resp = 0  # 错误
                    
                    # 只有在按键后才允许routine结束
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_xab_ab.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_xab_ab.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_xab_ab" ---
            for thisComponent in prac_xab_ab.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_xab_ab
            prac_xab_ab.tStop = globalClock.getTime(format='float')
            prac_xab_ab.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_xab_ab.stopped', prac_xab_ab.tStop)
            # check responses
            if prac_xab_ab_key_resp.keys in ['', [], None]:  # No response was made
                prac_xab_ab_key_resp.keys = None
            prac_xab.addData('prac_xab_ab_key_resp.keys',prac_xab_ab_key_resp.keys)
            if prac_xab_ab_key_resp.keys != None:  # we had a response
                prac_xab.addData('prac_xab_ab_key_resp.rt', prac_xab_ab_key_resp.rt)
                prac_xab.addData('prac_xab_ab_key_resp.duration', prac_xab_ab_key_resp.duration)
            # Run 'End Routine' code from prac_xab_ab_code
            # 保存数据到实验文件
            thisExp.addData('prac_xab_block', prac_xab_block)
            thisExp.addData('prac_xab_trial', prac_xab_trial)
            thisExp.addData('prac_xab_resp', prac_xab_resp)
            thisExp.addData('prac_xab_rt', prac_xab_rt)
            thisExp.addData('prac_user_key', prac_user_key)  # 额外保存被试按的键，便于分析
            thisExp.nextEntry()
            # the Routine "prac_xab_ab" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac_xab_blank2" ---
            # create an object to store info about Routine prac_xab_blank2
            prac_xab_blank2 = data.Routine(
                name='prac_xab_blank2',
                components=[pracl_xab_blank2_text],
            )
            prac_xab_blank2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for prac_xab_blank2
            prac_xab_blank2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_xab_blank2.tStart = globalClock.getTime(format='float')
            prac_xab_blank2.status = STARTED
            thisExp.addData('prac_xab_blank2.started', prac_xab_blank2.tStart)
            prac_xab_blank2.maxDuration = None
            # keep track of which components have finished
            prac_xab_blank2Components = prac_xab_blank2.components
            for thisComponent in prac_xab_blank2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_xab_blank2" ---
            # if trial has changed, end Routine now
            if isinstance(prac_xab, data.TrialHandler2) and thisPrac_xab.thisN != prac_xab.thisTrial.thisN:
                continueRoutine = False
            prac_xab_blank2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pracl_xab_blank2_text* updates
                
                # if pracl_xab_blank2_text is starting this frame...
                if pracl_xab_blank2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pracl_xab_blank2_text.frameNStart = frameN  # exact frame index
                    pracl_xab_blank2_text.tStart = t  # local t and not account for scr refresh
                    pracl_xab_blank2_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracl_xab_blank2_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracl_xab_blank2_text.started')
                    # update status
                    pracl_xab_blank2_text.status = STARTED
                    pracl_xab_blank2_text.setAutoDraw(True)
                
                # if pracl_xab_blank2_text is active this frame...
                if pracl_xab_blank2_text.status == STARTED:
                    # update params
                    pass
                
                # if pracl_xab_blank2_text is stopping this frame...
                if pracl_xab_blank2_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracl_xab_blank2_text.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        pracl_xab_blank2_text.tStop = t  # not accounting for scr refresh
                        pracl_xab_blank2_text.tStopRefresh = tThisFlipGlobal  # on global time
                        pracl_xab_blank2_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracl_xab_blank2_text.stopped')
                        # update status
                        pracl_xab_blank2_text.status = FINISHED
                        pracl_xab_blank2_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_xab_blank2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_xab_blank2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_xab_blank2" ---
            for thisComponent in prac_xab_blank2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_xab_blank2
            prac_xab_blank2.tStop = globalClock.getTime(format='float')
            prac_xab_blank2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_xab_blank2.stopped', prac_xab_blank2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_xab_blank2.maxDurationReached:
                routineTimer.addTime(-prac_xab_blank2.maxDuration)
            elif prac_xab_blank2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'prac_xab'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "prac_xab_end" ---
        # create an object to store info about Routine prac_xab_end
        prac_xab_end = data.Routine(
            name='prac_xab_end',
            components=[prac_xab_end_center_text, prac_xab_end_bottom_text, prac_xab_end_key_resp],
        )
        prac_xab_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for prac_xab_end_key_resp
        prac_xab_end_key_resp.keys = []
        prac_xab_end_key_resp.rt = []
        _prac_xab_end_key_resp_allKeys = []
        # Run 'Begin Routine' code from prac_xan_end_code
        ###########
        #以下代码是为了按“p”键能够返回练习
        # 清空之前的按键缓存，准备接收新按键
        event.clearEvents()
        
        ##########
        # 如果按了P键重新开始练习，block计数器+1
        prac_xab_block += 1
        
        #########
        # 控制文字的左对齐
        prac_xab_end_center_text.alignText = 'left'
        
        
        # store start times for prac_xab_end
        prac_xab_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_xab_end.tStart = globalClock.getTime(format='float')
        prac_xab_end.status = STARTED
        thisExp.addData('prac_xab_end.started', prac_xab_end.tStart)
        prac_xab_end.maxDuration = None
        # keep track of which components have finished
        prac_xab_endComponents = prac_xab_end.components
        for thisComponent in prac_xab_end.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_xab_end" ---
        # if trial has changed, end Routine now
        if isinstance(continue_prac_xab, data.TrialHandler2) and thisContinue_prac_xab.thisN != continue_prac_xab.thisTrial.thisN:
            continueRoutine = False
        prac_xab_end.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_xab_end_center_text* updates
            
            # if prac_xab_end_center_text is starting this frame...
            if prac_xab_end_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_xab_end_center_text.frameNStart = frameN  # exact frame index
                prac_xab_end_center_text.tStart = t  # local t and not account for scr refresh
                prac_xab_end_center_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_xab_end_center_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_xab_end_center_text.started')
                # update status
                prac_xab_end_center_text.status = STARTED
                prac_xab_end_center_text.setAutoDraw(True)
            
            # if prac_xab_end_center_text is active this frame...
            if prac_xab_end_center_text.status == STARTED:
                # update params
                pass
            
            # *prac_xab_end_bottom_text* updates
            
            # if prac_xab_end_bottom_text is starting this frame...
            if prac_xab_end_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_xab_end_bottom_text.frameNStart = frameN  # exact frame index
                prac_xab_end_bottom_text.tStart = t  # local t and not account for scr refresh
                prac_xab_end_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_xab_end_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_xab_end_bottom_text.started')
                # update status
                prac_xab_end_bottom_text.status = STARTED
                prac_xab_end_bottom_text.setAutoDraw(True)
            
            # if prac_xab_end_bottom_text is active this frame...
            if prac_xab_end_bottom_text.status == STARTED:
                # update params
                pass
            
            # *prac_xab_end_key_resp* updates
            waitOnFlip = False
            
            # if prac_xab_end_key_resp is starting this frame...
            if prac_xab_end_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_xab_end_key_resp.frameNStart = frameN  # exact frame index
                prac_xab_end_key_resp.tStart = t  # local t and not account for scr refresh
                prac_xab_end_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_xab_end_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_xab_end_key_resp.started')
                # update status
                prac_xab_end_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_xab_end_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_xab_end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_xab_end_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_xab_end_key_resp.getKeys(keyList=['space','p'], ignoreKeys=["escape"], waitRelease=False)
                _prac_xab_end_key_resp_allKeys.extend(theseKeys)
                if len(_prac_xab_end_key_resp_allKeys):
                    prac_xab_end_key_resp.keys = _prac_xab_end_key_resp_allKeys[-1].name  # just the last key pressed
                    prac_xab_end_key_resp.rt = _prac_xab_end_key_resp_allKeys[-1].rt
                    prac_xab_end_key_resp.duration = _prac_xab_end_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_xab_end.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_xab_end.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_xab_end" ---
        for thisComponent in prac_xab_end.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_xab_end
        prac_xab_end.tStop = globalClock.getTime(format='float')
        prac_xab_end.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_xab_end.stopped', prac_xab_end.tStop)
        # check responses
        if prac_xab_end_key_resp.keys in ['', [], None]:  # No response was made
            prac_xab_end_key_resp.keys = None
        continue_prac_xab.addData('prac_xab_end_key_resp.keys',prac_xab_end_key_resp.keys)
        if prac_xab_end_key_resp.keys != None:  # we had a response
            continue_prac_xab.addData('prac_xab_end_key_resp.rt', prac_xab_end_key_resp.rt)
            continue_prac_xab.addData('prac_xab_end_key_resp.duration', prac_xab_end_key_resp.duration)
        # Run 'End Routine' code from prac_xan_end_code
        # 等待任意按键
        keys = event.waitKeys()  
        
        if 'p' in keys:
            # 继续练习，什么都不做
            pass
        else:
            # 结束练习
            continue_prac_xab.finished = True
        
        # the Routine "prac_xab_end" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 999.0 repeats of 'continue_prac_xab'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    form_xab_block = data.TrialHandler2(
        name='form_xab_block',
        nReps=4.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(form_xab_block)  # add the loop to the experiment
    thisForm_xab_block = form_xab_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisForm_xab_block.rgb)
    if thisForm_xab_block != None:
        for paramName in thisForm_xab_block:
            globals()[paramName] = thisForm_xab_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisForm_xab_block in form_xab_block:
        currentLoop = form_xab_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisForm_xab_block.rgb)
        if thisForm_xab_block != None:
            for paramName in thisForm_xab_block:
                globals()[paramName] = thisForm_xab_block[paramName]
        
        # set up handler to look after randomisation of conditions etc
        form_xab = data.TrialHandler2(
            name='form_xab',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('formal_xab_faces.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(form_xab)  # add the loop to the experiment
        thisForm_xab = form_xab.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisForm_xab.rgb)
        if thisForm_xab != None:
            for paramName in thisForm_xab:
                globals()[paramName] = thisForm_xab[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisForm_xab in form_xab:
            currentLoop = form_xab
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisForm_xab.rgb)
            if thisForm_xab != None:
                for paramName in thisForm_xab:
                    globals()[paramName] = thisForm_xab[paramName]
            
            # --- Prepare to start Routine "xab_x" ---
            # create an object to store info about Routine xab_x
            xab_x = data.Routine(
                name='xab_x',
                components=[xab_x_cross_text, xab_x_image],
            )
            xab_x.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            xab_x_image.setImage(formal_x)
            # store start times for xab_x
            xab_x.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            xab_x.tStart = globalClock.getTime(format='float')
            xab_x.status = STARTED
            thisExp.addData('xab_x.started', xab_x.tStart)
            xab_x.maxDuration = None
            # keep track of which components have finished
            xab_xComponents = xab_x.components
            for thisComponent in xab_x.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "xab_x" ---
            # if trial has changed, end Routine now
            if isinstance(form_xab, data.TrialHandler2) and thisForm_xab.thisN != form_xab.thisTrial.thisN:
                continueRoutine = False
            xab_x.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.25:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *xab_x_cross_text* updates
                
                # if xab_x_cross_text is starting this frame...
                if xab_x_cross_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    xab_x_cross_text.frameNStart = frameN  # exact frame index
                    xab_x_cross_text.tStart = t  # local t and not account for scr refresh
                    xab_x_cross_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(xab_x_cross_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'xab_x_cross_text.started')
                    # update status
                    xab_x_cross_text.status = STARTED
                    xab_x_cross_text.setAutoDraw(True)
                
                # if xab_x_cross_text is active this frame...
                if xab_x_cross_text.status == STARTED:
                    # update params
                    pass
                
                # if xab_x_cross_text is stopping this frame...
                if xab_x_cross_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > xab_x_cross_text.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        xab_x_cross_text.tStop = t  # not accounting for scr refresh
                        xab_x_cross_text.tStopRefresh = tThisFlipGlobal  # on global time
                        xab_x_cross_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'xab_x_cross_text.stopped')
                        # update status
                        xab_x_cross_text.status = FINISHED
                        xab_x_cross_text.setAutoDraw(False)
                
                # *xab_x_image* updates
                
                # if xab_x_image is starting this frame...
                if xab_x_image.status == NOT_STARTED and tThisFlip >= 0.25-frameTolerance:
                    # keep track of start time/frame for later
                    xab_x_image.frameNStart = frameN  # exact frame index
                    xab_x_image.tStart = t  # local t and not account for scr refresh
                    xab_x_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(xab_x_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'xab_x_image.started')
                    # update status
                    xab_x_image.status = STARTED
                    xab_x_image.setAutoDraw(True)
                
                # if xab_x_image is active this frame...
                if xab_x_image.status == STARTED:
                    # update params
                    pass
                
                # if xab_x_image is stopping this frame...
                if xab_x_image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > xab_x_image.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        xab_x_image.tStop = t  # not accounting for scr refresh
                        xab_x_image.tStopRefresh = tThisFlipGlobal  # on global time
                        xab_x_image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'xab_x_image.stopped')
                        # update status
                        xab_x_image.status = FINISHED
                        xab_x_image.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    xab_x.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in xab_x.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "xab_x" ---
            for thisComponent in xab_x.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for xab_x
            xab_x.tStop = globalClock.getTime(format='float')
            xab_x.tStopRefresh = tThisFlipGlobal
            thisExp.addData('xab_x.stopped', xab_x.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if xab_x.maxDurationReached:
                routineTimer.addTime(-xab_x.maxDuration)
            elif xab_x.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.250000)
            
            # --- Prepare to start Routine "xab_blank1" ---
            # create an object to store info about Routine xab_blank1
            xab_blank1 = data.Routine(
                name='xab_blank1',
                components=[xab_blank1_text],
            )
            xab_blank1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for xab_blank1
            xab_blank1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            xab_blank1.tStart = globalClock.getTime(format='float')
            xab_blank1.status = STARTED
            thisExp.addData('xab_blank1.started', xab_blank1.tStart)
            xab_blank1.maxDuration = None
            # keep track of which components have finished
            xab_blank1Components = xab_blank1.components
            for thisComponent in xab_blank1.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "xab_blank1" ---
            # if trial has changed, end Routine now
            if isinstance(form_xab, data.TrialHandler2) and thisForm_xab.thisN != form_xab.thisTrial.thisN:
                continueRoutine = False
            xab_blank1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *xab_blank1_text* updates
                
                # if xab_blank1_text is starting this frame...
                if xab_blank1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    xab_blank1_text.frameNStart = frameN  # exact frame index
                    xab_blank1_text.tStart = t  # local t and not account for scr refresh
                    xab_blank1_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(xab_blank1_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'xab_blank1_text.started')
                    # update status
                    xab_blank1_text.status = STARTED
                    xab_blank1_text.setAutoDraw(True)
                
                # if xab_blank1_text is active this frame...
                if xab_blank1_text.status == STARTED:
                    # update params
                    pass
                
                # if xab_blank1_text is stopping this frame...
                if xab_blank1_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > xab_blank1_text.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        xab_blank1_text.tStop = t  # not accounting for scr refresh
                        xab_blank1_text.tStopRefresh = tThisFlipGlobal  # on global time
                        xab_blank1_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'xab_blank1_text.stopped')
                        # update status
                        xab_blank1_text.status = FINISHED
                        xab_blank1_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    xab_blank1.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in xab_blank1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "xab_blank1" ---
            for thisComponent in xab_blank1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for xab_blank1
            xab_blank1.tStop = globalClock.getTime(format='float')
            xab_blank1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('xab_blank1.stopped', xab_blank1.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if xab_blank1.maxDurationReached:
                routineTimer.addTime(-xab_blank1.maxDuration)
            elif xab_blank1.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "xab_ab" ---
            # create an object to store info about Routine xab_ab
            xab_ab = data.Routine(
                name='xab_ab',
                components=[xab_ab_a_image, xab_ab_b_image, xab_ab_key_resp],
            )
            xab_ab.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            xab_ab_a_image.setImage(formal_a)
            xab_ab_b_image.setImage(formal_b)
            # create starting attributes for xab_ab_key_resp
            xab_ab_key_resp.keys = []
            xab_ab_key_resp.rt = []
            _xab_ab_key_resp_allKeys = []
            # Run 'Begin Routine' code from xab_ab_code
            # 防止被试在刺激呈现之前按键反应
            # 完全重置keyboard组件
            xab_ab_key_resp.status = NOT_STARTED
            xab_ab_key_resp.keys = []
            xab_ab_key_resp.rt = []
            xab_ab_key_resp.clock.reset()
            xab_ab_key_resp.clearEvents()
            event.clearEvents(eventType='keyboard')
            
            # 记录routine开始时间（用于计算反应时）
            form_start_time = core.getTime()
            
            # 试次计数+1
            formal_xab_trial += 1
            
            # 初始化反应相关变量
            form_xab_resp = None  # 反应正确性（1=正确，0=错误）
            form_xab_rt = None    # 反应时
            form_user_key = None       # 被试按下的键
            
            # store start times for xab_ab
            xab_ab.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            xab_ab.tStart = globalClock.getTime(format='float')
            xab_ab.status = STARTED
            thisExp.addData('xab_ab.started', xab_ab.tStart)
            xab_ab.maxDuration = None
            # keep track of which components have finished
            xab_abComponents = xab_ab.components
            for thisComponent in xab_ab.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "xab_ab" ---
            # if trial has changed, end Routine now
            if isinstance(form_xab, data.TrialHandler2) and thisForm_xab.thisN != form_xab.thisTrial.thisN:
                continueRoutine = False
            xab_ab.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *xab_ab_a_image* updates
                
                # if xab_ab_a_image is starting this frame...
                if xab_ab_a_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    xab_ab_a_image.frameNStart = frameN  # exact frame index
                    xab_ab_a_image.tStart = t  # local t and not account for scr refresh
                    xab_ab_a_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(xab_ab_a_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'xab_ab_a_image.started')
                    # update status
                    xab_ab_a_image.status = STARTED
                    xab_ab_a_image.setAutoDraw(True)
                
                # if xab_ab_a_image is active this frame...
                if xab_ab_a_image.status == STARTED:
                    # update params
                    pass
                
                # *xab_ab_b_image* updates
                
                # if xab_ab_b_image is starting this frame...
                if xab_ab_b_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    xab_ab_b_image.frameNStart = frameN  # exact frame index
                    xab_ab_b_image.tStart = t  # local t and not account for scr refresh
                    xab_ab_b_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(xab_ab_b_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'xab_ab_b_image.started')
                    # update status
                    xab_ab_b_image.status = STARTED
                    xab_ab_b_image.setAutoDraw(True)
                
                # if xab_ab_b_image is active this frame...
                if xab_ab_b_image.status == STARTED:
                    # update params
                    pass
                
                # *xab_ab_key_resp* updates
                waitOnFlip = False
                
                # if xab_ab_key_resp is starting this frame...
                if xab_ab_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    xab_ab_key_resp.frameNStart = frameN  # exact frame index
                    xab_ab_key_resp.tStart = t  # local t and not account for scr refresh
                    xab_ab_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(xab_ab_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'xab_ab_key_resp.started')
                    # update status
                    xab_ab_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(xab_ab_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(xab_ab_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if xab_ab_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = xab_ab_key_resp.getKeys(keyList=['f','j'], ignoreKeys=["escape"], waitRelease=False)
                    _xab_ab_key_resp_allKeys.extend(theseKeys)
                    if len(_xab_ab_key_resp_allKeys):
                        xab_ab_key_resp.keys = _xab_ab_key_resp_allKeys[-1].name  # just the last key pressed
                        xab_ab_key_resp.rt = _xab_ab_key_resp_allKeys[-1].rt
                        xab_ab_key_resp.duration = _xab_ab_key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from xab_ab_code
                # 强制控制routine不能自动结束
                continueRoutine = True  # 每帧都重新设置为True
                
                # 按键检测代码
                keys = event.getKeys(keyList=['f', 'j'])
                if keys and form_user_key is None:  # 只记录第一次按键
                    form_user_key = keys[0]  # 记录按下的键
                    form_xab_rt = round(core.getTime() - form_start_time, 3)  # 计算反应时
                    
                    # 判断反应是否正确
                    if form_user_key == formal_correct_resp:  # 与条件文件中的正确答案比较
                        form_xab_resp = 1  # 正确
                    else:
                        form_xab_resp = 0  # 错误
                    
                    # 只有在按键后才允许routine结束
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    xab_ab.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in xab_ab.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "xab_ab" ---
            for thisComponent in xab_ab.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for xab_ab
            xab_ab.tStop = globalClock.getTime(format='float')
            xab_ab.tStopRefresh = tThisFlipGlobal
            thisExp.addData('xab_ab.stopped', xab_ab.tStop)
            # check responses
            if xab_ab_key_resp.keys in ['', [], None]:  # No response was made
                xab_ab_key_resp.keys = None
            form_xab.addData('xab_ab_key_resp.keys',xab_ab_key_resp.keys)
            if xab_ab_key_resp.keys != None:  # we had a response
                form_xab.addData('xab_ab_key_resp.rt', xab_ab_key_resp.rt)
                form_xab.addData('xab_ab_key_resp.duration', xab_ab_key_resp.duration)
            # Run 'End Routine' code from xab_ab_code
            # 保存数据到实验文件
            thisExp.addData('formal_xab_block', formal_xab_block)
            thisExp.addData('formal_xab_trial', formal_xab_trial)
            thisExp.addData('form_xab_resp', form_xab_resp)
            thisExp.addData('form_xab_rt', form_xab_rt)
            thisExp.addData('form_user_key', form_user_key)  # 额外保存被试按的键，便于分析
            thisExp.nextEntry()
            # the Routine "xab_ab" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "xab_blank2" ---
            # create an object to store info about Routine xab_blank2
            xab_blank2 = data.Routine(
                name='xab_blank2',
                components=[xab_blank2_text_2],
            )
            xab_blank2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for xab_blank2
            xab_blank2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            xab_blank2.tStart = globalClock.getTime(format='float')
            xab_blank2.status = STARTED
            thisExp.addData('xab_blank2.started', xab_blank2.tStart)
            xab_blank2.maxDuration = None
            # keep track of which components have finished
            xab_blank2Components = xab_blank2.components
            for thisComponent in xab_blank2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "xab_blank2" ---
            # if trial has changed, end Routine now
            if isinstance(form_xab, data.TrialHandler2) and thisForm_xab.thisN != form_xab.thisTrial.thisN:
                continueRoutine = False
            xab_blank2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *xab_blank2_text_2* updates
                
                # if xab_blank2_text_2 is starting this frame...
                if xab_blank2_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    xab_blank2_text_2.frameNStart = frameN  # exact frame index
                    xab_blank2_text_2.tStart = t  # local t and not account for scr refresh
                    xab_blank2_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(xab_blank2_text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'xab_blank2_text_2.started')
                    # update status
                    xab_blank2_text_2.status = STARTED
                    xab_blank2_text_2.setAutoDraw(True)
                
                # if xab_blank2_text_2 is active this frame...
                if xab_blank2_text_2.status == STARTED:
                    # update params
                    pass
                
                # if xab_blank2_text_2 is stopping this frame...
                if xab_blank2_text_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > xab_blank2_text_2.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        xab_blank2_text_2.tStop = t  # not accounting for scr refresh
                        xab_blank2_text_2.tStopRefresh = tThisFlipGlobal  # on global time
                        xab_blank2_text_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'xab_blank2_text_2.stopped')
                        # update status
                        xab_blank2_text_2.status = FINISHED
                        xab_blank2_text_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    xab_blank2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in xab_blank2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "xab_blank2" ---
            for thisComponent in xab_blank2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for xab_blank2
            xab_blank2.tStop = globalClock.getTime(format='float')
            xab_blank2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('xab_blank2.stopped', xab_blank2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if xab_blank2.maxDurationReached:
                routineTimer.addTime(-xab_blank2.maxDuration)
            elif xab_blank2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'form_xab'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "xab_break" ---
        # create an object to store info about Routine xab_break
        xab_break = data.Routine(
            name='xab_break',
            components=[xab_break_center_text, xab_break_bottom_text, xab_break_key_resp],
        )
        xab_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for xab_break_key_resp
        xab_break_key_resp.keys = []
        xab_break_key_resp.rt = []
        _xab_break_key_resp_allKeys = []
        # Run 'Begin Routine' code from xab_break_code
        ##########
        # 每次休息，block计数器＋1
        formal_xab_block += 1
        
        #########
        # 显示实验进度
        progress_text = visual.TextStim(
            win=win,
            text=f'已完成{formal_xab_block - 1}/4部分',  # 直接在这里计算
            font='Arial',
            pos=[0, 0.1],  # 纵坐标0.1
            height=0.03,
            color='black'
        )
        
        # 记录文本开始显示的时间
        text_start_time = core.getTime()  
        # store start times for xab_break
        xab_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        xab_break.tStart = globalClock.getTime(format='float')
        xab_break.status = STARTED
        thisExp.addData('xab_break.started', xab_break.tStart)
        xab_break.maxDuration = None
        # keep track of which components have finished
        xab_breakComponents = xab_break.components
        for thisComponent in xab_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "xab_break" ---
        # if trial has changed, end Routine now
        if isinstance(form_xab_block, data.TrialHandler2) and thisForm_xab_block.thisN != form_xab_block.thisTrial.thisN:
            continueRoutine = False
        xab_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *xab_break_center_text* updates
            
            # if xab_break_center_text is starting this frame...
            if xab_break_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                xab_break_center_text.frameNStart = frameN  # exact frame index
                xab_break_center_text.tStart = t  # local t and not account for scr refresh
                xab_break_center_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(xab_break_center_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xab_break_center_text.started')
                # update status
                xab_break_center_text.status = STARTED
                xab_break_center_text.setAutoDraw(True)
            
            # if xab_break_center_text is active this frame...
            if xab_break_center_text.status == STARTED:
                # update params
                pass
            
            # if xab_break_center_text is stopping this frame...
            if xab_break_center_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > xab_break_center_text.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    xab_break_center_text.tStop = t  # not accounting for scr refresh
                    xab_break_center_text.tStopRefresh = tThisFlipGlobal  # on global time
                    xab_break_center_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'xab_break_center_text.stopped')
                    # update status
                    xab_break_center_text.status = FINISHED
                    xab_break_center_text.setAutoDraw(False)
            
            # *xab_break_bottom_text* updates
            
            # if xab_break_bottom_text is starting this frame...
            if xab_break_bottom_text.status == NOT_STARTED and tThisFlip >= 60-frameTolerance:
                # keep track of start time/frame for later
                xab_break_bottom_text.frameNStart = frameN  # exact frame index
                xab_break_bottom_text.tStart = t  # local t and not account for scr refresh
                xab_break_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(xab_break_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xab_break_bottom_text.started')
                # update status
                xab_break_bottom_text.status = STARTED
                xab_break_bottom_text.setAutoDraw(True)
            
            # if xab_break_bottom_text is active this frame...
            if xab_break_bottom_text.status == STARTED:
                # update params
                pass
            
            # *xab_break_key_resp* updates
            waitOnFlip = False
            
            # if xab_break_key_resp is starting this frame...
            if xab_break_key_resp.status == NOT_STARTED and tThisFlip >= 60-frameTolerance:
                # keep track of start time/frame for later
                xab_break_key_resp.frameNStart = frameN  # exact frame index
                xab_break_key_resp.tStart = t  # local t and not account for scr refresh
                xab_break_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(xab_break_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'xab_break_key_resp.started')
                # update status
                xab_break_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(xab_break_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(xab_break_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if xab_break_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = xab_break_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _xab_break_key_resp_allKeys.extend(theseKeys)
                if len(_xab_break_key_resp_allKeys):
                    xab_break_key_resp.keys = _xab_break_key_resp_allKeys[-1].name  # just the last key pressed
                    xab_break_key_resp.rt = _xab_break_key_resp_allKeys[-1].rt
                    xab_break_key_resp.duration = _xab_break_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from xab_break_code
            # 控制文本只显示60秒
            if core.getTime() - text_start_time <= 60.0:  
                progress_text.draw()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                xab_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in xab_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "xab_break" ---
        for thisComponent in xab_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for xab_break
        xab_break.tStop = globalClock.getTime(format='float')
        xab_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('xab_break.stopped', xab_break.tStop)
        # check responses
        if xab_break_key_resp.keys in ['', [], None]:  # No response was made
            xab_break_key_resp.keys = None
        form_xab_block.addData('xab_break_key_resp.keys',xab_break_key_resp.keys)
        if xab_break_key_resp.keys != None:  # we had a response
            form_xab_block.addData('xab_break_key_resp.rt', xab_break_key_resp.rt)
            form_xab_block.addData('xab_break_key_resp.duration', xab_break_key_resp.duration)
        # the Routine "xab_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 4.0 repeats of 'form_xab_block'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[end_center_text, end_key_resp],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_key_resp
    end_key_resp.keys = []
    end_key_resp.rt = []
    _end_key_resp_allKeys = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_center_text* updates
        
        # if end_center_text is starting this frame...
        if end_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_center_text.frameNStart = frameN  # exact frame index
            end_center_text.tStart = t  # local t and not account for scr refresh
            end_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_center_text.started')
            # update status
            end_center_text.status = STARTED
            end_center_text.setAutoDraw(True)
        
        # if end_center_text is active this frame...
        if end_center_text.status == STARTED:
            # update params
            pass
        
        # *end_key_resp* updates
        waitOnFlip = False
        
        # if end_key_resp is starting this frame...
        if end_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_key_resp.frameNStart = frameN  # exact frame index
            end_key_resp.tStart = t  # local t and not account for scr refresh
            end_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_key_resp.started')
            # update status
            end_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = end_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_key_resp_allKeys.extend(theseKeys)
            if len(_end_key_resp_allKeys):
                end_key_resp.keys = _end_key_resp_allKeys[-1].name  # just the last key pressed
                end_key_resp.rt = _end_key_resp_allKeys[-1].rt
                end_key_resp.duration = _end_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # check responses
    if end_key_resp.keys in ['', [], None]:  # No response was made
        end_key_resp.keys = None
    thisExp.addData('end_key_resp.keys',end_key_resp.keys)
    if end_key_resp.keys != None:  # we had a response
        thisExp.addData('end_key_resp.rt', end_key_resp.rt)
        thisExp.addData('end_key_resp.duration', end_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
