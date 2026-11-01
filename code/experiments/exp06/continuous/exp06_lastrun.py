#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 十二月 09, 2025, at 14:59
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

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'reverse correlation2'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'gender': ['female', 'male'],
    'handedness': ['right', 'left'],
    'age': '',
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
_winSize = [1707, 1067]
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
            logging.getLevel('warning')
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
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('instruction_all_key_resp') is None:
        # initialise instruction_all_key_resp
        instruction_all_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction_all_key_resp',
        )
    if deviceManager.getDevice('instruction_exp_key_resp') is None:
        # initialise instruction_exp_key_resp
        instruction_exp_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction_exp_key_resp',
        )
    if deviceManager.getDevice('instruction_practice_key_resp') is None:
        # initialise instruction_practice_key_resp
        instruction_practice_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction_practice_key_resp',
        )
    if deviceManager.getDevice('practice_rest_key_resp') is None:
        # initialise practice_rest_key_resp
        practice_rest_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practice_rest_key_resp',
        )
    if deviceManager.getDevice('practice_block_key_resp') is None:
        # initialise practice_block_key_resp
        practice_block_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practice_block_key_resp',
        )
    if deviceManager.getDevice('instruction_formal_key_resp') is None:
        # initialise instruction_formal_key_resp
        instruction_formal_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction_formal_key_resp',
        )
    if deviceManager.getDevice('rest_key_resp') is None:
        # initialise rest_key_resp
        rest_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rest_key_resp',
        )
    if deviceManager.getDevice('block_key_resp') is None:
        # initialise block_key_resp
        block_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='block_key_resp',
        )
    if deviceManager.getDevice('formal_continuous_end_key_resp') is None:
        # initialise formal_continuous_end_key_resp
        formal_continuous_end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='formal_continuous_end_key_resp',
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
            backend='ioHub',
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
    # enter 'rush' mode (raise CPU priority)
    if not PILOTING:
        core.rush(enable=True)
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
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
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
    
    # --- Initialize components for Routine "instruction_all" ---
    instruction_all_image = visual.ImageStim(
        win=win,
        name='instruction_all_image', units='norm', 
        image='Instruction/Instruction_all.PNG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instruction_all_key_resp = keyboard.Keyboard(deviceName='instruction_all_key_resp')
    
    # --- Initialize components for Routine "instruction_video" ---
    
    # --- Initialize components for Routine "instruction_exp" ---
    instruction_exp_image = visual.ImageStim(
        win=win,
        name='instruction_exp_image', units='norm', 
        image='Instruction/Instruction_exp.PNG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instruction_exp_key_resp = keyboard.Keyboard(deviceName='instruction_exp_key_resp')
    
    # --- Initialize components for Routine "instruction_practice" ---
    instruction_practice_image = visual.ImageStim(
        win=win,
        name='instruction_practice_image', units='norm', 
        image='Instruction/Instruction_practice.PNG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instruction_practice_key_resp = keyboard.Keyboard(deviceName='instruction_practice_key_resp')
    
    # --- Initialize components for Routine "fixation" ---
    fixation_text = visual.TextStim(win=win, name='fixation_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "practice_continuous_rect" ---
    practice_continuous_rating_marker = visual.Rect(
        win=win, name='practice_continuous_rating_marker',
        width=(0.6, 1)[0], height=(0.6, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    practice_continuous_rating_text = visual.TextStim(win=win, name='practice_continuous_rating_text',
        text='请左右拖动鼠标至您认为的合适位置后按下左键',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    practice_continuous_rating_slider = visual.Slider(win=win, name='practice_continuous_rating_slider',
        startValue=None, size=(0.5, 0.05), pos=(0, 0), units=win.units,
        labels=None, ticks=(0,1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    practice_continuous_rating_mouse = event.Mouse(win=win)
    x, y = [None, None]
    practice_continuous_rating_mouse.mouseClock = core.Clock()
    practice_continuous_image_left = visual.ImageStim(
        win=win,
        name='practice_continuous_image_left', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    practice_continuous_image_right = visual.ImageStim(
        win=win,
        name='practice_continuous_image_right', 
        image='default.png', mask=None, anchor='center',
        ori=1.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    practice_image_feedback = visual.ImageStim(
        win=win,
        name='practice_image_feedback', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.05, -0.24), draggable=False, size=(None, 0.05),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    practice_text_feedback = visual.TextStim(win=win, name='practice_text_feedback',
        text='',
        font='Arial',
        units='height', pos=(0.23, -0.235), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='red', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "practice_rest" ---
    practice_rest_text = visual.TextStim(win=win, name='practice_rest_text',
        text='按“空格“键继续',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    practice_rest_key_resp = keyboard.Keyboard(deviceName='practice_rest_key_resp')
    
    # --- Initialize components for Routine "practice_block" ---
    practice_block_text = visual.TextStim(win=win, name='practice_block_text',
        text='练习结束\n\n您有一分钟休息\n\n按空格进入正式实验',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    practice_block_key_resp = keyboard.Keyboard(deviceName='practice_block_key_resp')
    
    # --- Initialize components for Routine "instruction_formal" ---
    instruction_formal_image = visual.ImageStim(
        win=win,
        name='instruction_formal_image', units='norm', 
        image='Instruction/Instruction_formal.PNG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instruction_formal_key_resp = keyboard.Keyboard(deviceName='instruction_formal_key_resp')
    
    # --- Initialize components for Routine "fixation" ---
    fixation_text = visual.TextStim(win=win, name='fixation_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "formal_continuous_rect" ---
    formal_continuous_rating_marker = visual.Rect(
        win=win, name='formal_continuous_rating_marker',
        width=(0.6, 1)[0], height=(0.6, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    formal_continuous_rating_text = visual.TextStim(win=win, name='formal_continuous_rating_text',
        text='请左右拖动鼠标至您认为的合适位置后按下左键',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    formal_continuous_rating_slider = visual.Slider(win=win, name='formal_continuous_rating_slider',
        startValue=None, size=(0.5, 0.05), pos=(0, 0), units=win.units,
        labels=None, ticks=(0,1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    formal_continuous_rating_mouse = event.Mouse(win=win)
    x, y = [None, None]
    formal_continuous_rating_mouse.mouseClock = core.Clock()
    # Run 'Begin Experiment' code from formal_continuous_rating_code
    trialID = 0
    blockID = 1
    formal_continuous_image_left = visual.ImageStim(
        win=win,
        name='formal_continuous_image_left', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    formal_continuous_image_right = visual.ImageStim(
        win=win,
        name='formal_continuous_image_right', 
        image='default.png', mask=None, anchor='center',
        ori=1.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    
    # --- Initialize components for Routine "rest" ---
    rest_text = visual.TextStim(win=win, name='rest_text',
        text='按“空格”键继续',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    rest_key_resp = keyboard.Keyboard(deviceName='rest_key_resp')
    
    # --- Initialize components for Routine "block" ---
    block_text = visual.TextStim(win=win, name='block_text',
        text='您有一分钟时间休息\n\n按“空格”结束休息',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    block_key_resp = keyboard.Keyboard(deviceName='block_key_resp')
    
    # --- Initialize components for Routine "formal_continuous_end" ---
    formal_continuous_end_text = visual.TextStim(win=win, name='formal_continuous_end_text',
        text='实验结束！谢谢参与！\n\n按空格退出\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_continuous_end_key_resp = keyboard.Keyboard(deviceName='formal_continuous_end_key_resp')
    
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
    
    # --- Prepare to start Routine "instruction_all" ---
    # create an object to store info about Routine instruction_all
    instruction_all = data.Routine(
        name='instruction_all',
        components=[instruction_all_image, instruction_all_key_resp],
    )
    instruction_all.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction_all_key_resp
    instruction_all_key_resp.keys = []
    instruction_all_key_resp.rt = []
    _instruction_all_key_resp_allKeys = []
    # store start times for instruction_all
    instruction_all.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction_all.tStart = globalClock.getTime(format='float')
    instruction_all.status = STARTED
    thisExp.addData('instruction_all.started', instruction_all.tStart)
    instruction_all.maxDuration = None
    # keep track of which components have finished
    instruction_allComponents = instruction_all.components
    for thisComponent in instruction_all.components:
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
    
    # --- Run Routine "instruction_all" ---
    instruction_all.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_all_image* updates
        
        # if instruction_all_image is starting this frame...
        if instruction_all_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_all_image.frameNStart = frameN  # exact frame index
            instruction_all_image.tStart = t  # local t and not account for scr refresh
            instruction_all_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_all_image, 'tStartRefresh')  # time at next scr refresh
            # update status
            instruction_all_image.status = STARTED
            instruction_all_image.setAutoDraw(True)
        
        # if instruction_all_image is active this frame...
        if instruction_all_image.status == STARTED:
            # update params
            pass
        
        # *instruction_all_key_resp* updates
        waitOnFlip = False
        
        # if instruction_all_key_resp is starting this frame...
        if instruction_all_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_all_key_resp.frameNStart = frameN  # exact frame index
            instruction_all_key_resp.tStart = t  # local t and not account for scr refresh
            instruction_all_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_all_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_all_key_resp.started')
            # update status
            instruction_all_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction_all_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction_all_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction_all_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instruction_all_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction_all_key_resp_allKeys.extend(theseKeys)
            if len(_instruction_all_key_resp_allKeys):
                instruction_all_key_resp.keys = _instruction_all_key_resp_allKeys[-1].name  # just the last key pressed
                instruction_all_key_resp.rt = _instruction_all_key_resp_allKeys[-1].rt
                instruction_all_key_resp.duration = _instruction_all_key_resp_allKeys[-1].duration
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
            instruction_all.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruction_all.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_all" ---
    for thisComponent in instruction_all.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction_all
    instruction_all.tStop = globalClock.getTime(format='float')
    instruction_all.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction_all.stopped', instruction_all.tStop)
    # check responses
    if instruction_all_key_resp.keys in ['', [], None]:  # No response was made
        instruction_all_key_resp.keys = None
    thisExp.addData('instruction_all_key_resp.keys',instruction_all_key_resp.keys)
    if instruction_all_key_resp.keys != None:  # we had a response
        thisExp.addData('instruction_all_key_resp.rt', instruction_all_key_resp.rt)
        thisExp.addData('instruction_all_key_resp.duration', instruction_all_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instruction_all" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction_video" ---
    # create an object to store info about Routine instruction_video
    instruction_video = data.Routine(
        name='instruction_video',
        components=[],
    )
    instruction_video.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for instruction_video
    instruction_video.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction_video.tStart = globalClock.getTime(format='float')
    instruction_video.status = STARTED
    thisExp.addData('instruction_video.started', instruction_video.tStart)
    instruction_video.maxDuration = None
    # keep track of which components have finished
    instruction_videoComponents = instruction_video.components
    for thisComponent in instruction_video.components:
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
    
    # --- Run Routine "instruction_video" ---
    instruction_video.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
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
            instruction_video.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruction_video.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_video" ---
    for thisComponent in instruction_video.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction_video
    instruction_video.tStop = globalClock.getTime(format='float')
    instruction_video.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction_video.stopped', instruction_video.tStop)
    thisExp.nextEntry()
    # the Routine "instruction_video" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction_exp" ---
    # create an object to store info about Routine instruction_exp
    instruction_exp = data.Routine(
        name='instruction_exp',
        components=[instruction_exp_image, instruction_exp_key_resp],
    )
    instruction_exp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction_exp_key_resp
    instruction_exp_key_resp.keys = []
    instruction_exp_key_resp.rt = []
    _instruction_exp_key_resp_allKeys = []
    # store start times for instruction_exp
    instruction_exp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction_exp.tStart = globalClock.getTime(format='float')
    instruction_exp.status = STARTED
    thisExp.addData('instruction_exp.started', instruction_exp.tStart)
    instruction_exp.maxDuration = None
    # keep track of which components have finished
    instruction_expComponents = instruction_exp.components
    for thisComponent in instruction_exp.components:
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
    
    # --- Run Routine "instruction_exp" ---
    instruction_exp.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_exp_image* updates
        
        # if instruction_exp_image is starting this frame...
        if instruction_exp_image.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_exp_image.frameNStart = frameN  # exact frame index
            instruction_exp_image.tStart = t  # local t and not account for scr refresh
            instruction_exp_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_exp_image, 'tStartRefresh')  # time at next scr refresh
            # update status
            instruction_exp_image.status = STARTED
            instruction_exp_image.setAutoDraw(True)
        
        # if instruction_exp_image is active this frame...
        if instruction_exp_image.status == STARTED:
            # update params
            pass
        
        # *instruction_exp_key_resp* updates
        waitOnFlip = False
        
        # if instruction_exp_key_resp is starting this frame...
        if instruction_exp_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_exp_key_resp.frameNStart = frameN  # exact frame index
            instruction_exp_key_resp.tStart = t  # local t and not account for scr refresh
            instruction_exp_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_exp_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_exp_key_resp.started')
            # update status
            instruction_exp_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction_exp_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction_exp_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction_exp_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instruction_exp_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction_exp_key_resp_allKeys.extend(theseKeys)
            if len(_instruction_exp_key_resp_allKeys):
                instruction_exp_key_resp.keys = _instruction_exp_key_resp_allKeys[-1].name  # just the last key pressed
                instruction_exp_key_resp.rt = _instruction_exp_key_resp_allKeys[-1].rt
                instruction_exp_key_resp.duration = _instruction_exp_key_resp_allKeys[-1].duration
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
            instruction_exp.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruction_exp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_exp" ---
    for thisComponent in instruction_exp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction_exp
    instruction_exp.tStop = globalClock.getTime(format='float')
    instruction_exp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction_exp.stopped', instruction_exp.tStop)
    # check responses
    if instruction_exp_key_resp.keys in ['', [], None]:  # No response was made
        instruction_exp_key_resp.keys = None
    thisExp.addData('instruction_exp_key_resp.keys',instruction_exp_key_resp.keys)
    if instruction_exp_key_resp.keys != None:  # we had a response
        thisExp.addData('instruction_exp_key_resp.rt', instruction_exp_key_resp.rt)
        thisExp.addData('instruction_exp_key_resp.duration', instruction_exp_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instruction_exp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction_practice" ---
    # create an object to store info about Routine instruction_practice
    instruction_practice = data.Routine(
        name='instruction_practice',
        components=[instruction_practice_image, instruction_practice_key_resp],
    )
    instruction_practice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction_practice_key_resp
    instruction_practice_key_resp.keys = []
    instruction_practice_key_resp.rt = []
    _instruction_practice_key_resp_allKeys = []
    # store start times for instruction_practice
    instruction_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction_practice.tStart = globalClock.getTime(format='float')
    instruction_practice.status = STARTED
    thisExp.addData('instruction_practice.started', instruction_practice.tStart)
    instruction_practice.maxDuration = None
    # keep track of which components have finished
    instruction_practiceComponents = instruction_practice.components
    for thisComponent in instruction_practice.components:
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
    
    # --- Run Routine "instruction_practice" ---
    instruction_practice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_practice_image* updates
        
        # if instruction_practice_image is starting this frame...
        if instruction_practice_image.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_practice_image.frameNStart = frameN  # exact frame index
            instruction_practice_image.tStart = t  # local t and not account for scr refresh
            instruction_practice_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_practice_image, 'tStartRefresh')  # time at next scr refresh
            # update status
            instruction_practice_image.status = STARTED
            instruction_practice_image.setAutoDraw(True)
        
        # if instruction_practice_image is active this frame...
        if instruction_practice_image.status == STARTED:
            # update params
            pass
        
        # *instruction_practice_key_resp* updates
        waitOnFlip = False
        
        # if instruction_practice_key_resp is starting this frame...
        if instruction_practice_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_practice_key_resp.frameNStart = frameN  # exact frame index
            instruction_practice_key_resp.tStart = t  # local t and not account for scr refresh
            instruction_practice_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_practice_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_practice_key_resp.started')
            # update status
            instruction_practice_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction_practice_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction_practice_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction_practice_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instruction_practice_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction_practice_key_resp_allKeys.extend(theseKeys)
            if len(_instruction_practice_key_resp_allKeys):
                instruction_practice_key_resp.keys = _instruction_practice_key_resp_allKeys[-1].name  # just the last key pressed
                instruction_practice_key_resp.rt = _instruction_practice_key_resp_allKeys[-1].rt
                instruction_practice_key_resp.duration = _instruction_practice_key_resp_allKeys[-1].duration
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
            instruction_practice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruction_practice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_practice" ---
    for thisComponent in instruction_practice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction_practice
    instruction_practice.tStop = globalClock.getTime(format='float')
    instruction_practice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction_practice.stopped', instruction_practice.tStop)
    # check responses
    if instruction_practice_key_resp.keys in ['', [], None]:  # No response was made
        instruction_practice_key_resp.keys = None
    thisExp.addData('instruction_practice_key_resp.keys',instruction_practice_key_resp.keys)
    if instruction_practice_key_resp.keys != None:  # we had a response
        thisExp.addData('instruction_practice_key_resp.rt', instruction_practice_key_resp.rt)
        thisExp.addData('instruction_practice_key_resp.duration', instruction_practice_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instruction_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler2(
        name='practice_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'practice.xlsx', 
        selection='0:3'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[fixation_text],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
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
        
        # --- Run Routine "fixation" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_text* updates
            
            # if fixation_text is starting this frame...
            if fixation_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_text.frameNStart = frameN  # exact frame index
                fixation_text.tStart = t  # local t and not account for scr refresh
                fixation_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_text.started')
                # update status
                fixation_text.status = STARTED
                fixation_text.setAutoDraw(True)
            
            # if fixation_text is active this frame...
            if fixation_text.status == STARTED:
                # update params
                pass
            
            # if fixation_text is stopping this frame...
            if fixation_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_text.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_text.tStop = t  # not accounting for scr refresh
                    fixation_text.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_text.stopped')
                    # update status
                    fixation_text.status = FINISHED
                    fixation_text.setAutoDraw(False)
            
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
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "practice_continuous_rect" ---
        # create an object to store info about Routine practice_continuous_rect
        practice_continuous_rect = data.Routine(
            name='practice_continuous_rect',
            components=[practice_continuous_rating_marker, practice_continuous_rating_text, practice_continuous_rating_slider, practice_continuous_rating_mouse, practice_continuous_image_left, practice_continuous_image_right, practice_image_feedback, practice_text_feedback],
        )
        practice_continuous_rect.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        practice_continuous_rating_slider.reset()
        # setup some python lists for storing info about the practice_continuous_rating_mouse
        practice_continuous_rating_mouse.x = []
        practice_continuous_rating_mouse.y = []
        practice_continuous_rating_mouse.leftButton = []
        practice_continuous_rating_mouse.midButton = []
        practice_continuous_rating_mouse.rightButton = []
        practice_continuous_rating_mouse.time = []
        practice_continuous_rating_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        practice_continuous_rating_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from practice_continuous_rating_code
        # --- 初始化评分参数 ---
        practice_continuous_rating_slider_width = 0.5       # 滑块宽度（相对单位）
        practice_continuous_rating_slider_height = 0.05      # 滑块高度
        practice_continuous_rating_slider_orientation = 0    # 滑块方向（0：水平）
        practice_continuous_rating_slider_ticks = [0, 1]       # 滑块刻度
        practice_continuous_rating_slider_granularity = 0.1    # 滑块精度
        practice_continuous_rating_slider_decimals = 2         # 显示小数位数
        
        # --- 设置滑块样式 ---
        practice_continuous_rating_slider.decimals = practice_continuous_rating_slider_decimals
        marker = practice_continuous_rating_slider.marker
        marker.shape = 'square'                   # 标记形状为方形
        marker.size = (0.01, 0.05)                # 标记大小
        
        # --- 初始化变量 ---
        start_time = core.getTime() # 当前例程起始时间
        practice_continuous_rating_rt = None                        # 记录响应时间
        practice_continuous_rating = None # 连续响应值
        
        print("practice_continuous_rating 的类型是:", type(practice_continuous_rating))
        
        print(trialID)
        
        mouseRec = practice_continuous_rating_mouse.getPos() 
        practice_continuous_rating_mouse.setPos((0, 0))
        
        has_moved = False
        
        #####反馈设置初始化#####
        feedback_pic_list = ['Instruction/right_fear.png','Instruction/left_fear.png']
        feedback_text = 0
        feedback_point = 0
        feedback_pic = randchoice(feedback_pic_list)
        
        practice_continuous_image_left.setPos((-0.57, 0))
        practice_continuous_image_left.setSize((0.3, 0.3))
        practice_continuous_image_left.setImage(left_image)
        practice_continuous_image_right.setPos((0.57, 0))
        practice_continuous_image_right.setSize((0.3, 0.3))
        practice_continuous_image_right.setOri(0.0)
        practice_continuous_image_right.setImage(right_image)
        # store start times for practice_continuous_rect
        practice_continuous_rect.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_continuous_rect.tStart = globalClock.getTime(format='float')
        practice_continuous_rect.status = STARTED
        thisExp.addData('practice_continuous_rect.started', practice_continuous_rect.tStart)
        practice_continuous_rect.maxDuration = None
        # keep track of which components have finished
        practice_continuous_rectComponents = practice_continuous_rect.components
        for thisComponent in practice_continuous_rect.components:
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
        
        # --- Run Routine "practice_continuous_rect" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        practice_continuous_rect.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_continuous_rating_marker* updates
            
            # if practice_continuous_rating_marker is starting this frame...
            if practice_continuous_rating_marker.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_continuous_rating_marker.frameNStart = frameN  # exact frame index
                practice_continuous_rating_marker.tStart = t  # local t and not account for scr refresh
                practice_continuous_rating_marker.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_continuous_rating_marker, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_continuous_rating_marker.started')
                # update status
                practice_continuous_rating_marker.status = STARTED
                practice_continuous_rating_marker.setAutoDraw(True)
            
            # if practice_continuous_rating_marker is active this frame...
            if practice_continuous_rating_marker.status == STARTED:
                # update params
                pass
            
            # *practice_continuous_rating_text* updates
            
            # if practice_continuous_rating_text is starting this frame...
            if practice_continuous_rating_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_continuous_rating_text.frameNStart = frameN  # exact frame index
                practice_continuous_rating_text.tStart = t  # local t and not account for scr refresh
                practice_continuous_rating_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_continuous_rating_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_continuous_rating_text.started')
                # update status
                practice_continuous_rating_text.status = STARTED
                practice_continuous_rating_text.setAutoDraw(True)
            
            # if practice_continuous_rating_text is active this frame...
            if practice_continuous_rating_text.status == STARTED:
                # update params
                pass
            
            # *practice_continuous_rating_slider* updates
            
            # if practice_continuous_rating_slider is starting this frame...
            if practice_continuous_rating_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_continuous_rating_slider.frameNStart = frameN  # exact frame index
                practice_continuous_rating_slider.tStart = t  # local t and not account for scr refresh
                practice_continuous_rating_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_continuous_rating_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_continuous_rating_slider.started')
                # update status
                practice_continuous_rating_slider.status = STARTED
                practice_continuous_rating_slider.setAutoDraw(True)
            
            # if practice_continuous_rating_slider is active this frame...
            if practice_continuous_rating_slider.status == STARTED:
                # update params
                pass
            # *practice_continuous_rating_mouse* updates
            
            # if practice_continuous_rating_mouse is starting this frame...
            if practice_continuous_rating_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_continuous_rating_mouse.frameNStart = frameN  # exact frame index
                practice_continuous_rating_mouse.tStart = t  # local t and not account for scr refresh
                practice_continuous_rating_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_continuous_rating_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('practice_continuous_rating_mouse.started', t)
                # update status
                practice_continuous_rating_mouse.status = STARTED
                prevButtonState = practice_continuous_rating_mouse.getPressed()  # if button is down already this ISN'T a new click
            if practice_continuous_rating_mouse.status == STARTED:  # only update if started and not finished!
                buttons = practice_continuous_rating_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(practice_continuous_rating_marker, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(practice_continuous_rating_mouse):
                                gotValidClick = True
                                practice_continuous_rating_mouse.clicked_name.append(obj.name)
                                practice_continuous_rating_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = practice_continuous_rating_mouse.getPos()
                            practice_continuous_rating_mouse.x.append(x)
                            practice_continuous_rating_mouse.y.append(y)
                            buttons = practice_continuous_rating_mouse.getPressed()
                            practice_continuous_rating_mouse.leftButton.append(buttons[0])
                            practice_continuous_rating_mouse.midButton.append(buttons[1])
                            practice_continuous_rating_mouse.rightButton.append(buttons[2])
                            practice_continuous_rating_mouse.time.append(practice_continuous_rating_mouse.mouseClock.getTime())
            # Run 'Each Frame' code from practice_continuous_rating_code
            
            # 实时更新 marker 位置
            mouse_Rec = practice_continuous_rating_mouse.getPos()
            if practice_continuous_rating_marker.contains(practice_continuous_rating_mouse):
               if mouse_Rec[0] != 0:
                    has_moved = True
               practice_continuous_rating_slider.markerPos = mouse_Rec[practice_continuous_rating_slider_orientation] / practice_continuous_rating_slider_width * (practice_continuous_rating_slider_ticks[-1]-practice_continuous_rating_slider_ticks[0]) + (practice_continuous_rating_slider_ticks[0]+practice_continuous_rating_slider_ticks[-1]) / 2
               if practice_continuous_rating_mouse.getPressed()[0] and has_moved:
                   practice_continuous_rating = round(practice_continuous_rating_slider.markerPos, practice_continuous_rating_slider_decimals)           
                   if practice_continuous_rating_rt is None:  # 仅记录第一次点击的反应时
                        practice_continuous_rating_rt = core.getTime() - start_time
               else:
                   practice_continuous_rating = None
            elif mouse_Rec[0] > 0.25:
                practice_continuous_rating_mouse.setPos((0.25, 0))
            elif mouse_Rec[0] < -0.25:
                practice_continuous_rating_mouse.setPos((-0.25, 0))
            win.mouseVisible = False
            
            feedback_point = int((practice_continuous_rating_slider.markerPos - 0.5)*200)
            if feedback_point  > 0:
                feedback_pic = feedback_pic_list[0]
                feedback_text = abs(feedback_point)
            elif feedback_point  < 0:
                feedback_pic = feedback_pic_list[1]
                feedback_text = abs(feedback_point)
            
            if gotValidClick and has_moved:
                continueRoutine = False
            else:
                gotValidClick = False
            
            
            # *practice_continuous_image_left* updates
            
            # if practice_continuous_image_left is starting this frame...
            if practice_continuous_image_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_continuous_image_left.frameNStart = frameN  # exact frame index
                practice_continuous_image_left.tStart = t  # local t and not account for scr refresh
                practice_continuous_image_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_continuous_image_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_continuous_image_left.started')
                # update status
                practice_continuous_image_left.status = STARTED
                practice_continuous_image_left.setAutoDraw(True)
            
            # if practice_continuous_image_left is active this frame...
            if practice_continuous_image_left.status == STARTED:
                # update params
                pass
            
            # *practice_continuous_image_right* updates
            
            # if practice_continuous_image_right is starting this frame...
            if practice_continuous_image_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_continuous_image_right.frameNStart = frameN  # exact frame index
                practice_continuous_image_right.tStart = t  # local t and not account for scr refresh
                practice_continuous_image_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_continuous_image_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_continuous_image_right.started')
                # update status
                practice_continuous_image_right.status = STARTED
                practice_continuous_image_right.setAutoDraw(True)
            
            # if practice_continuous_image_right is active this frame...
            if practice_continuous_image_right.status == STARTED:
                # update params
                pass
            
            # *practice_image_feedback* updates
            
            # if practice_image_feedback is starting this frame...
            if practice_image_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_image_feedback.frameNStart = frameN  # exact frame index
                practice_image_feedback.tStart = t  # local t and not account for scr refresh
                practice_image_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_image_feedback, 'tStartRefresh')  # time at next scr refresh
                # update status
                practice_image_feedback.status = STARTED
                practice_image_feedback.setAutoDraw(True)
            
            # if practice_image_feedback is active this frame...
            if practice_image_feedback.status == STARTED:
                # update params
                practice_image_feedback.setImage(feedback_pic, log=False)
            
            # *practice_text_feedback* updates
            
            # if practice_text_feedback is starting this frame...
            if practice_text_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_text_feedback.frameNStart = frameN  # exact frame index
                practice_text_feedback.tStart = t  # local t and not account for scr refresh
                practice_text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_text_feedback, 'tStartRefresh')  # time at next scr refresh
                # update status
                practice_text_feedback.status = STARTED
                practice_text_feedback.setAutoDraw(True)
            
            # if practice_text_feedback is active this frame...
            if practice_text_feedback.status == STARTED:
                # update params
                practice_text_feedback.setText(feedback_text
                , log=False)
            
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
                practice_continuous_rect.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_continuous_rect.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_continuous_rect" ---
        for thisComponent in practice_continuous_rect.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_continuous_rect
        practice_continuous_rect.tStop = globalClock.getTime(format='float')
        practice_continuous_rect.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_continuous_rect.stopped', practice_continuous_rect.tStop)
        practice_loop.addData('practice_continuous_rating_slider.response', practice_continuous_rating_slider.getRating())
        practice_loop.addData('practice_continuous_rating_slider.rt', practice_continuous_rating_slider.getRT())
        # store data for practice_loop (TrialHandler)
        practice_loop.addData('practice_continuous_rating_mouse.x', practice_continuous_rating_mouse.x)
        practice_loop.addData('practice_continuous_rating_mouse.y', practice_continuous_rating_mouse.y)
        practice_loop.addData('practice_continuous_rating_mouse.leftButton', practice_continuous_rating_mouse.leftButton)
        practice_loop.addData('practice_continuous_rating_mouse.midButton', practice_continuous_rating_mouse.midButton)
        practice_loop.addData('practice_continuous_rating_mouse.rightButton', practice_continuous_rating_mouse.rightButton)
        practice_loop.addData('practice_continuous_rating_mouse.time', practice_continuous_rating_mouse.time)
        practice_loop.addData('practice_continuous_rating_mouse.clicked_name', practice_continuous_rating_mouse.clicked_name)
        # Run 'End Routine' code from practice_continuous_rating_code
        
        # 创建新变量side_chosen
        if practice_continuous_rating_mouse.x:  # 检查是否有鼠标数据
            # 取最后一次记录的x坐标
            last_x = practice_continuous_rating_mouse.x[-1]
            
            # 判断x值并赋值
            if last_x > 0:
                side_chosen = 1
            elif last_x < 0:
                side_chosen = 0
            else:
                side_chosen = -1  # 处理x=0的情况
        else:
            side_chosen = None  # 无数据时设为空
        
        #保存数据
        thisExp.addData('practice_continuous_rating_rt', practice_continuous_rating_rt)
        thisExp.addData('practice_continuous_rating', practice_continuous_rating)
        thisExp.addData('Raw_chosen',last_x)#记录原始选择
        thisExp.addData('side_chosen', side_chosen)  # 新增的数据列
        win.mouseVisible = False
        
        # the Routine "practice_continuous_rect" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "practice_rest" ---
        # create an object to store info about Routine practice_rest
        practice_rest = data.Routine(
            name='practice_rest',
            components=[practice_rest_text, practice_rest_key_resp],
        )
        practice_rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for practice_rest_key_resp
        practice_rest_key_resp.keys = []
        practice_rest_key_resp.rt = []
        _practice_rest_key_resp_allKeys = []
        # store start times for practice_rest
        practice_rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_rest.tStart = globalClock.getTime(format='float')
        practice_rest.status = STARTED
        thisExp.addData('practice_rest.started', practice_rest.tStart)
        practice_rest.maxDuration = None
        # keep track of which components have finished
        practice_restComponents = practice_rest.components
        for thisComponent in practice_rest.components:
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
        
        # --- Run Routine "practice_rest" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        practice_rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_rest_text* updates
            
            # if practice_rest_text is starting this frame...
            if practice_rest_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_rest_text.frameNStart = frameN  # exact frame index
                practice_rest_text.tStart = t  # local t and not account for scr refresh
                practice_rest_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_rest_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_rest_text.started')
                # update status
                practice_rest_text.status = STARTED
                practice_rest_text.setAutoDraw(True)
            
            # if practice_rest_text is active this frame...
            if practice_rest_text.status == STARTED:
                # update params
                pass
            
            # *practice_rest_key_resp* updates
            waitOnFlip = False
            
            # if practice_rest_key_resp is starting this frame...
            if practice_rest_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_rest_key_resp.frameNStart = frameN  # exact frame index
                practice_rest_key_resp.tStart = t  # local t and not account for scr refresh
                practice_rest_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_rest_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_rest_key_resp.started')
                # update status
                practice_rest_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_rest_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(practice_rest_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if practice_rest_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = practice_rest_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _practice_rest_key_resp_allKeys.extend(theseKeys)
                if len(_practice_rest_key_resp_allKeys):
                    practice_rest_key_resp.keys = _practice_rest_key_resp_allKeys[-1].name  # just the last key pressed
                    practice_rest_key_resp.rt = _practice_rest_key_resp_allKeys[-1].rt
                    practice_rest_key_resp.duration = _practice_rest_key_resp_allKeys[-1].duration
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
                practice_rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_rest" ---
        for thisComponent in practice_rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_rest
        practice_rest.tStop = globalClock.getTime(format='float')
        practice_rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_rest.stopped', practice_rest.tStop)
        # check responses
        if practice_rest_key_resp.keys in ['', [], None]:  # No response was made
            practice_rest_key_resp.keys = None
        practice_loop.addData('practice_rest_key_resp.keys',practice_rest_key_resp.keys)
        if practice_rest_key_resp.keys != None:  # we had a response
            practice_loop.addData('practice_rest_key_resp.rt', practice_rest_key_resp.rt)
            practice_loop.addData('practice_rest_key_resp.duration', practice_rest_key_resp.duration)
        # the Routine "practice_rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "practice_block" ---
    # create an object to store info about Routine practice_block
    practice_block = data.Routine(
        name='practice_block',
        components=[practice_block_text, practice_block_key_resp],
    )
    practice_block.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for practice_block_key_resp
    practice_block_key_resp.keys = []
    practice_block_key_resp.rt = []
    _practice_block_key_resp_allKeys = []
    # store start times for practice_block
    practice_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    practice_block.tStart = globalClock.getTime(format='float')
    practice_block.status = STARTED
    thisExp.addData('practice_block.started', practice_block.tStart)
    practice_block.maxDuration = None
    # keep track of which components have finished
    practice_blockComponents = practice_block.components
    for thisComponent in practice_block.components:
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
    
    # --- Run Routine "practice_block" ---
    practice_block.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_block_text* updates
        
        # if practice_block_text is starting this frame...
        if practice_block_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_block_text.frameNStart = frameN  # exact frame index
            practice_block_text.tStart = t  # local t and not account for scr refresh
            practice_block_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_block_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_block_text.started')
            # update status
            practice_block_text.status = STARTED
            practice_block_text.setAutoDraw(True)
        
        # if practice_block_text is active this frame...
        if practice_block_text.status == STARTED:
            # update params
            pass
        
        # *practice_block_key_resp* updates
        waitOnFlip = False
        
        # if practice_block_key_resp is starting this frame...
        if practice_block_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_block_key_resp.frameNStart = frameN  # exact frame index
            practice_block_key_resp.tStart = t  # local t and not account for scr refresh
            practice_block_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_block_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_block_key_resp.started')
            # update status
            practice_block_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(practice_block_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(practice_block_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if practice_block_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = practice_block_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _practice_block_key_resp_allKeys.extend(theseKeys)
            if len(_practice_block_key_resp_allKeys):
                practice_block_key_resp.keys = _practice_block_key_resp_allKeys[-1].name  # just the last key pressed
                practice_block_key_resp.rt = _practice_block_key_resp_allKeys[-1].rt
                practice_block_key_resp.duration = _practice_block_key_resp_allKeys[-1].duration
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
            practice_block.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_block.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_block" ---
    for thisComponent in practice_block.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for practice_block
    practice_block.tStop = globalClock.getTime(format='float')
    practice_block.tStopRefresh = tThisFlipGlobal
    thisExp.addData('practice_block.stopped', practice_block.tStop)
    # check responses
    if practice_block_key_resp.keys in ['', [], None]:  # No response was made
        practice_block_key_resp.keys = None
    thisExp.addData('practice_block_key_resp.keys',practice_block_key_resp.keys)
    if practice_block_key_resp.keys != None:  # we had a response
        thisExp.addData('practice_block_key_resp.rt', practice_block_key_resp.rt)
        thisExp.addData('practice_block_key_resp.duration', practice_block_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "practice_block" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction_formal" ---
    # create an object to store info about Routine instruction_formal
    instruction_formal = data.Routine(
        name='instruction_formal',
        components=[instruction_formal_image, instruction_formal_key_resp],
    )
    instruction_formal.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction_formal_key_resp
    instruction_formal_key_resp.keys = []
    instruction_formal_key_resp.rt = []
    _instruction_formal_key_resp_allKeys = []
    # store start times for instruction_formal
    instruction_formal.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction_formal.tStart = globalClock.getTime(format='float')
    instruction_formal.status = STARTED
    thisExp.addData('instruction_formal.started', instruction_formal.tStart)
    instruction_formal.maxDuration = None
    # keep track of which components have finished
    instruction_formalComponents = instruction_formal.components
    for thisComponent in instruction_formal.components:
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
    
    # --- Run Routine "instruction_formal" ---
    instruction_formal.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_formal_image* updates
        
        # if instruction_formal_image is starting this frame...
        if instruction_formal_image.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_formal_image.frameNStart = frameN  # exact frame index
            instruction_formal_image.tStart = t  # local t and not account for scr refresh
            instruction_formal_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_formal_image, 'tStartRefresh')  # time at next scr refresh
            # update status
            instruction_formal_image.status = STARTED
            instruction_formal_image.setAutoDraw(True)
        
        # if instruction_formal_image is active this frame...
        if instruction_formal_image.status == STARTED:
            # update params
            pass
        
        # *instruction_formal_key_resp* updates
        waitOnFlip = False
        
        # if instruction_formal_key_resp is starting this frame...
        if instruction_formal_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_formal_key_resp.frameNStart = frameN  # exact frame index
            instruction_formal_key_resp.tStart = t  # local t and not account for scr refresh
            instruction_formal_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_formal_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_formal_key_resp.started')
            # update status
            instruction_formal_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction_formal_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction_formal_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction_formal_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instruction_formal_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction_formal_key_resp_allKeys.extend(theseKeys)
            if len(_instruction_formal_key_resp_allKeys):
                instruction_formal_key_resp.keys = _instruction_formal_key_resp_allKeys[-1].name  # just the last key pressed
                instruction_formal_key_resp.rt = _instruction_formal_key_resp_allKeys[-1].rt
                instruction_formal_key_resp.duration = _instruction_formal_key_resp_allKeys[-1].duration
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
            instruction_formal.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruction_formal.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_formal" ---
    for thisComponent in instruction_formal.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction_formal
    instruction_formal.tStop = globalClock.getTime(format='float')
    instruction_formal.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction_formal.stopped', instruction_formal.tStop)
    # check responses
    if instruction_formal_key_resp.keys in ['', [], None]:  # No response was made
        instruction_formal_key_resp.keys = None
    thisExp.addData('instruction_formal_key_resp.keys',instruction_formal_key_resp.keys)
    if instruction_formal_key_resp.keys != None:  # we had a response
        thisExp.addData('instruction_formal_key_resp.rt', instruction_formal_key_resp.rt)
        thisExp.addData('instruction_formal_key_resp.duration', instruction_formal_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instruction_formal" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal_continuous_rating_trials_loop = data.TrialHandler2(
        name='formal_continuous_rating_trials_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('stim_formal2.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(formal_continuous_rating_trials_loop)  # add the loop to the experiment
    thisFormal_continuous_rating_trials_loop = formal_continuous_rating_trials_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFormal_continuous_rating_trials_loop.rgb)
    if thisFormal_continuous_rating_trials_loop != None:
        for paramName in thisFormal_continuous_rating_trials_loop:
            globals()[paramName] = thisFormal_continuous_rating_trials_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFormal_continuous_rating_trials_loop in formal_continuous_rating_trials_loop:
        currentLoop = formal_continuous_rating_trials_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFormal_continuous_rating_trials_loop.rgb)
        if thisFormal_continuous_rating_trials_loop != None:
            for paramName in thisFormal_continuous_rating_trials_loop:
                globals()[paramName] = thisFormal_continuous_rating_trials_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[fixation_text],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
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
        
        # --- Run Routine "fixation" ---
        # if trial has changed, end Routine now
        if isinstance(formal_continuous_rating_trials_loop, data.TrialHandler2) and thisFormal_continuous_rating_trials_loop.thisN != formal_continuous_rating_trials_loop.thisTrial.thisN:
            continueRoutine = False
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_text* updates
            
            # if fixation_text is starting this frame...
            if fixation_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_text.frameNStart = frameN  # exact frame index
                fixation_text.tStart = t  # local t and not account for scr refresh
                fixation_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_text.started')
                # update status
                fixation_text.status = STARTED
                fixation_text.setAutoDraw(True)
            
            # if fixation_text is active this frame...
            if fixation_text.status == STARTED:
                # update params
                pass
            
            # if fixation_text is stopping this frame...
            if fixation_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_text.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_text.tStop = t  # not accounting for scr refresh
                    fixation_text.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_text.stopped')
                    # update status
                    fixation_text.status = FINISHED
                    fixation_text.setAutoDraw(False)
            
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
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "formal_continuous_rect" ---
        # create an object to store info about Routine formal_continuous_rect
        formal_continuous_rect = data.Routine(
            name='formal_continuous_rect',
            components=[formal_continuous_rating_marker, formal_continuous_rating_text, formal_continuous_rating_slider, formal_continuous_rating_mouse, formal_continuous_image_left, formal_continuous_image_right],
        )
        formal_continuous_rect.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        formal_continuous_rating_slider.reset()
        # setup some python lists for storing info about the formal_continuous_rating_mouse
        formal_continuous_rating_mouse.x = []
        formal_continuous_rating_mouse.y = []
        formal_continuous_rating_mouse.leftButton = []
        formal_continuous_rating_mouse.midButton = []
        formal_continuous_rating_mouse.rightButton = []
        formal_continuous_rating_mouse.time = []
        formal_continuous_rating_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        formal_continuous_rating_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from formal_continuous_rating_code
        # --- 初始化评分参数 ---
        formal_continuous_rating_slider_width = 0.5       # 滑块宽度（相对单位）
        formal_continuous_rating_slider_height = 0.05      # 滑块高度
        formal_continuous_rating_slider_orientation = 0    # 滑块方向（0：水平）
        formal_continuous_rating_slider_ticks = [0, 1]       # 滑块刻度
        formal_continuous_rating_slider_granularity = 0.1    # 滑块精度
        formal_continuous_rating_slider_decimals = 2         # 显示小数位数
        
        # --- 设置滑块样式 ---
        formal_continuous_rating_slider.decimals = formal_continuous_rating_slider_decimals
        marker = formal_continuous_rating_slider.marker
        marker.shape = 'square'                   # 标记形状为方形
        marker.size = (0.01, 0.05)                # 标记大小
        
        # --- 初始化变量 ---
        start_time = core.getTime() # 当前例程起始时间
        formal_continuous_rating_rt = None                        # 记录响应时间
        formal_continuous_rating = None # 连续响应值
        
        print("formal_continuous_rating 的类型是:", type(formal_continuous_rating))
        
        print(trialID)
        
        mouseRec = formal_continuous_rating_mouse.getPos() 
        formal_continuous_rating_mouse.setPos((0, 0))
        has_moved = False
        formal_continuous_image_left.setPos((-0.57, 0))
        formal_continuous_image_left.setSize((0.3, 0.3))
        formal_continuous_image_left.setImage(left_image)
        formal_continuous_image_right.setPos((0.57, 0))
        formal_continuous_image_right.setSize((0.3, 0.3))
        formal_continuous_image_right.setOri(0.0)
        formal_continuous_image_right.setImage(right_image)
        # store start times for formal_continuous_rect
        formal_continuous_rect.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_continuous_rect.tStart = globalClock.getTime(format='float')
        formal_continuous_rect.status = STARTED
        thisExp.addData('formal_continuous_rect.started', formal_continuous_rect.tStart)
        formal_continuous_rect.maxDuration = None
        # keep track of which components have finished
        formal_continuous_rectComponents = formal_continuous_rect.components
        for thisComponent in formal_continuous_rect.components:
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
        
        # --- Run Routine "formal_continuous_rect" ---
        # if trial has changed, end Routine now
        if isinstance(formal_continuous_rating_trials_loop, data.TrialHandler2) and thisFormal_continuous_rating_trials_loop.thisN != formal_continuous_rating_trials_loop.thisTrial.thisN:
            continueRoutine = False
        formal_continuous_rect.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_continuous_rating_marker* updates
            
            # if formal_continuous_rating_marker is starting this frame...
            if formal_continuous_rating_marker.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_continuous_rating_marker.frameNStart = frameN  # exact frame index
                formal_continuous_rating_marker.tStart = t  # local t and not account for scr refresh
                formal_continuous_rating_marker.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_continuous_rating_marker, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_continuous_rating_marker.started')
                # update status
                formal_continuous_rating_marker.status = STARTED
                formal_continuous_rating_marker.setAutoDraw(True)
            
            # if formal_continuous_rating_marker is active this frame...
            if formal_continuous_rating_marker.status == STARTED:
                # update params
                pass
            
            # *formal_continuous_rating_text* updates
            
            # if formal_continuous_rating_text is starting this frame...
            if formal_continuous_rating_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_continuous_rating_text.frameNStart = frameN  # exact frame index
                formal_continuous_rating_text.tStart = t  # local t and not account for scr refresh
                formal_continuous_rating_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_continuous_rating_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_continuous_rating_text.started')
                # update status
                formal_continuous_rating_text.status = STARTED
                formal_continuous_rating_text.setAutoDraw(True)
            
            # if formal_continuous_rating_text is active this frame...
            if formal_continuous_rating_text.status == STARTED:
                # update params
                pass
            
            # *formal_continuous_rating_slider* updates
            
            # if formal_continuous_rating_slider is starting this frame...
            if formal_continuous_rating_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_continuous_rating_slider.frameNStart = frameN  # exact frame index
                formal_continuous_rating_slider.tStart = t  # local t and not account for scr refresh
                formal_continuous_rating_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_continuous_rating_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_continuous_rating_slider.started')
                # update status
                formal_continuous_rating_slider.status = STARTED
                formal_continuous_rating_slider.setAutoDraw(True)
            
            # if formal_continuous_rating_slider is active this frame...
            if formal_continuous_rating_slider.status == STARTED:
                # update params
                pass
            # *formal_continuous_rating_mouse* updates
            
            # if formal_continuous_rating_mouse is starting this frame...
            if formal_continuous_rating_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_continuous_rating_mouse.frameNStart = frameN  # exact frame index
                formal_continuous_rating_mouse.tStart = t  # local t and not account for scr refresh
                formal_continuous_rating_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_continuous_rating_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('formal_continuous_rating_mouse.started', t)
                # update status
                formal_continuous_rating_mouse.status = STARTED
                prevButtonState = formal_continuous_rating_mouse.getPressed()  # if button is down already this ISN'T a new click
            if formal_continuous_rating_mouse.status == STARTED:  # only update if started and not finished!
                buttons = formal_continuous_rating_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(formal_continuous_rating_marker, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(formal_continuous_rating_mouse):
                                gotValidClick = True
                                formal_continuous_rating_mouse.clicked_name.append(obj.name)
                                formal_continuous_rating_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = formal_continuous_rating_mouse.getPos()
                            formal_continuous_rating_mouse.x.append(x)
                            formal_continuous_rating_mouse.y.append(y)
                            buttons = formal_continuous_rating_mouse.getPressed()
                            formal_continuous_rating_mouse.leftButton.append(buttons[0])
                            formal_continuous_rating_mouse.midButton.append(buttons[1])
                            formal_continuous_rating_mouse.rightButton.append(buttons[2])
                            formal_continuous_rating_mouse.time.append(formal_continuous_rating_mouse.mouseClock.getTime())
            # Run 'Each Frame' code from formal_continuous_rating_code
            
            # 实时更新 marker 位置
            mouse_Rec = formal_continuous_rating_mouse.getPos()
            if formal_continuous_rating_marker.contains(formal_continuous_rating_mouse):
               if mouse_Rec[0] != 0:
                    has_moved = True
               formal_continuous_rating_slider.markerPos = mouse_Rec[formal_continuous_rating_slider_orientation] / formal_continuous_rating_slider_width * (formal_continuous_rating_slider_ticks[-1]-formal_continuous_rating_slider_ticks[0]) + (formal_continuous_rating_slider_ticks[0]+formal_continuous_rating_slider_ticks[-1]) / 2
               if formal_continuous_rating_mouse.getPressed()[0]:
                   formal_continuous_rating = round(formal_continuous_rating_slider.markerPos, formal_continuous_rating_slider_decimals)           
                   if formal_continuous_rating_rt is None:  # 仅记录第一次点击的反应时
                        formal_continuous_rating_rt = core.getTime() - start_time
               else:
                   formal_continuous_rating = None
            elif mouse_Rec[0] > 0.25:
                formal_continuous_rating_mouse.setPos((0.25, 0))
            elif mouse_Rec[0] < -0.25:
                formal_continuous_rating_mouse.setPos((-0.25, 0))
            win.mouseVisible = False
            
            if gotValidClick and has_moved:
                continueRoutine = False
            else:
                gotValidClick = False
            
            # *formal_continuous_image_left* updates
            
            # if formal_continuous_image_left is starting this frame...
            if formal_continuous_image_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_continuous_image_left.frameNStart = frameN  # exact frame index
                formal_continuous_image_left.tStart = t  # local t and not account for scr refresh
                formal_continuous_image_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_continuous_image_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_continuous_image_left.started')
                # update status
                formal_continuous_image_left.status = STARTED
                formal_continuous_image_left.setAutoDraw(True)
            
            # if formal_continuous_image_left is active this frame...
            if formal_continuous_image_left.status == STARTED:
                # update params
                pass
            
            # *formal_continuous_image_right* updates
            
            # if formal_continuous_image_right is starting this frame...
            if formal_continuous_image_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_continuous_image_right.frameNStart = frameN  # exact frame index
                formal_continuous_image_right.tStart = t  # local t and not account for scr refresh
                formal_continuous_image_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_continuous_image_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_continuous_image_right.started')
                # update status
                formal_continuous_image_right.status = STARTED
                formal_continuous_image_right.setAutoDraw(True)
            
            # if formal_continuous_image_right is active this frame...
            if formal_continuous_image_right.status == STARTED:
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
                formal_continuous_rect.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_continuous_rect.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_continuous_rect" ---
        for thisComponent in formal_continuous_rect.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_continuous_rect
        formal_continuous_rect.tStop = globalClock.getTime(format='float')
        formal_continuous_rect.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_continuous_rect.stopped', formal_continuous_rect.tStop)
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_slider.response', formal_continuous_rating_slider.getRating())
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_slider.rt', formal_continuous_rating_slider.getRT())
        # store data for formal_continuous_rating_trials_loop (TrialHandler)
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_mouse.x', formal_continuous_rating_mouse.x)
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_mouse.y', formal_continuous_rating_mouse.y)
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_mouse.leftButton', formal_continuous_rating_mouse.leftButton)
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_mouse.midButton', formal_continuous_rating_mouse.midButton)
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_mouse.rightButton', formal_continuous_rating_mouse.rightButton)
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_mouse.time', formal_continuous_rating_mouse.time)
        formal_continuous_rating_trials_loop.addData('formal_continuous_rating_mouse.clicked_name', formal_continuous_rating_mouse.clicked_name)
        # Run 'End Routine' code from formal_continuous_rating_code
        #更新试次
        trialID = trialID + 1
        # 创建新变量side_chosen
        if formal_continuous_rating_mouse.x:  # 检查是否有鼠标数据
            # 取最后一次记录的x坐标
            last_x = formal_continuous_rating_mouse.x[-1]
            
            # 判断x值并赋值
            if last_x > 0:
                side_chosen = 1
            elif last_x < 0:
                side_chosen = 0
            else:
                side_chosen = -1  # 处理x=0的情况
        else:
            side_chosen = None  # 无数据时设为空
        
        #保存数据
        thisExp.addData('formal_continuous_rating_rt', formal_continuous_rating_rt)
        thisExp.addData('formal_continuous_rating', formal_continuous_rating)
        thisExp.addData('trialID', trialID)
        thisExp.addData('blockID', blockID)
        thisExp.addData('Raw_chosen',last_x)#记录原始选择
        thisExp.addData('side_chosen', side_chosen)  # 新增的数据列
        win.mouseVisible = False
        
        # the Routine "formal_continuous_rect" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "rest" ---
        # create an object to store info about Routine rest
        rest = data.Routine(
            name='rest',
            components=[rest_text, rest_key_resp],
        )
        rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for rest_key_resp
        rest_key_resp.keys = []
        rest_key_resp.rt = []
        _rest_key_resp_allKeys = []
        # store start times for rest
        rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest.tStart = globalClock.getTime(format='float')
        rest.status = STARTED
        thisExp.addData('rest.started', rest.tStart)
        rest.maxDuration = None
        # keep track of which components have finished
        restComponents = rest.components
        for thisComponent in rest.components:
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
        
        # --- Run Routine "rest" ---
        # if trial has changed, end Routine now
        if isinstance(formal_continuous_rating_trials_loop, data.TrialHandler2) and thisFormal_continuous_rating_trials_loop.thisN != formal_continuous_rating_trials_loop.thisTrial.thisN:
            continueRoutine = False
        rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_text* updates
            
            # if rest_text is starting this frame...
            if rest_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_text.frameNStart = frameN  # exact frame index
                rest_text.tStart = t  # local t and not account for scr refresh
                rest_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_text.started')
                # update status
                rest_text.status = STARTED
                rest_text.setAutoDraw(True)
            
            # if rest_text is active this frame...
            if rest_text.status == STARTED:
                # update params
                pass
            
            # *rest_key_resp* updates
            waitOnFlip = False
            
            # if rest_key_resp is starting this frame...
            if rest_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_key_resp.frameNStart = frameN  # exact frame index
                rest_key_resp.tStart = t  # local t and not account for scr refresh
                rest_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_key_resp.started')
                # update status
                rest_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(rest_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(rest_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if rest_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = rest_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _rest_key_resp_allKeys.extend(theseKeys)
                if len(_rest_key_resp_allKeys):
                    rest_key_resp.keys = _rest_key_resp_allKeys[-1].name  # just the last key pressed
                    rest_key_resp.rt = _rest_key_resp_allKeys[-1].rt
                    rest_key_resp.duration = _rest_key_resp_allKeys[-1].duration
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
                rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest
        rest.tStop = globalClock.getTime(format='float')
        rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest.stopped', rest.tStop)
        # check responses
        if rest_key_resp.keys in ['', [], None]:  # No response was made
            rest_key_resp.keys = None
        formal_continuous_rating_trials_loop.addData('rest_key_resp.keys',rest_key_resp.keys)
        if rest_key_resp.keys != None:  # we had a response
            formal_continuous_rating_trials_loop.addData('rest_key_resp.rt', rest_key_resp.rt)
            formal_continuous_rating_trials_loop.addData('rest_key_resp.duration', rest_key_resp.duration)
        # the Routine "rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "block" ---
        # create an object to store info about Routine block
        block = data.Routine(
            name='block',
            components=[block_text, block_key_resp],
        )
        block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for block_key_resp
        block_key_resp.keys = []
        block_key_resp.rt = []
        _block_key_resp_allKeys = []
        # Run 'Begin Routine' code from block_code
        print(trialID)
        if trialID in [100,200, 300, 400, 500]:
            continueRoutine = True
        else:
            continueRoutine = False
        print(trialID)
        #if formal_block_trials.thisRepN == 3:  # 如果是第4个block (索引从0开始)
        #   continueRoutine = False  # 直接跳过这个routine，不呈现休息的指导语
         
        if trialID in [100, 200, 300, 400, 500]:
           blockID = blockID + 1
        # store start times for block
        block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block.tStart = globalClock.getTime(format='float')
        block.status = STARTED
        thisExp.addData('block.started', block.tStart)
        block.maxDuration = None
        # keep track of which components have finished
        blockComponents = block.components
        for thisComponent in block.components:
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
        
        # --- Run Routine "block" ---
        # if trial has changed, end Routine now
        if isinstance(formal_continuous_rating_trials_loop, data.TrialHandler2) and thisFormal_continuous_rating_trials_loop.thisN != formal_continuous_rating_trials_loop.thisTrial.thisN:
            continueRoutine = False
        block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_text* updates
            
            # if block_text is starting this frame...
            if block_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_text.frameNStart = frameN  # exact frame index
                block_text.tStart = t  # local t and not account for scr refresh
                block_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_text.started')
                # update status
                block_text.status = STARTED
                block_text.setAutoDraw(True)
            
            # if block_text is active this frame...
            if block_text.status == STARTED:
                # update params
                pass
            
            # *block_key_resp* updates
            waitOnFlip = False
            
            # if block_key_resp is starting this frame...
            if block_key_resp.status == NOT_STARTED and tThisFlip >= 60-frameTolerance:
                # keep track of start time/frame for later
                block_key_resp.frameNStart = frameN  # exact frame index
                block_key_resp.tStart = t  # local t and not account for scr refresh
                block_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_key_resp.started')
                # update status
                block_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(block_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(block_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if block_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = block_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _block_key_resp_allKeys.extend(theseKeys)
                if len(_block_key_resp_allKeys):
                    block_key_resp.keys = _block_key_resp_allKeys[-1].name  # just the last key pressed
                    block_key_resp.rt = _block_key_resp_allKeys[-1].rt
                    block_key_resp.duration = _block_key_resp_allKeys[-1].duration
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
                block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block" ---
        for thisComponent in block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block
        block.tStop = globalClock.getTime(format='float')
        block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block.stopped', block.tStop)
        # check responses
        if block_key_resp.keys in ['', [], None]:  # No response was made
            block_key_resp.keys = None
        formal_continuous_rating_trials_loop.addData('block_key_resp.keys',block_key_resp.keys)
        if block_key_resp.keys != None:  # we had a response
            formal_continuous_rating_trials_loop.addData('block_key_resp.rt', block_key_resp.rt)
            formal_continuous_rating_trials_loop.addData('block_key_resp.duration', block_key_resp.duration)
        # the Routine "block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'formal_continuous_rating_trials_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "formal_continuous_end" ---
    # create an object to store info about Routine formal_continuous_end
    formal_continuous_end = data.Routine(
        name='formal_continuous_end',
        components=[formal_continuous_end_text, formal_continuous_end_key_resp],
    )
    formal_continuous_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for formal_continuous_end_key_resp
    formal_continuous_end_key_resp.keys = []
    formal_continuous_end_key_resp.rt = []
    _formal_continuous_end_key_resp_allKeys = []
    # store start times for formal_continuous_end
    formal_continuous_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    formal_continuous_end.tStart = globalClock.getTime(format='float')
    formal_continuous_end.status = STARTED
    thisExp.addData('formal_continuous_end.started', formal_continuous_end.tStart)
    formal_continuous_end.maxDuration = None
    # keep track of which components have finished
    formal_continuous_endComponents = formal_continuous_end.components
    for thisComponent in formal_continuous_end.components:
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
    
    # --- Run Routine "formal_continuous_end" ---
    formal_continuous_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *formal_continuous_end_text* updates
        
        # if formal_continuous_end_text is starting this frame...
        if formal_continuous_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            formal_continuous_end_text.frameNStart = frameN  # exact frame index
            formal_continuous_end_text.tStart = t  # local t and not account for scr refresh
            formal_continuous_end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(formal_continuous_end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'formal_continuous_end_text.started')
            # update status
            formal_continuous_end_text.status = STARTED
            formal_continuous_end_text.setAutoDraw(True)
        
        # if formal_continuous_end_text is active this frame...
        if formal_continuous_end_text.status == STARTED:
            # update params
            pass
        
        # *formal_continuous_end_key_resp* updates
        waitOnFlip = False
        
        # if formal_continuous_end_key_resp is starting this frame...
        if formal_continuous_end_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            formal_continuous_end_key_resp.frameNStart = frameN  # exact frame index
            formal_continuous_end_key_resp.tStart = t  # local t and not account for scr refresh
            formal_continuous_end_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(formal_continuous_end_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'formal_continuous_end_key_resp.started')
            # update status
            formal_continuous_end_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(formal_continuous_end_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(formal_continuous_end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if formal_continuous_end_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = formal_continuous_end_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _formal_continuous_end_key_resp_allKeys.extend(theseKeys)
            if len(_formal_continuous_end_key_resp_allKeys):
                formal_continuous_end_key_resp.keys = _formal_continuous_end_key_resp_allKeys[-1].name  # just the last key pressed
                formal_continuous_end_key_resp.rt = _formal_continuous_end_key_resp_allKeys[-1].rt
                formal_continuous_end_key_resp.duration = _formal_continuous_end_key_resp_allKeys[-1].duration
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
            formal_continuous_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in formal_continuous_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "formal_continuous_end" ---
    for thisComponent in formal_continuous_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for formal_continuous_end
    formal_continuous_end.tStop = globalClock.getTime(format='float')
    formal_continuous_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('formal_continuous_end.stopped', formal_continuous_end.tStop)
    # check responses
    if formal_continuous_end_key_resp.keys in ['', [], None]:  # No response was made
        formal_continuous_end_key_resp.keys = None
    thisExp.addData('formal_continuous_end_key_resp.keys',formal_continuous_end_key_resp.keys)
    if formal_continuous_end_key_resp.keys != None:  # we had a response
        thisExp.addData('formal_continuous_end_key_resp.rt', formal_continuous_end_key_resp.rt)
        thisExp.addData('formal_continuous_end_key_resp.duration', formal_continuous_end_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "formal_continuous_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)
    # end 'rush' mode
    core.rush(enable=False)


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
