#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 三月 01, 2026, at 18:36
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
expName = 'exp04A'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'sex': 'female/male',
    'age': '18~30',
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
            logging.getLevel('exp')
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
    if deviceManager.getDevice('intro_key_resp') is None:
        # initialise intro_key_resp
        intro_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_key_resp',
        )
    if deviceManager.getDevice('intro_2_key_resp') is None:
        # initialise intro_2_key_resp
        intro_2_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_2_key_resp',
        )
    if deviceManager.getDevice('intro_3_key_resp') is None:
        # initialise intro_3_key_resp
        intro_3_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_3_key_resp',
        )
    if deviceManager.getDevice('intro_4_key_resp') is None:
        # initialise intro_4_key_resp
        intro_4_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_4_key_resp',
        )
    if deviceManager.getDevice('intro_5_key_resp') is None:
        # initialise intro_5_key_resp
        intro_5_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_5_key_resp',
        )
    if deviceManager.getDevice('intro_6_key_resp') is None:
        # initialise intro_6_key_resp
        intro_6_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_6_key_resp',
        )
    if deviceManager.getDevice('intro_7_key_resp') is None:
        # initialise intro_7_key_resp
        intro_7_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_7_key_resp',
        )
    if deviceManager.getDevice('prac_wo_mix_begin_key_resp') is None:
        # initialise prac_wo_mix_begin_key_resp
        prac_wo_mix_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_wo_mix_begin_key_resp',
        )
    if deviceManager.getDevice('prac_wo_end_key_resp') is None:
        # initialise prac_wo_end_key_resp
        prac_wo_end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_wo_end_key_resp',
        )
    if deviceManager.getDevice('wo_mix_begin_key_resp') is None:
        # initialise wo_mix_begin_key_resp
        wo_mix_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='wo_mix_begin_key_resp',
        )
    if deviceManager.getDevice('wo_mix_brea_key_resp') is None:
        # initialise wo_mix_brea_key_resp
        wo_mix_brea_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='wo_mix_brea_key_resp',
        )
    if deviceManager.getDevice('formal_wo_end_key_resp') is None:
        # initialise formal_wo_end_key_resp
        formal_wo_end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='formal_wo_end_key_resp',
        )
    if deviceManager.getDevice('with_intro_1_key_resp') is None:
        # initialise with_intro_1_key_resp
        with_intro_1_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='with_intro_1_key_resp',
        )
    if deviceManager.getDevice('with_intro_2_key_resp') is None:
        # initialise with_intro_2_key_resp
        with_intro_2_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='with_intro_2_key_resp',
        )
    if deviceManager.getDevice('with_intro_3_key_resp') is None:
        # initialise with_intro_3_key_resp
        with_intro_3_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='with_intro_3_key_resp',
        )
    if deviceManager.getDevice('with_intro_4_key_resp') is None:
        # initialise with_intro_4_key_resp
        with_intro_4_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='with_intro_4_key_resp',
        )
    if deviceManager.getDevice('with_intro_5_key_resp') is None:
        # initialise with_intro_5_key_resp
        with_intro_5_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='with_intro_5_key_resp',
        )
    if deviceManager.getDevice('with_intro_6_key_resp') is None:
        # initialise with_intro_6_key_resp
        with_intro_6_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='with_intro_6_key_resp',
        )
    if deviceManager.getDevice('with_intro_7_key_resp') is None:
        # initialise with_intro_7_key_resp
        with_intro_7_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='with_intro_7_key_resp',
        )
    if deviceManager.getDevice('with_intro_8_key_resp') is None:
        # initialise with_intro_8_key_resp
        with_intro_8_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='with_intro_8_key_resp',
        )
    if deviceManager.getDevice('prac_with_begin_key_resp') is None:
        # initialise prac_with_begin_key_resp
        prac_with_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_with_begin_key_resp',
        )
    if deviceManager.getDevice('prac_cate_begin_key_resp') is None:
        # initialise prac_cate_begin_key_resp
        prac_cate_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_cate_begin_key_resp',
        )
    if deviceManager.getDevice('prac_with_begin_2_key_resp') is None:
        # initialise prac_with_begin_2_key_resp
        prac_with_begin_2_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_with_begin_2_key_resp',
        )
    if deviceManager.getDevice('prac_cont_begin_key_resp') is None:
        # initialise prac_cont_begin_key_resp
        prac_cont_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_cont_begin_key_resp',
        )
    if deviceManager.getDevice('prac_end_key_resp') is None:
        # initialise prac_end_key_resp
        prac_end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_end_key_resp',
        )
    if deviceManager.getDevice('formal_cate_begin_key_resp') is None:
        # initialise formal_cate_begin_key_resp
        formal_cate_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='formal_cate_begin_key_resp',
        )
    if deviceManager.getDevice('brea_key_resp') is None:
        # initialise brea_key_resp
        brea_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='brea_key_resp',
        )
    if deviceManager.getDevice('formal_cont_begin_key_resp') is None:
        # initialise formal_cont_begin_key_resp
        formal_cont_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='formal_cont_begin_key_resp',
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
    
    # --- Initialize components for Routine "intro" ---
    intro_text = visual.TextStim(win=win, name='intro_text',
        text='欢迎参与我们的实验！\n在这项研究中，您将看到很多面孔图片，每次一张。\n您的任务是判断每张面孔图片表达的情绪。\n\n按空格键查看更多说明。\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_key_resp = keyboard.Keyboard(deviceName='intro_key_resp')
    # Run 'Begin Experiment' code from intro_code
    win.mouseVisible = False
    
    # --- Initialize components for Routine "intro_2" ---
    intro_2_text = visual.TextStim(win=win, name='intro_2_text',
        text='首先，您会看到一个短暂呈现的面孔图片，例如下图。\n请注意，您可能会看到同一个人出现多次，但不一定是同样的表情，\n您只需要对当前呈现的面孔情绪做出判断。\n',
        font='Open Sans',
        pos=(0, 0.15), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_2_key_resp = keyboard.Keyboard(deviceName='intro_2_key_resp')
    intro_2_image = visual.ImageStim(
        win=win,
        name='intro_2_image', 
        image='stimuli/m_y50f_10.bmp', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "intro_3" ---
    intro_3_text = visual.TextStim(win=win, name='intro_3_text',
        text='接着，您可能会进行2种判断任务中的一种。\n1.分类判断：\n面孔消失后，如果您看到‘恐惧’和‘平静’两个文字（如下所示），\n这表示您需要对刚才呈现的面孔图片进行分类情绪判断，\n即从"恐惧"或"平静"两个选项中任选其一。',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_3_key_resp = keyboard.Keyboard(deviceName='intro_3_key_resp')
    intro_3_left_calm = visual.TextStim(win=win, name='intro_3_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    intro_3_right_fear = visual.TextStim(win=win, name='intro_3_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "intro_4" ---
    intro_4_text = visual.TextStim(win=win, name='intro_4_text',
        text='您需要尽快使用鼠标左键选择一个与面孔相符的情绪类别，\n被选中的情绪词周围会出现白色方框(如下所示)。\n',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_4_key_resp = keyboard.Keyboard(deviceName='intro_4_key_resp')
    intro_4_rect_left = visual.Rect(
        win=win, name='intro_4_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, -0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    intro_4_left_calm = visual.TextStim(win=win, name='intro_4_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_4_right_fear = visual.TextStim(win=win, name='intro_4_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "intro_5" ---
    intro_5_text = visual.TextStim(win=win, name='intro_5_text',
        text='2.连续判断：\n面孔消失后，如果您看到‘恐惧’和‘平静’两个文字且中间有一条横线（如下所示），\n这表示您需要对刚才呈现的面孔图片进行连续情绪判断。\n横线的最左端表示0%的恐惧（即100%的平静），最右端表示100%的恐惧。 \n\n',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_5_key_resp = keyboard.Keyboard(deviceName='intro_5_key_resp')
    intro_5_left_calm = visual.TextStim(win=win, name='intro_5_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    intro_5_right_fear = visual.TextStim(win=win, name='intro_5_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_5_cont_fix_line = visual.Rect(
        win=win, name='intro_5_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, -0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "intro_6" ---
    intro_6_text = visual.TextStim(win=win, name='intro_6_text',
        text='您需要尽快对与面孔相符的情绪程度做出标记，\n在横线区域内移动鼠标（无需单击）即可沿线移动滑块，\n当移动到目标程度的位置时，点击鼠标左键即可完成判断。\n',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_6_key_resp = keyboard.Keyboard(deviceName='intro_6_key_resp')
    intro_6_left_calm = visual.TextStim(win=win, name='intro_6_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    intro_6_right_fear = visual.TextStim(win=win, name='intro_6_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_6_marker = visual.Rect(
        win=win, name='intro_6_marker',
        width=(0.01, 0.05)[0], height=(0.01, 0.05)[1],
        ori=0.0, pos=(0.1, -0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    intro_6_cont_fix_line = visual.Rect(
        win=win, name='intro_6_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, -0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    
    # --- Initialize components for Routine "intro_7" ---
    intro_7_text = visual.TextStim(win=win, name='intro_7_text',
        text='祝贺你！你已经了解了实验程序。\n现在我们来练习一下。您需要对一系列图片进行情绪判断。\n\n准备好按空格键开始。\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_7_key_resp = keyboard.Keyboard(deviceName='intro_7_key_resp')
    
    # --- Initialize components for Routine "prac_wo_mix_begin" ---
    prac_wo_mix_begin_text = visual.TextStim(win=win, name='prac_wo_mix_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_wo_mix_begin_key_resp = keyboard.Keyboard(deviceName='prac_wo_mix_begin_key_resp')
    
    # --- Initialize components for Routine "prac_wo_mix_fix" ---
    prac_wo_mix_fix_text = visual.TextStim(win=win, name='prac_wo_mix_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_wo_mix_face" ---
    prac_wo_mix_face_image = visual.ImageStim(
        win=win,
        name='prac_wo_mix_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "prac_wo_mix_cate_resp" ---
    prac_wo_mix_cate_resp_prompt = visual.TextStim(win=win, name='prac_wo_mix_cate_resp_prompt',
        text='请选择情绪类别',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_wo_mix_cate_resp_rect_left = visual.Rect(
        win=win, name='prac_wo_mix_cate_resp_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    prac_wo_mix_cate_resp_rect_right = visual.Rect(
        win=win, name='prac_wo_mix_cate_resp_rect_right',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(0.3, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    prac_wo_mix_cate_resp_left_calm = visual.TextStim(win=win, name='prac_wo_mix_cate_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_wo_mix_cate_resp_right_fear = visual.TextStim(win=win, name='prac_wo_mix_cate_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    prac_wo_mix_cate_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_wo_mix_cate_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prac_wo_mix_cont_resp" ---
    prac_wo_mix_cont_resp_prompt = visual.TextStim(win=win, name='prac_wo_mix_cont_resp_prompt',
        text='请选择情绪程度',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_wo_mix_cont_resp_slider_shape = visual.Rect(
        win=win, name='prac_wo_mix_cont_resp_slider_shape',
        width=(0.6, 0.07)[0], height=(0.6, 0.07)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    prac_wo_mix_cont_resp_slider = visual.Slider(win=win, name='prac_wo_mix_cont_resp_slider',
        startValue=None, size=(0.5, 0.05), pos=(0, 0), units=win.units,
        labels=None, ticks=(0, 1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor=[1.0000, 1.0000, 1.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    prac_wo_mix_cont_resp_left_calm = visual.TextStim(win=win, name='prac_wo_mix_cont_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_wo_mix_cont_resp_right_fear = visual.TextStim(win=win, name='prac_wo_mix_cont_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    prac_wo_mix_cont_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_wo_mix_cont_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prac_wo_end" ---
    prac_wo_end_text = visual.TextStim(win=win, name='prac_wo_end_text',
        text='练习部分到此为止。\n如果您有任何疑问，请咨询实验员。\n如果您已经准备好，请按键盘方向键右键“>”开始正式实验。\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_wo_end_key_resp = keyboard.Keyboard(deviceName='prac_wo_end_key_resp')
    
    # --- Initialize components for Routine "wo_mix_begin" ---
    wo_mix_begin_text = visual.TextStim(win=win, name='wo_mix_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wo_mix_begin_key_resp = keyboard.Keyboard(deviceName='wo_mix_begin_key_resp')
    
    # --- Initialize components for Routine "wo_mix_fix" ---
    wo_mix_fix_text = visual.TextStim(win=win, name='wo_mix_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from wo_mix_fix_Init_trial
    trialID = 0
    
    
    
    # --- Initialize components for Routine "wo_mix_face" ---
    wo_mix_face_image = visual.ImageStim(
        win=win,
        name='wo_mix_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "wo_mix_cate_resp" ---
    wo_mix_cate_resp_prompt = visual.TextStim(win=win, name='wo_mix_cate_resp_prompt',
        text='请选择情绪类别',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wo_mix_cate_resp_rect_left = visual.Rect(
        win=win, name='wo_mix_cate_resp_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    wo_mix_cate_resp_rect_right = visual.Rect(
        win=win, name='wo_mix_cate_resp_rect_right',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(0.3, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    wo_mix_cate_resp_left_calm = visual.TextStim(win=win, name='wo_mix_cate_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    wo_mix_cate_resp_right_fear = visual.TextStim(win=win, name='wo_mix_cate_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    wo_mix_cate_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    wo_mix_cate_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "wo_mix_cont_resp" ---
    wo_mix_cont_resp_prompt = visual.TextStim(win=win, name='wo_mix_cont_resp_prompt',
        text='请选择情绪程度',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wo_mix_cont_resp_slider_shape = visual.Rect(
        win=win, name='wo_mix_cont_resp_slider_shape',
        width=(0.6, 0.07)[0], height=(0.6, 0.07)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    wo_mix_cont_resp_slider = visual.Slider(win=win, name='wo_mix_cont_resp_slider',
        startValue=None, size=(0.5, 0.05), pos=(0, 0), units=win.units,
        labels=None, ticks=(0, 1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor=[1.0000, 1.0000, 1.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    wo_mix_cont_resp_left_calm = visual.TextStim(win=win, name='wo_mix_cont_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    wo_mix_cont_resp_right_fear = visual.TextStim(win=win, name='wo_mix_cont_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    wo_mix_cont_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    wo_mix_cont_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "wo_mix_brea" ---
    wo_mix_brea_text = visual.TextStim(win=win, name='wo_mix_brea_text',
        text='现在，请您稍事休息，\n当您感觉准备好继续实验时，\n请按下方向键右键“>”继续进行实验任务。\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wo_mix_brea_key_resp = keyboard.Keyboard(deviceName='wo_mix_brea_key_resp')
    
    # --- Initialize components for Routine "formal_wo_end" ---
    formal_wo_end_text = visual.TextStim(win=win, name='formal_wo_end_text',
        text='第一部分的实验到此为止。恭喜你已经完成了整个实验的50%。\n请您稍事休息。\n如果您已经准备好，请按键盘方向键右键“>”继续第二部分的实验。\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_wo_end_key_resp = keyboard.Keyboard(deviceName='formal_wo_end_key_resp')
    
    # --- Initialize components for Routine "with_intro_1" ---
    with_intro_1_text = visual.TextStim(win=win, name='with_intro_1_text',
        text='类似于之前的实验，您将看到很多面孔图片，每次一张。\n您的任务是判断每张面孔图片表达的情绪。\n\n按空格键查看更多说明。\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_1_key_resp = keyboard.Keyboard(deviceName='with_intro_1_key_resp')
    # Run 'Begin Experiment' code from with_intro_1_code
    win.mouseVisible = False
    
    # --- Initialize components for Routine "with_intro_2" ---
    with_intro_2_text = visual.TextStim(win=win, name='with_intro_2_text',
        text='1.分类判断：\n\n如果您看到‘恐惧’和‘平静’两个文字（如下所示），\n这表示您需要对接下来呈现的面孔图片进行分类情绪判断，\n即从"恐惧"或"平静"两个选项中任选其一。',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_2_key_resp = keyboard.Keyboard(deviceName='with_intro_2_key_resp')
    with_intro_2_left_calm = visual.TextStim(win=win, name='with_intro_2_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    with_intro_2_right_fear = visual.TextStim(win=win, name='with_intro_2_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "with_intro_3" ---
    with_intro_3_text = visual.TextStim(win=win, name='with_intro_3_text',
        text='接着，您会看到一个短暂呈现的面孔图片，例如下图。\n请注意，您可能会看到同一个人出现多次，但不一定是同样的表情，\n您只需要对当前呈现的面孔情绪做出判断。\n',
        font='Open Sans',
        pos=(0, 0.15), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_3_key_resp = keyboard.Keyboard(deviceName='with_intro_3_key_resp')
    with_intro_3_image = visual.ImageStim(
        win=win,
        name='with_intro_3_image', 
        image='stimuli/m_y50f_10.bmp', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "with_intro_4" ---
    with_intro_4_text = visual.TextStim(win=win, name='with_intro_4_text',
        text='面孔消失后，请尽快使用鼠标左键选择一个与面孔相符的情绪类别，\n被选中的情绪词周围会出现白色方框(如下所示)。\n\n',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_4_key_resp = keyboard.Keyboard(deviceName='with_intro_4_key_resp')
    with_intro_4_rect_left = visual.Rect(
        win=win, name='with_intro_4_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, -0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    with_intro_4_left_calm = visual.TextStim(win=win, name='with_intro_4_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    with_intro_4_right_fear = visual.TextStim(win=win, name='with_intro_4_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "with_intro_5" ---
    with_intro_5_text = visual.TextStim(win=win, name='with_intro_5_text',
        text='2.连续判断：\n\n如果您看到‘恐惧’和‘平静’两个文字且中间有一条横线（如下所示），\n这表示您需要对接下来呈现的面孔图片进行连续情绪判断。\n横线的最左端表示0%的恐惧（即100%的平静），最右端表示100%的恐惧。 \n\n',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_5_key_resp = keyboard.Keyboard(deviceName='with_intro_5_key_resp')
    with_intro_5_left_calm = visual.TextStim(win=win, name='with_intro_5_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    with_intro_5_right_fear = visual.TextStim(win=win, name='with_intro_5_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    with_intro_5_cont_fix_line = visual.Rect(
        win=win, name='with_intro_5_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, -0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "with_intro_6" ---
    with_intro_6_text = visual.TextStim(win=win, name='with_intro_6_text',
        text='接着，您会看到一个短暂呈现的面孔图片，例如下图。\n请注意，您可能会看到同一个人出现多次，但不一定是同样的表情，\n您只需要对当前呈现的面孔情绪做出判断。\n\n',
        font='Open Sans',
        pos=(0, 0.15), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_6_key_resp = keyboard.Keyboard(deviceName='with_intro_6_key_resp')
    with_intro_6_image = visual.ImageStim(
        win=win,
        name='with_intro_6_image', 
        image='stimuli/m_y28m_70.bmp', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "with_intro_7" ---
    with_intro_7_text = visual.TextStim(win=win, name='with_intro_7_text',
        text='面孔消失后，您需要尽快对与面孔相符的情绪程度做出标记，\n在横线区域内移动鼠标（无需单击）即可沿线移动滑块，\n当移动到目标程度的位置时，点击鼠标左键即可完成判断。\n\n',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_7_key_resp = keyboard.Keyboard(deviceName='with_intro_7_key_resp')
    with_intro_7_left_calm = visual.TextStim(win=win, name='with_intro_7_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    with_intro_7_right_fear = visual.TextStim(win=win, name='with_intro_7_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    with_intro_7_marker = visual.Rect(
        win=win, name='with_intro_7_marker',
        width=(0.01, 0.05)[0], height=(0.01, 0.05)[1],
        ori=0.0, pos=(0.1, -0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    with_intro_7_cont_fix_line = visual.Rect(
        win=win, name='with_intro_7_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, -0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    
    # --- Initialize components for Routine "with_intro_8" ---
    with_intro_8_text = visual.TextStim(win=win, name='with_intro_8_text',
        text='祝贺你！你已经了解了这一部分的实验程序。\n现在我们来练习一下。您需要对一系列图片进行情绪判断。\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_8_key_resp = keyboard.Keyboard(deviceName='with_intro_8_key_resp')
    
    # --- Initialize components for Routine "prac_with_begin" ---
    prac_with_begin_text = visual.TextStim(win=win, name='prac_with_begin_text',
        text='请对接下来呈现的图片进行分类判断，\n您需要在面孔呈现后，尽快用鼠标选择一个与面孔相符的情绪类别。\n\n准备好按空格键开始。',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_with_begin_key_resp = keyboard.Keyboard(deviceName='prac_with_begin_key_resp')
    
    # --- Initialize components for Routine "prac_cate_begin" ---
    prac_cate_begin_text = visual.TextStim(win=win, name='prac_cate_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_cate_begin_key_resp = keyboard.Keyboard(deviceName='prac_cate_begin_key_resp')
    
    # --- Initialize components for Routine "prac_cate_fix" ---
    prac_cate_fix_text = visual.TextStim(win=win, name='prac_cate_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_cate_fix_left_calm = visual.TextStim(win=win, name='prac_cate_fix_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_cate_fix_right_fear = visual.TextStim(win=win, name='prac_cate_fix_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "prac_cate_face" ---
    prac_cate_face_image = visual.ImageStim(
        win=win,
        name='prac_cate_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "prac_cate_resp" ---
    prac_cate_resp_prompt = visual.TextStim(win=win, name='prac_cate_resp_prompt',
        text='请选择情绪类别',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_cate_resp_rect_left = visual.Rect(
        win=win, name='prac_cate_resp_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    prac_cate_resp_rect_right = visual.Rect(
        win=win, name='prac_cate_resp_rect_right',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(0.3, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    prac_cate_resp_left_calm = visual.TextStim(win=win, name='prac_cate_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_cate_resp_right_fear = visual.TextStim(win=win, name='prac_cate_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    prac_cate_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_cate_resp_mouse.mouseClock = core.Clock()
    # Run 'Begin Experiment' code from prac_cate_resp_code
    # 在 'Begin Experiment' 代码框中
    win.mouseVisible = False
    
    
    
    
    
    # --- Initialize components for Routine "prac_with_begin_2" ---
    prac_with_begin_2_text = visual.TextStim(win=win, name='prac_with_begin_2_text',
        text='分类判断练习完成。接下来将进行连续判断的练习。\n面孔消失后，尽快在横线区域内移动鼠标，\n当移动到目标程度位置时，点击鼠标左键完成判断。\n\n准备好后按键盘方向键右键“>”开始。\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_with_begin_2_key_resp = keyboard.Keyboard(deviceName='prac_with_begin_2_key_resp')
    
    # --- Initialize components for Routine "prac_cont_begin" ---
    prac_cont_begin_text = visual.TextStim(win=win, name='prac_cont_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_cont_begin_key_resp = keyboard.Keyboard(deviceName='prac_cont_begin_key_resp')
    
    # --- Initialize components for Routine "prac_cont_fix" ---
    prac_cont_fix_text = visual.TextStim(win=win, name='prac_cont_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_cont_fix_left_calm = visual.TextStim(win=win, name='prac_cont_fix_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_cont_fix_right_fear = visual.TextStim(win=win, name='prac_cont_fix_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_cont_fix_line = visual.Rect(
        win=win, name='prac_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "prac_cont_face" ---
    prac_cont_face_image = visual.ImageStim(
        win=win,
        name='prac_cont_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "prac_cont_resp" ---
    prac_cont_resp_prompt = visual.TextStim(win=win, name='prac_cont_resp_prompt',
        text='请选择情绪程度',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_cont_resp_slider_shape = visual.Rect(
        win=win, name='prac_cont_resp_slider_shape',
        width=(0.6, 0.07)[0], height=(0.6, 0.07)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    prac_cont_resp_slider = visual.Slider(win=win, name='prac_cont_resp_slider',
        startValue=None, size=(0.5, 0.05), pos=(0, 0), units=win.units,
        labels=None, ticks=(0, 1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor=[1.0000, 1.0000, 1.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    prac_cont_resp_left_calm = visual.TextStim(win=win, name='prac_cont_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_cont_resp_right_fear = visual.TextStim(win=win, name='prac_cont_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    prac_cont_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_cont_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prac_end" ---
    prac_end_text = visual.TextStim(win=win, name='prac_end_text',
        text='练习部分到此为止。\n如果您有任何疑问，请咨询实验员。\n如果您已经准备好，请按方向键右键“>”开始正式实验。\n请对接下来呈现的面孔进行分类判断。',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_end_key_resp = keyboard.Keyboard(deviceName='prac_end_key_resp')
    
    # --- Initialize components for Routine "formal_cate_begin" ---
    formal_cate_begin_text = visual.TextStim(win=win, name='formal_cate_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_cate_begin_key_resp = keyboard.Keyboard(deviceName='formal_cate_begin_key_resp')
    
    # --- Initialize components for Routine "formal_cate_fix" ---
    formal_cate_fix_text = visual.TextStim(win=win, name='formal_cate_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_cate_fix_left_calm = visual.TextStim(win=win, name='formal_cate_fix_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    formal_cate_fix_right_fear = visual.TextStim(win=win, name='formal_cate_fix_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    # Run 'Begin Experiment' code from formal_cate_fix_Init_trial
    trialID = 0
    
    # --- Initialize components for Routine "formal_cate_face" ---
    formal_cate_face_image = visual.ImageStim(
        win=win,
        name='formal_cate_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "formal_cate_resp" ---
    formal_cate_resp_prompt = visual.TextStim(win=win, name='formal_cate_resp_prompt',
        text='请选择情绪类别',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_cate_resp_rect_left = visual.Rect(
        win=win, name='formal_cate_resp_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    formal_cate_resp_rect_right = visual.Rect(
        win=win, name='formal_cate_resp_rect_right',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(0.3, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    formal_cate_resp_left_calm = visual.TextStim(win=win, name='formal_cate_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    formal_cate_resp_right_fear = visual.TextStim(win=win, name='formal_cate_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    formal_cate_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    formal_cate_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "brea" ---
    brea_text = visual.TextStim(win=win, name='brea_text',
        text='现在，请您稍事休息，\n当您感觉准备好继续实验时，\n请按下方向键右键“>”继续进行实验任务。\n请对接下来呈现的面孔进行连续判断。',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    brea_key_resp = keyboard.Keyboard(deviceName='brea_key_resp')
    
    # --- Initialize components for Routine "formal_cont_begin" ---
    formal_cont_begin_text = visual.TextStim(win=win, name='formal_cont_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_cont_begin_key_resp = keyboard.Keyboard(deviceName='formal_cont_begin_key_resp')
    
    # --- Initialize components for Routine "formal_cont_fix" ---
    formal_cont_fix_text = visual.TextStim(win=win, name='formal_cont_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_cont_fix_left_calm = visual.TextStim(win=win, name='formal_cont_fix_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    formal_cont_fix_right_fear = visual.TextStim(win=win, name='formal_cont_fix_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    formal_cont_fix_line = visual.Rect(
        win=win, name='formal_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    # Run 'Begin Experiment' code from formal_cont_fix_Init_trial
    trialID = 0
    
    # --- Initialize components for Routine "formal_cont_face" ---
    formal_cont_face_image = visual.ImageStim(
        win=win,
        name='formal_cont_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "formal_cont_resp" ---
    formal_cont_resp_prompt = visual.TextStim(win=win, name='formal_cont_resp_prompt',
        text='请选择情绪程度',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_cont_resp_slider_shape = visual.Rect(
        win=win, name='formal_cont_resp_slider_shape',
        width=(0.6, 0.07)[0], height=(0.6, 0.07)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    formal_cont_resp_slider = visual.Slider(win=win, name='formal_cont_resp_slider',
        startValue=None, size=(0.5, 0.05), pos=(0, 0), units=win.units,
        labels=None, ticks=(0, 1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor=[1.0000, 1.0000, 1.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    formal_cont_resp_left_calm = visual.TextStim(win=win, name='formal_cont_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    formal_cont_resp_right_fear = visual.TextStim(win=win, name='formal_cont_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    formal_cont_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    formal_cont_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "end_exp" ---
    end_exp_text = visual.TextStim(win=win, name='end_exp_text',
        text='实验任务已全部完成，感谢您的参与！',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
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
    
    # --- Prepare to start Routine "intro" ---
    # create an object to store info about Routine intro
    intro = data.Routine(
        name='intro',
        components=[intro_text, intro_key_resp],
    )
    intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_key_resp
    intro_key_resp.keys = []
    intro_key_resp.rt = []
    _intro_key_resp_allKeys = []
    # store start times for intro
    intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro.tStart = globalClock.getTime(format='float')
    intro.status = STARTED
    thisExp.addData('intro.started', intro.tStart)
    intro.maxDuration = None
    # keep track of which components have finished
    introComponents = intro.components
    for thisComponent in intro.components:
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
    
    # --- Run Routine "intro" ---
    intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_text* updates
        
        # if intro_text is starting this frame...
        if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text.frameNStart = frameN  # exact frame index
            intro_text.tStart = t  # local t and not account for scr refresh
            intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_text.started')
            # update status
            intro_text.status = STARTED
            intro_text.setAutoDraw(True)
        
        # if intro_text is active this frame...
        if intro_text.status == STARTED:
            # update params
            pass
        
        # *intro_key_resp* updates
        waitOnFlip = False
        
        # if intro_key_resp is starting this frame...
        if intro_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_key_resp.frameNStart = frameN  # exact frame index
            intro_key_resp.tStart = t  # local t and not account for scr refresh
            intro_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_key_resp.started')
            # update status
            intro_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_key_resp_allKeys.extend(theseKeys)
            if len(_intro_key_resp_allKeys):
                intro_key_resp.keys = _intro_key_resp_allKeys[-1].name  # just the last key pressed
                intro_key_resp.rt = _intro_key_resp_allKeys[-1].rt
                intro_key_resp.duration = _intro_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # Run 'Each Frame' code from intro_code
        win.mouseVisible = False
        
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
            intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro
    intro.tStop = globalClock.getTime(format='float')
    intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro.stopped', intro.tStop)
    # check responses
    if intro_key_resp.keys in ['', [], None]:  # No response was made
        intro_key_resp.keys = None
    thisExp.addData('intro_key_resp.keys',intro_key_resp.keys)
    if intro_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_key_resp.rt', intro_key_resp.rt)
        thisExp.addData('intro_key_resp.duration', intro_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_2" ---
    # create an object to store info about Routine intro_2
    intro_2 = data.Routine(
        name='intro_2',
        components=[intro_2_text, intro_2_key_resp, intro_2_image],
    )
    intro_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_2_key_resp
    intro_2_key_resp.keys = []
    intro_2_key_resp.rt = []
    _intro_2_key_resp_allKeys = []
    # store start times for intro_2
    intro_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_2.tStart = globalClock.getTime(format='float')
    intro_2.status = STARTED
    thisExp.addData('intro_2.started', intro_2.tStart)
    intro_2.maxDuration = None
    # keep track of which components have finished
    intro_2Components = intro_2.components
    for thisComponent in intro_2.components:
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
    
    # --- Run Routine "intro_2" ---
    intro_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_2_text* updates
        
        # if intro_2_text is starting this frame...
        if intro_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_2_text.frameNStart = frameN  # exact frame index
            intro_2_text.tStart = t  # local t and not account for scr refresh
            intro_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_2_text.started')
            # update status
            intro_2_text.status = STARTED
            intro_2_text.setAutoDraw(True)
        
        # if intro_2_text is active this frame...
        if intro_2_text.status == STARTED:
            # update params
            pass
        
        # *intro_2_key_resp* updates
        waitOnFlip = False
        
        # if intro_2_key_resp is starting this frame...
        if intro_2_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_2_key_resp.frameNStart = frameN  # exact frame index
            intro_2_key_resp.tStart = t  # local t and not account for scr refresh
            intro_2_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_2_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_2_key_resp.started')
            # update status
            intro_2_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_2_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_2_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_2_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_2_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_2_key_resp_allKeys.extend(theseKeys)
            if len(_intro_2_key_resp_allKeys):
                intro_2_key_resp.keys = _intro_2_key_resp_allKeys[-1].name  # just the last key pressed
                intro_2_key_resp.rt = _intro_2_key_resp_allKeys[-1].rt
                intro_2_key_resp.duration = _intro_2_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *intro_2_image* updates
        
        # if intro_2_image is starting this frame...
        if intro_2_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_2_image.frameNStart = frameN  # exact frame index
            intro_2_image.tStart = t  # local t and not account for scr refresh
            intro_2_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_2_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_2_image.started')
            # update status
            intro_2_image.status = STARTED
            intro_2_image.setAutoDraw(True)
        
        # if intro_2_image is active this frame...
        if intro_2_image.status == STARTED:
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
            intro_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_2" ---
    for thisComponent in intro_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_2
    intro_2.tStop = globalClock.getTime(format='float')
    intro_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_2.stopped', intro_2.tStop)
    # check responses
    if intro_2_key_resp.keys in ['', [], None]:  # No response was made
        intro_2_key_resp.keys = None
    thisExp.addData('intro_2_key_resp.keys',intro_2_key_resp.keys)
    if intro_2_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_2_key_resp.rt', intro_2_key_resp.rt)
        thisExp.addData('intro_2_key_resp.duration', intro_2_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_3" ---
    # create an object to store info about Routine intro_3
    intro_3 = data.Routine(
        name='intro_3',
        components=[intro_3_text, intro_3_key_resp, intro_3_left_calm, intro_3_right_fear],
    )
    intro_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_3_key_resp
    intro_3_key_resp.keys = []
    intro_3_key_resp.rt = []
    _intro_3_key_resp_allKeys = []
    # store start times for intro_3
    intro_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_3.tStart = globalClock.getTime(format='float')
    intro_3.status = STARTED
    thisExp.addData('intro_3.started', intro_3.tStart)
    intro_3.maxDuration = None
    # keep track of which components have finished
    intro_3Components = intro_3.components
    for thisComponent in intro_3.components:
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
    
    # --- Run Routine "intro_3" ---
    intro_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_3_text* updates
        
        # if intro_3_text is starting this frame...
        if intro_3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_3_text.frameNStart = frameN  # exact frame index
            intro_3_text.tStart = t  # local t and not account for scr refresh
            intro_3_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_3_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_3_text.started')
            # update status
            intro_3_text.status = STARTED
            intro_3_text.setAutoDraw(True)
        
        # if intro_3_text is active this frame...
        if intro_3_text.status == STARTED:
            # update params
            pass
        
        # *intro_3_key_resp* updates
        waitOnFlip = False
        
        # if intro_3_key_resp is starting this frame...
        if intro_3_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_3_key_resp.frameNStart = frameN  # exact frame index
            intro_3_key_resp.tStart = t  # local t and not account for scr refresh
            intro_3_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_3_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_3_key_resp.started')
            # update status
            intro_3_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_3_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_3_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_3_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_3_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_3_key_resp_allKeys.extend(theseKeys)
            if len(_intro_3_key_resp_allKeys):
                intro_3_key_resp.keys = _intro_3_key_resp_allKeys[-1].name  # just the last key pressed
                intro_3_key_resp.rt = _intro_3_key_resp_allKeys[-1].rt
                intro_3_key_resp.duration = _intro_3_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *intro_3_left_calm* updates
        
        # if intro_3_left_calm is starting this frame...
        if intro_3_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_3_left_calm.frameNStart = frameN  # exact frame index
            intro_3_left_calm.tStart = t  # local t and not account for scr refresh
            intro_3_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_3_left_calm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_3_left_calm.started')
            # update status
            intro_3_left_calm.status = STARTED
            intro_3_left_calm.setAutoDraw(True)
        
        # if intro_3_left_calm is active this frame...
        if intro_3_left_calm.status == STARTED:
            # update params
            pass
        
        # *intro_3_right_fear* updates
        
        # if intro_3_right_fear is starting this frame...
        if intro_3_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_3_right_fear.frameNStart = frameN  # exact frame index
            intro_3_right_fear.tStart = t  # local t and not account for scr refresh
            intro_3_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_3_right_fear, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_3_right_fear.started')
            # update status
            intro_3_right_fear.status = STARTED
            intro_3_right_fear.setAutoDraw(True)
        
        # if intro_3_right_fear is active this frame...
        if intro_3_right_fear.status == STARTED:
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
            intro_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_3" ---
    for thisComponent in intro_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_3
    intro_3.tStop = globalClock.getTime(format='float')
    intro_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_3.stopped', intro_3.tStop)
    # check responses
    if intro_3_key_resp.keys in ['', [], None]:  # No response was made
        intro_3_key_resp.keys = None
    thisExp.addData('intro_3_key_resp.keys',intro_3_key_resp.keys)
    if intro_3_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_3_key_resp.rt', intro_3_key_resp.rt)
        thisExp.addData('intro_3_key_resp.duration', intro_3_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_4" ---
    # create an object to store info about Routine intro_4
    intro_4 = data.Routine(
        name='intro_4',
        components=[intro_4_text, intro_4_key_resp, intro_4_rect_left, intro_4_left_calm, intro_4_right_fear],
    )
    intro_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_4_key_resp
    intro_4_key_resp.keys = []
    intro_4_key_resp.rt = []
    _intro_4_key_resp_allKeys = []
    # store start times for intro_4
    intro_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_4.tStart = globalClock.getTime(format='float')
    intro_4.status = STARTED
    thisExp.addData('intro_4.started', intro_4.tStart)
    intro_4.maxDuration = None
    # keep track of which components have finished
    intro_4Components = intro_4.components
    for thisComponent in intro_4.components:
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
    
    # --- Run Routine "intro_4" ---
    intro_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_4_text* updates
        
        # if intro_4_text is starting this frame...
        if intro_4_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_4_text.frameNStart = frameN  # exact frame index
            intro_4_text.tStart = t  # local t and not account for scr refresh
            intro_4_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_4_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_4_text.started')
            # update status
            intro_4_text.status = STARTED
            intro_4_text.setAutoDraw(True)
        
        # if intro_4_text is active this frame...
        if intro_4_text.status == STARTED:
            # update params
            pass
        
        # *intro_4_key_resp* updates
        waitOnFlip = False
        
        # if intro_4_key_resp is starting this frame...
        if intro_4_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_4_key_resp.frameNStart = frameN  # exact frame index
            intro_4_key_resp.tStart = t  # local t and not account for scr refresh
            intro_4_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_4_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_4_key_resp.started')
            # update status
            intro_4_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_4_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_4_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_4_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_4_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_4_key_resp_allKeys.extend(theseKeys)
            if len(_intro_4_key_resp_allKeys):
                intro_4_key_resp.keys = _intro_4_key_resp_allKeys[-1].name  # just the last key pressed
                intro_4_key_resp.rt = _intro_4_key_resp_allKeys[-1].rt
                intro_4_key_resp.duration = _intro_4_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *intro_4_rect_left* updates
        
        # if intro_4_rect_left is starting this frame...
        if intro_4_rect_left.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_4_rect_left.frameNStart = frameN  # exact frame index
            intro_4_rect_left.tStart = t  # local t and not account for scr refresh
            intro_4_rect_left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_4_rect_left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_4_rect_left.started')
            # update status
            intro_4_rect_left.status = STARTED
            intro_4_rect_left.setAutoDraw(True)
        
        # if intro_4_rect_left is active this frame...
        if intro_4_rect_left.status == STARTED:
            # update params
            pass
        
        # *intro_4_left_calm* updates
        
        # if intro_4_left_calm is starting this frame...
        if intro_4_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_4_left_calm.frameNStart = frameN  # exact frame index
            intro_4_left_calm.tStart = t  # local t and not account for scr refresh
            intro_4_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_4_left_calm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_4_left_calm.started')
            # update status
            intro_4_left_calm.status = STARTED
            intro_4_left_calm.setAutoDraw(True)
        
        # if intro_4_left_calm is active this frame...
        if intro_4_left_calm.status == STARTED:
            # update params
            pass
        
        # *intro_4_right_fear* updates
        
        # if intro_4_right_fear is starting this frame...
        if intro_4_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_4_right_fear.frameNStart = frameN  # exact frame index
            intro_4_right_fear.tStart = t  # local t and not account for scr refresh
            intro_4_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_4_right_fear, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_4_right_fear.started')
            # update status
            intro_4_right_fear.status = STARTED
            intro_4_right_fear.setAutoDraw(True)
        
        # if intro_4_right_fear is active this frame...
        if intro_4_right_fear.status == STARTED:
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
            intro_4.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_4" ---
    for thisComponent in intro_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_4
    intro_4.tStop = globalClock.getTime(format='float')
    intro_4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_4.stopped', intro_4.tStop)
    # check responses
    if intro_4_key_resp.keys in ['', [], None]:  # No response was made
        intro_4_key_resp.keys = None
    thisExp.addData('intro_4_key_resp.keys',intro_4_key_resp.keys)
    if intro_4_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_4_key_resp.rt', intro_4_key_resp.rt)
        thisExp.addData('intro_4_key_resp.duration', intro_4_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_5" ---
    # create an object to store info about Routine intro_5
    intro_5 = data.Routine(
        name='intro_5',
        components=[intro_5_text, intro_5_key_resp, intro_5_left_calm, intro_5_right_fear, intro_5_cont_fix_line],
    )
    intro_5.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_5_key_resp
    intro_5_key_resp.keys = []
    intro_5_key_resp.rt = []
    _intro_5_key_resp_allKeys = []
    # store start times for intro_5
    intro_5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_5.tStart = globalClock.getTime(format='float')
    intro_5.status = STARTED
    thisExp.addData('intro_5.started', intro_5.tStart)
    intro_5.maxDuration = None
    # keep track of which components have finished
    intro_5Components = intro_5.components
    for thisComponent in intro_5.components:
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
    
    # --- Run Routine "intro_5" ---
    intro_5.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_5_text* updates
        
        # if intro_5_text is starting this frame...
        if intro_5_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_5_text.frameNStart = frameN  # exact frame index
            intro_5_text.tStart = t  # local t and not account for scr refresh
            intro_5_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_5_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_5_text.started')
            # update status
            intro_5_text.status = STARTED
            intro_5_text.setAutoDraw(True)
        
        # if intro_5_text is active this frame...
        if intro_5_text.status == STARTED:
            # update params
            pass
        
        # *intro_5_key_resp* updates
        waitOnFlip = False
        
        # if intro_5_key_resp is starting this frame...
        if intro_5_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_5_key_resp.frameNStart = frameN  # exact frame index
            intro_5_key_resp.tStart = t  # local t and not account for scr refresh
            intro_5_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_5_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_5_key_resp.started')
            # update status
            intro_5_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_5_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_5_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_5_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_5_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_5_key_resp_allKeys.extend(theseKeys)
            if len(_intro_5_key_resp_allKeys):
                intro_5_key_resp.keys = _intro_5_key_resp_allKeys[-1].name  # just the last key pressed
                intro_5_key_resp.rt = _intro_5_key_resp_allKeys[-1].rt
                intro_5_key_resp.duration = _intro_5_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *intro_5_left_calm* updates
        
        # if intro_5_left_calm is starting this frame...
        if intro_5_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_5_left_calm.frameNStart = frameN  # exact frame index
            intro_5_left_calm.tStart = t  # local t and not account for scr refresh
            intro_5_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_5_left_calm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_5_left_calm.started')
            # update status
            intro_5_left_calm.status = STARTED
            intro_5_left_calm.setAutoDraw(True)
        
        # if intro_5_left_calm is active this frame...
        if intro_5_left_calm.status == STARTED:
            # update params
            pass
        
        # *intro_5_right_fear* updates
        
        # if intro_5_right_fear is starting this frame...
        if intro_5_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_5_right_fear.frameNStart = frameN  # exact frame index
            intro_5_right_fear.tStart = t  # local t and not account for scr refresh
            intro_5_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_5_right_fear, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_5_right_fear.started')
            # update status
            intro_5_right_fear.status = STARTED
            intro_5_right_fear.setAutoDraw(True)
        
        # if intro_5_right_fear is active this frame...
        if intro_5_right_fear.status == STARTED:
            # update params
            pass
        
        # *intro_5_cont_fix_line* updates
        
        # if intro_5_cont_fix_line is starting this frame...
        if intro_5_cont_fix_line.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_5_cont_fix_line.frameNStart = frameN  # exact frame index
            intro_5_cont_fix_line.tStart = t  # local t and not account for scr refresh
            intro_5_cont_fix_line.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_5_cont_fix_line, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_5_cont_fix_line.started')
            # update status
            intro_5_cont_fix_line.status = STARTED
            intro_5_cont_fix_line.setAutoDraw(True)
        
        # if intro_5_cont_fix_line is active this frame...
        if intro_5_cont_fix_line.status == STARTED:
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
            intro_5.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_5.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_5" ---
    for thisComponent in intro_5.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_5
    intro_5.tStop = globalClock.getTime(format='float')
    intro_5.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_5.stopped', intro_5.tStop)
    # check responses
    if intro_5_key_resp.keys in ['', [], None]:  # No response was made
        intro_5_key_resp.keys = None
    thisExp.addData('intro_5_key_resp.keys',intro_5_key_resp.keys)
    if intro_5_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_5_key_resp.rt', intro_5_key_resp.rt)
        thisExp.addData('intro_5_key_resp.duration', intro_5_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro_5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_6" ---
    # create an object to store info about Routine intro_6
    intro_6 = data.Routine(
        name='intro_6',
        components=[intro_6_text, intro_6_key_resp, intro_6_left_calm, intro_6_right_fear, intro_6_marker, intro_6_cont_fix_line],
    )
    intro_6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_6_key_resp
    intro_6_key_resp.keys = []
    intro_6_key_resp.rt = []
    _intro_6_key_resp_allKeys = []
    # store start times for intro_6
    intro_6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_6.tStart = globalClock.getTime(format='float')
    intro_6.status = STARTED
    thisExp.addData('intro_6.started', intro_6.tStart)
    intro_6.maxDuration = None
    # keep track of which components have finished
    intro_6Components = intro_6.components
    for thisComponent in intro_6.components:
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
    
    # --- Run Routine "intro_6" ---
    intro_6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_6_text* updates
        
        # if intro_6_text is starting this frame...
        if intro_6_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_6_text.frameNStart = frameN  # exact frame index
            intro_6_text.tStart = t  # local t and not account for scr refresh
            intro_6_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_6_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_6_text.started')
            # update status
            intro_6_text.status = STARTED
            intro_6_text.setAutoDraw(True)
        
        # if intro_6_text is active this frame...
        if intro_6_text.status == STARTED:
            # update params
            pass
        
        # *intro_6_key_resp* updates
        waitOnFlip = False
        
        # if intro_6_key_resp is starting this frame...
        if intro_6_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_6_key_resp.frameNStart = frameN  # exact frame index
            intro_6_key_resp.tStart = t  # local t and not account for scr refresh
            intro_6_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_6_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_6_key_resp.started')
            # update status
            intro_6_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_6_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_6_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_6_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_6_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_6_key_resp_allKeys.extend(theseKeys)
            if len(_intro_6_key_resp_allKeys):
                intro_6_key_resp.keys = _intro_6_key_resp_allKeys[-1].name  # just the last key pressed
                intro_6_key_resp.rt = _intro_6_key_resp_allKeys[-1].rt
                intro_6_key_resp.duration = _intro_6_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *intro_6_left_calm* updates
        
        # if intro_6_left_calm is starting this frame...
        if intro_6_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_6_left_calm.frameNStart = frameN  # exact frame index
            intro_6_left_calm.tStart = t  # local t and not account for scr refresh
            intro_6_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_6_left_calm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_6_left_calm.started')
            # update status
            intro_6_left_calm.status = STARTED
            intro_6_left_calm.setAutoDraw(True)
        
        # if intro_6_left_calm is active this frame...
        if intro_6_left_calm.status == STARTED:
            # update params
            pass
        
        # *intro_6_right_fear* updates
        
        # if intro_6_right_fear is starting this frame...
        if intro_6_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_6_right_fear.frameNStart = frameN  # exact frame index
            intro_6_right_fear.tStart = t  # local t and not account for scr refresh
            intro_6_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_6_right_fear, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_6_right_fear.started')
            # update status
            intro_6_right_fear.status = STARTED
            intro_6_right_fear.setAutoDraw(True)
        
        # if intro_6_right_fear is active this frame...
        if intro_6_right_fear.status == STARTED:
            # update params
            pass
        
        # *intro_6_marker* updates
        
        # if intro_6_marker is starting this frame...
        if intro_6_marker.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_6_marker.frameNStart = frameN  # exact frame index
            intro_6_marker.tStart = t  # local t and not account for scr refresh
            intro_6_marker.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_6_marker, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_6_marker.started')
            # update status
            intro_6_marker.status = STARTED
            intro_6_marker.setAutoDraw(True)
        
        # if intro_6_marker is active this frame...
        if intro_6_marker.status == STARTED:
            # update params
            pass
        
        # *intro_6_cont_fix_line* updates
        
        # if intro_6_cont_fix_line is starting this frame...
        if intro_6_cont_fix_line.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_6_cont_fix_line.frameNStart = frameN  # exact frame index
            intro_6_cont_fix_line.tStart = t  # local t and not account for scr refresh
            intro_6_cont_fix_line.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_6_cont_fix_line, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_6_cont_fix_line.started')
            # update status
            intro_6_cont_fix_line.status = STARTED
            intro_6_cont_fix_line.setAutoDraw(True)
        
        # if intro_6_cont_fix_line is active this frame...
        if intro_6_cont_fix_line.status == STARTED:
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
            intro_6.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_6" ---
    for thisComponent in intro_6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_6
    intro_6.tStop = globalClock.getTime(format='float')
    intro_6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_6.stopped', intro_6.tStop)
    # check responses
    if intro_6_key_resp.keys in ['', [], None]:  # No response was made
        intro_6_key_resp.keys = None
    thisExp.addData('intro_6_key_resp.keys',intro_6_key_resp.keys)
    if intro_6_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_6_key_resp.rt', intro_6_key_resp.rt)
        thisExp.addData('intro_6_key_resp.duration', intro_6_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro_6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_7" ---
    # create an object to store info about Routine intro_7
    intro_7 = data.Routine(
        name='intro_7',
        components=[intro_7_text, intro_7_key_resp],
    )
    intro_7.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_7_key_resp
    intro_7_key_resp.keys = []
    intro_7_key_resp.rt = []
    _intro_7_key_resp_allKeys = []
    # store start times for intro_7
    intro_7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_7.tStart = globalClock.getTime(format='float')
    intro_7.status = STARTED
    thisExp.addData('intro_7.started', intro_7.tStart)
    intro_7.maxDuration = None
    # keep track of which components have finished
    intro_7Components = intro_7.components
    for thisComponent in intro_7.components:
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
    
    # --- Run Routine "intro_7" ---
    intro_7.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_7_text* updates
        
        # if intro_7_text is starting this frame...
        if intro_7_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_7_text.frameNStart = frameN  # exact frame index
            intro_7_text.tStart = t  # local t and not account for scr refresh
            intro_7_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_7_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_7_text.started')
            # update status
            intro_7_text.status = STARTED
            intro_7_text.setAutoDraw(True)
        
        # if intro_7_text is active this frame...
        if intro_7_text.status == STARTED:
            # update params
            pass
        
        # *intro_7_key_resp* updates
        waitOnFlip = False
        
        # if intro_7_key_resp is starting this frame...
        if intro_7_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_7_key_resp.frameNStart = frameN  # exact frame index
            intro_7_key_resp.tStart = t  # local t and not account for scr refresh
            intro_7_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_7_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_7_key_resp.started')
            # update status
            intro_7_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_7_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_7_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_7_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_7_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_7_key_resp_allKeys.extend(theseKeys)
            if len(_intro_7_key_resp_allKeys):
                intro_7_key_resp.keys = _intro_7_key_resp_allKeys[-1].name  # just the last key pressed
                intro_7_key_resp.rt = _intro_7_key_resp_allKeys[-1].rt
                intro_7_key_resp.duration = _intro_7_key_resp_allKeys[-1].duration
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
            intro_7.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_7.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_7" ---
    for thisComponent in intro_7.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_7
    intro_7.tStop = globalClock.getTime(format='float')
    intro_7.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_7.stopped', intro_7.tStop)
    # check responses
    if intro_7_key_resp.keys in ['', [], None]:  # No response was made
        intro_7_key_resp.keys = None
    thisExp.addData('intro_7_key_resp.keys',intro_7_key_resp.keys)
    if intro_7_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_7_key_resp.rt', intro_7_key_resp.rt)
        thisExp.addData('intro_7_key_resp.duration', intro_7_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro_7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    prac_wo_mix = data.TrialHandler2(
        name='prac_wo_mix',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'pics - 4.5.xlsx', 
        selection='0'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(prac_wo_mix)  # add the loop to the experiment
    thisPrac_wo_mix = prac_wo_mix.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPrac_wo_mix.rgb)
    if thisPrac_wo_mix != None:
        for paramName in thisPrac_wo_mix:
            globals()[paramName] = thisPrac_wo_mix[paramName]
    
    for thisPrac_wo_mix in prac_wo_mix:
        currentLoop = prac_wo_mix
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_wo_mix.rgb)
        if thisPrac_wo_mix != None:
            for paramName in thisPrac_wo_mix:
                globals()[paramName] = thisPrac_wo_mix[paramName]
        
        # --- Prepare to start Routine "prac_wo_mix_begin" ---
        # create an object to store info about Routine prac_wo_mix_begin
        prac_wo_mix_begin = data.Routine(
            name='prac_wo_mix_begin',
            components=[prac_wo_mix_begin_text, prac_wo_mix_begin_key_resp],
        )
        prac_wo_mix_begin.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for prac_wo_mix_begin_key_resp
        prac_wo_mix_begin_key_resp.keys = []
        prac_wo_mix_begin_key_resp.rt = []
        _prac_wo_mix_begin_key_resp_allKeys = []
        # store start times for prac_wo_mix_begin
        prac_wo_mix_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_wo_mix_begin.tStart = globalClock.getTime(format='float')
        prac_wo_mix_begin.status = STARTED
        thisExp.addData('prac_wo_mix_begin.started', prac_wo_mix_begin.tStart)
        prac_wo_mix_begin.maxDuration = None
        # keep track of which components have finished
        prac_wo_mix_beginComponents = prac_wo_mix_begin.components
        for thisComponent in prac_wo_mix_begin.components:
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
        
        # --- Run Routine "prac_wo_mix_begin" ---
        # if trial has changed, end Routine now
        if isinstance(prac_wo_mix, data.TrialHandler2) and thisPrac_wo_mix.thisN != prac_wo_mix.thisTrial.thisN:
            continueRoutine = False
        prac_wo_mix_begin.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_wo_mix_begin_text* updates
            
            # if prac_wo_mix_begin_text is starting this frame...
            if prac_wo_mix_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_begin_text.frameNStart = frameN  # exact frame index
                prac_wo_mix_begin_text.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_begin_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_begin_text.started')
                # update status
                prac_wo_mix_begin_text.status = STARTED
                prac_wo_mix_begin_text.setAutoDraw(True)
            
            # if prac_wo_mix_begin_text is active this frame...
            if prac_wo_mix_begin_text.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_begin_key_resp* updates
            waitOnFlip = False
            
            # if prac_wo_mix_begin_key_resp is starting this frame...
            if prac_wo_mix_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_begin_key_resp.frameNStart = frameN  # exact frame index
                prac_wo_mix_begin_key_resp.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_begin_key_resp.started')
                # update status
                prac_wo_mix_begin_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_wo_mix_begin_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_wo_mix_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_wo_mix_begin_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_wo_mix_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _prac_wo_mix_begin_key_resp_allKeys.extend(theseKeys)
                if len(_prac_wo_mix_begin_key_resp_allKeys):
                    prac_wo_mix_begin_key_resp.keys = _prac_wo_mix_begin_key_resp_allKeys[-1].name  # just the last key pressed
                    prac_wo_mix_begin_key_resp.rt = _prac_wo_mix_begin_key_resp_allKeys[-1].rt
                    prac_wo_mix_begin_key_resp.duration = _prac_wo_mix_begin_key_resp_allKeys[-1].duration
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
                prac_wo_mix_begin.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_begin.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_begin" ---
        for thisComponent in prac_wo_mix_begin.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_wo_mix_begin
        prac_wo_mix_begin.tStop = globalClock.getTime(format='float')
        prac_wo_mix_begin.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_wo_mix_begin.stopped', prac_wo_mix_begin.tStop)
        # check responses
        if prac_wo_mix_begin_key_resp.keys in ['', [], None]:  # No response was made
            prac_wo_mix_begin_key_resp.keys = None
        prac_wo_mix.addData('prac_wo_mix_begin_key_resp.keys',prac_wo_mix_begin_key_resp.keys)
        if prac_wo_mix_begin_key_resp.keys != None:  # we had a response
            prac_wo_mix.addData('prac_wo_mix_begin_key_resp.rt', prac_wo_mix_begin_key_resp.rt)
            prac_wo_mix.addData('prac_wo_mix_begin_key_resp.duration', prac_wo_mix_begin_key_resp.duration)
        # the Routine "prac_wo_mix_begin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_wo_mix_fix" ---
        # create an object to store info about Routine prac_wo_mix_fix
        prac_wo_mix_fix = data.Routine(
            name='prac_wo_mix_fix',
            components=[prac_wo_mix_fix_text],
        )
        prac_wo_mix_fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for prac_wo_mix_fix
        prac_wo_mix_fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_wo_mix_fix.tStart = globalClock.getTime(format='float')
        prac_wo_mix_fix.status = STARTED
        thisExp.addData('prac_wo_mix_fix.started', prac_wo_mix_fix.tStart)
        prac_wo_mix_fix.maxDuration = None
        # keep track of which components have finished
        prac_wo_mix_fixComponents = prac_wo_mix_fix.components
        for thisComponent in prac_wo_mix_fix.components:
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
        
        # --- Run Routine "prac_wo_mix_fix" ---
        # if trial has changed, end Routine now
        if isinstance(prac_wo_mix, data.TrialHandler2) and thisPrac_wo_mix.thisN != prac_wo_mix.thisTrial.thisN:
            continueRoutine = False
        prac_wo_mix_fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_wo_mix_fix_text* updates
            
            # if prac_wo_mix_fix_text is starting this frame...
            if prac_wo_mix_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_fix_text.frameNStart = frameN  # exact frame index
                prac_wo_mix_fix_text.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_fix_text.started')
                # update status
                prac_wo_mix_fix_text.status = STARTED
                prac_wo_mix_fix_text.setAutoDraw(True)
            
            # if prac_wo_mix_fix_text is active this frame...
            if prac_wo_mix_fix_text.status == STARTED:
                # update params
                pass
            
            # if prac_wo_mix_fix_text is stopping this frame...
            if prac_wo_mix_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_wo_mix_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_wo_mix_fix_text.tStop = t  # not accounting for scr refresh
                    prac_wo_mix_fix_text.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_wo_mix_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_wo_mix_fix_text.stopped')
                    # update status
                    prac_wo_mix_fix_text.status = FINISHED
                    prac_wo_mix_fix_text.setAutoDraw(False)
            
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
                prac_wo_mix_fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_fix" ---
        for thisComponent in prac_wo_mix_fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_wo_mix_fix
        prac_wo_mix_fix.tStop = globalClock.getTime(format='float')
        prac_wo_mix_fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_wo_mix_fix.stopped', prac_wo_mix_fix.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_wo_mix_fix.maxDurationReached:
            routineTimer.addTime(-prac_wo_mix_fix.maxDuration)
        elif prac_wo_mix_fix.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "prac_wo_mix_face" ---
        # create an object to store info about Routine prac_wo_mix_face
        prac_wo_mix_face = data.Routine(
            name='prac_wo_mix_face',
            components=[prac_wo_mix_face_image],
        )
        prac_wo_mix_face.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_wo_mix_face_image.setImage(mixprac)
        # store start times for prac_wo_mix_face
        prac_wo_mix_face.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_wo_mix_face.tStart = globalClock.getTime(format='float')
        prac_wo_mix_face.status = STARTED
        thisExp.addData('prac_wo_mix_face.started', prac_wo_mix_face.tStart)
        prac_wo_mix_face.maxDuration = None
        # keep track of which components have finished
        prac_wo_mix_faceComponents = prac_wo_mix_face.components
        for thisComponent in prac_wo_mix_face.components:
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
        
        # --- Run Routine "prac_wo_mix_face" ---
        # if trial has changed, end Routine now
        if isinstance(prac_wo_mix, data.TrialHandler2) and thisPrac_wo_mix.thisN != prac_wo_mix.thisTrial.thisN:
            continueRoutine = False
        prac_wo_mix_face.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_wo_mix_face_image* updates
            
            # if prac_wo_mix_face_image is starting this frame...
            if prac_wo_mix_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_face_image.frameNStart = frameN  # exact frame index
                prac_wo_mix_face_image.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_face_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_face_image.started')
                # update status
                prac_wo_mix_face_image.status = STARTED
                prac_wo_mix_face_image.setAutoDraw(True)
            
            # if prac_wo_mix_face_image is active this frame...
            if prac_wo_mix_face_image.status == STARTED:
                # update params
                pass
            
            # if prac_wo_mix_face_image is stopping this frame...
            if prac_wo_mix_face_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_wo_mix_face_image.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_wo_mix_face_image.tStop = t  # not accounting for scr refresh
                    prac_wo_mix_face_image.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_wo_mix_face_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_wo_mix_face_image.stopped')
                    # update status
                    prac_wo_mix_face_image.status = FINISHED
                    prac_wo_mix_face_image.setAutoDraw(False)
            
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
                prac_wo_mix_face.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_face.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_face" ---
        for thisComponent in prac_wo_mix_face.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_wo_mix_face
        prac_wo_mix_face.tStop = globalClock.getTime(format='float')
        prac_wo_mix_face.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_wo_mix_face.stopped', prac_wo_mix_face.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_wo_mix_face.maxDurationReached:
            routineTimer.addTime(-prac_wo_mix_face.maxDuration)
        elif prac_wo_mix_face.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "prac_wo_mix_cate_resp" ---
        # create an object to store info about Routine prac_wo_mix_cate_resp
        prac_wo_mix_cate_resp = data.Routine(
            name='prac_wo_mix_cate_resp',
            components=[prac_wo_mix_cate_resp_prompt, prac_wo_mix_cate_resp_rect_left, prac_wo_mix_cate_resp_rect_right, prac_wo_mix_cate_resp_left_calm, prac_wo_mix_cate_resp_right_fear, prac_wo_mix_cate_resp_mouse],
        )
        prac_wo_mix_cate_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the prac_wo_mix_cate_resp_mouse
        prac_wo_mix_cate_resp_mouse.x = []
        prac_wo_mix_cate_resp_mouse.y = []
        prac_wo_mix_cate_resp_mouse.leftButton = []
        prac_wo_mix_cate_resp_mouse.midButton = []
        prac_wo_mix_cate_resp_mouse.rightButton = []
        prac_wo_mix_cate_resp_mouse.time = []
        prac_wo_mix_cate_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        prac_wo_mix_cate_resp_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from prac_wo_mix_cate_resp_code
        # 设置Rectangle的初始状态为不可见
        prac_wo_mix_cate_resp_rect_left.opacity = 0
        prac_wo_mix_cate_resp_rect_right.opacity = 0
        
        prac_wo_mix_cate_resp_mouse.setPos((0, 0))
        
        ################################
        
        # 检查当前试次应该运行的例程
        if prac_wo_mix.thisTrial['prac_mix_type'] == 'prac_wo_mix_cate_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        
        
        # store start times for prac_wo_mix_cate_resp
        prac_wo_mix_cate_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_wo_mix_cate_resp.tStart = globalClock.getTime(format='float')
        prac_wo_mix_cate_resp.status = STARTED
        thisExp.addData('prac_wo_mix_cate_resp.started', prac_wo_mix_cate_resp.tStart)
        prac_wo_mix_cate_resp.maxDuration = None
        # keep track of which components have finished
        prac_wo_mix_cate_respComponents = prac_wo_mix_cate_resp.components
        for thisComponent in prac_wo_mix_cate_resp.components:
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
        
        # --- Run Routine "prac_wo_mix_cate_resp" ---
        # if trial has changed, end Routine now
        if isinstance(prac_wo_mix, data.TrialHandler2) and thisPrac_wo_mix.thisN != prac_wo_mix.thisTrial.thisN:
            continueRoutine = False
        prac_wo_mix_cate_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_wo_mix_cate_resp_prompt* updates
            
            # if prac_wo_mix_cate_resp_prompt is starting this frame...
            if prac_wo_mix_cate_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cate_resp_prompt.frameNStart = frameN  # exact frame index
                prac_wo_mix_cate_resp_prompt.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cate_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cate_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cate_resp_prompt.started')
                # update status
                prac_wo_mix_cate_resp_prompt.status = STARTED
                prac_wo_mix_cate_resp_prompt.setAutoDraw(True)
            
            # if prac_wo_mix_cate_resp_prompt is active this frame...
            if prac_wo_mix_cate_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_cate_resp_rect_left* updates
            
            # if prac_wo_mix_cate_resp_rect_left is starting this frame...
            if prac_wo_mix_cate_resp_rect_left.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cate_resp_rect_left.frameNStart = frameN  # exact frame index
                prac_wo_mix_cate_resp_rect_left.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cate_resp_rect_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cate_resp_rect_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cate_resp_rect_left.started')
                # update status
                prac_wo_mix_cate_resp_rect_left.status = STARTED
                prac_wo_mix_cate_resp_rect_left.setAutoDraw(True)
            
            # if prac_wo_mix_cate_resp_rect_left is active this frame...
            if prac_wo_mix_cate_resp_rect_left.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_cate_resp_rect_right* updates
            
            # if prac_wo_mix_cate_resp_rect_right is starting this frame...
            if prac_wo_mix_cate_resp_rect_right.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cate_resp_rect_right.frameNStart = frameN  # exact frame index
                prac_wo_mix_cate_resp_rect_right.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cate_resp_rect_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cate_resp_rect_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cate_resp_rect_right.started')
                # update status
                prac_wo_mix_cate_resp_rect_right.status = STARTED
                prac_wo_mix_cate_resp_rect_right.setAutoDraw(True)
            
            # if prac_wo_mix_cate_resp_rect_right is active this frame...
            if prac_wo_mix_cate_resp_rect_right.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_cate_resp_left_calm* updates
            
            # if prac_wo_mix_cate_resp_left_calm is starting this frame...
            if prac_wo_mix_cate_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cate_resp_left_calm.frameNStart = frameN  # exact frame index
                prac_wo_mix_cate_resp_left_calm.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cate_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cate_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cate_resp_left_calm.started')
                # update status
                prac_wo_mix_cate_resp_left_calm.status = STARTED
                prac_wo_mix_cate_resp_left_calm.setAutoDraw(True)
            
            # if prac_wo_mix_cate_resp_left_calm is active this frame...
            if prac_wo_mix_cate_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_cate_resp_right_fear* updates
            
            # if prac_wo_mix_cate_resp_right_fear is starting this frame...
            if prac_wo_mix_cate_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cate_resp_right_fear.frameNStart = frameN  # exact frame index
                prac_wo_mix_cate_resp_right_fear.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cate_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cate_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cate_resp_right_fear.started')
                # update status
                prac_wo_mix_cate_resp_right_fear.status = STARTED
                prac_wo_mix_cate_resp_right_fear.setAutoDraw(True)
            
            # if prac_wo_mix_cate_resp_right_fear is active this frame...
            if prac_wo_mix_cate_resp_right_fear.status == STARTED:
                # update params
                pass
            # *prac_wo_mix_cate_resp_mouse* updates
            
            # if prac_wo_mix_cate_resp_mouse is starting this frame...
            if prac_wo_mix_cate_resp_mouse.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cate_resp_mouse.frameNStart = frameN  # exact frame index
                prac_wo_mix_cate_resp_mouse.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cate_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cate_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cate_resp_mouse.started')
                # update status
                prac_wo_mix_cate_resp_mouse.status = STARTED
                prevButtonState = prac_wo_mix_cate_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if prac_wo_mix_cate_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = prac_wo_mix_cate_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([prac_wo_mix_cate_resp_left_calm, prac_wo_mix_cate_resp_right_fear], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(prac_wo_mix_cate_resp_mouse):
                                gotValidClick = True
                                prac_wo_mix_cate_resp_mouse.clicked_name.append(obj.name)
                                prac_wo_mix_cate_resp_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = prac_wo_mix_cate_resp_mouse.getPos()
                            prac_wo_mix_cate_resp_mouse.x.append(x)
                            prac_wo_mix_cate_resp_mouse.y.append(y)
                            buttons = prac_wo_mix_cate_resp_mouse.getPressed()
                            prac_wo_mix_cate_resp_mouse.leftButton.append(buttons[0])
                            prac_wo_mix_cate_resp_mouse.midButton.append(buttons[1])
                            prac_wo_mix_cate_resp_mouse.rightButton.append(buttons[2])
                            prac_wo_mix_cate_resp_mouse.time.append(prac_wo_mix_cate_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from prac_wo_mix_cate_resp_code
            # 检测鼠标是否悬停在左边单词上方
            if prac_wo_mix_cate_resp_left_calm.contains(prac_wo_mix_cate_resp_mouse):
                prac_wo_mix_cate_resp_rect_left.opacity = 1  # 设置左边Rectangle可见
            else:
                prac_wo_mix_cate_resp_rect_left.opacity = 0  # 设置左边Rectangle不可见
            
            # 检测鼠标是否悬停在右边单词上方
            if prac_wo_mix_cate_resp_right_fear.contains(prac_wo_mix_cate_resp_mouse):
                prac_wo_mix_cate_resp_rect_right.opacity = 1  # 设置右边Rectangle可见
            else:
                prac_wo_mix_cate_resp_rect_right.opacity = 0  # 设置右边Rectangle不可见
            
            win.mouseVisible = True
            
            if prac_wo_mix_cate_resp_mouse.isPressedIn(prac_wo_mix_cate_resp_left_calm, buttons=[0]):
                continueRoutine = False
            elif prac_wo_mix_cate_resp_mouse.isPressedIn(prac_wo_mix_cate_resp_right_fear, buttons=[0]):
                continueRoutine = False
            else:
                continueRoutine = True
            
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
                prac_wo_mix_cate_resp.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_cate_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_cate_resp" ---
        for thisComponent in prac_wo_mix_cate_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_wo_mix_cate_resp
        prac_wo_mix_cate_resp.tStop = globalClock.getTime(format='float')
        prac_wo_mix_cate_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_wo_mix_cate_resp.stopped', prac_wo_mix_cate_resp.tStop)
        # store data for prac_wo_mix (TrialHandler)
        prac_wo_mix.addData('prac_wo_mix_cate_resp_mouse.x', prac_wo_mix_cate_resp_mouse.x)
        prac_wo_mix.addData('prac_wo_mix_cate_resp_mouse.y', prac_wo_mix_cate_resp_mouse.y)
        prac_wo_mix.addData('prac_wo_mix_cate_resp_mouse.leftButton', prac_wo_mix_cate_resp_mouse.leftButton)
        prac_wo_mix.addData('prac_wo_mix_cate_resp_mouse.midButton', prac_wo_mix_cate_resp_mouse.midButton)
        prac_wo_mix.addData('prac_wo_mix_cate_resp_mouse.rightButton', prac_wo_mix_cate_resp_mouse.rightButton)
        prac_wo_mix.addData('prac_wo_mix_cate_resp_mouse.time', prac_wo_mix_cate_resp_mouse.time)
        prac_wo_mix.addData('prac_wo_mix_cate_resp_mouse.clicked_name', prac_wo_mix_cate_resp_mouse.clicked_name)
        # Run 'End Routine' code from prac_wo_mix_cate_resp_code
        
        win.mouseVisible = False
        
        # the Routine "prac_wo_mix_cate_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_wo_mix_cont_resp" ---
        # create an object to store info about Routine prac_wo_mix_cont_resp
        prac_wo_mix_cont_resp = data.Routine(
            name='prac_wo_mix_cont_resp',
            components=[prac_wo_mix_cont_resp_prompt, prac_wo_mix_cont_resp_slider_shape, prac_wo_mix_cont_resp_slider, prac_wo_mix_cont_resp_left_calm, prac_wo_mix_cont_resp_right_fear, prac_wo_mix_cont_resp_mouse],
        )
        prac_wo_mix_cont_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_wo_mix_cont_resp_slider.reset()
        # setup some python lists for storing info about the prac_wo_mix_cont_resp_mouse
        prac_wo_mix_cont_resp_mouse.x = []
        prac_wo_mix_cont_resp_mouse.y = []
        prac_wo_mix_cont_resp_mouse.leftButton = []
        prac_wo_mix_cont_resp_mouse.midButton = []
        prac_wo_mix_cont_resp_mouse.rightButton = []
        prac_wo_mix_cont_resp_mouse.time = []
        prac_wo_mix_cont_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        prac_wo_mix_cont_resp_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from prac_wo_mix_cont_resp_code
        # Change the following parameters according to your own exp
        slider_width = 0.5
        slider_height = .05
        slider_orientation = 0
        slider_ticks=[0, 1]
        slider_granularity = .1
        slider_decimals = 2
        prac_wo_mix_cont_resp_slider.decimals = 2
        # 获取Slider组件的标记（marker）
        marker = prac_wo_mix_cont_resp_slider.marker
        # 将标记的形状更改为一个特定尺寸的方块
        marker.shape = 'square'
        marker.size = (0.01, 0.05)  # 设置方块的尺寸
        # Begin Routine
        slider_data=[]
        
        # record the current position of the mouse
        mouseRec=prac_wo_mix_cont_resp_mouse.getPos()
        
        prac_wo_mix_cont_resp_mouse.setPos((0, 0))
        
        ################################
        
        # 检查当前试次应该运行的例程
        if prac_wo_mix.thisTrial['prac_mix_type'] == 'prac_wo_mix_cont_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        
        # store start times for prac_wo_mix_cont_resp
        prac_wo_mix_cont_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_wo_mix_cont_resp.tStart = globalClock.getTime(format='float')
        prac_wo_mix_cont_resp.status = STARTED
        thisExp.addData('prac_wo_mix_cont_resp.started', prac_wo_mix_cont_resp.tStart)
        prac_wo_mix_cont_resp.maxDuration = None
        # keep track of which components have finished
        prac_wo_mix_cont_respComponents = prac_wo_mix_cont_resp.components
        for thisComponent in prac_wo_mix_cont_resp.components:
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
        
        # --- Run Routine "prac_wo_mix_cont_resp" ---
        # if trial has changed, end Routine now
        if isinstance(prac_wo_mix, data.TrialHandler2) and thisPrac_wo_mix.thisN != prac_wo_mix.thisTrial.thisN:
            continueRoutine = False
        prac_wo_mix_cont_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_wo_mix_cont_resp_prompt* updates
            
            # if prac_wo_mix_cont_resp_prompt is starting this frame...
            if prac_wo_mix_cont_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cont_resp_prompt.frameNStart = frameN  # exact frame index
                prac_wo_mix_cont_resp_prompt.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cont_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cont_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cont_resp_prompt.started')
                # update status
                prac_wo_mix_cont_resp_prompt.status = STARTED
                prac_wo_mix_cont_resp_prompt.setAutoDraw(True)
            
            # if prac_wo_mix_cont_resp_prompt is active this frame...
            if prac_wo_mix_cont_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_cont_resp_slider_shape* updates
            
            # if prac_wo_mix_cont_resp_slider_shape is starting this frame...
            if prac_wo_mix_cont_resp_slider_shape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cont_resp_slider_shape.frameNStart = frameN  # exact frame index
                prac_wo_mix_cont_resp_slider_shape.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cont_resp_slider_shape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cont_resp_slider_shape, 'tStartRefresh')  # time at next scr refresh
                # update status
                prac_wo_mix_cont_resp_slider_shape.status = STARTED
                prac_wo_mix_cont_resp_slider_shape.setAutoDraw(True)
            
            # if prac_wo_mix_cont_resp_slider_shape is active this frame...
            if prac_wo_mix_cont_resp_slider_shape.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_cont_resp_slider* updates
            
            # if prac_wo_mix_cont_resp_slider is starting this frame...
            if prac_wo_mix_cont_resp_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cont_resp_slider.frameNStart = frameN  # exact frame index
                prac_wo_mix_cont_resp_slider.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cont_resp_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cont_resp_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cont_resp_slider.started')
                # update status
                prac_wo_mix_cont_resp_slider.status = STARTED
                prac_wo_mix_cont_resp_slider.setAutoDraw(True)
            
            # if prac_wo_mix_cont_resp_slider is active this frame...
            if prac_wo_mix_cont_resp_slider.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_cont_resp_left_calm* updates
            
            # if prac_wo_mix_cont_resp_left_calm is starting this frame...
            if prac_wo_mix_cont_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cont_resp_left_calm.frameNStart = frameN  # exact frame index
                prac_wo_mix_cont_resp_left_calm.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cont_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cont_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cont_resp_left_calm.started')
                # update status
                prac_wo_mix_cont_resp_left_calm.status = STARTED
                prac_wo_mix_cont_resp_left_calm.setAutoDraw(True)
            
            # if prac_wo_mix_cont_resp_left_calm is active this frame...
            if prac_wo_mix_cont_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *prac_wo_mix_cont_resp_right_fear* updates
            
            # if prac_wo_mix_cont_resp_right_fear is starting this frame...
            if prac_wo_mix_cont_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cont_resp_right_fear.frameNStart = frameN  # exact frame index
                prac_wo_mix_cont_resp_right_fear.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cont_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cont_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_wo_mix_cont_resp_right_fear.started')
                # update status
                prac_wo_mix_cont_resp_right_fear.status = STARTED
                prac_wo_mix_cont_resp_right_fear.setAutoDraw(True)
            
            # if prac_wo_mix_cont_resp_right_fear is active this frame...
            if prac_wo_mix_cont_resp_right_fear.status == STARTED:
                # update params
                pass
            # *prac_wo_mix_cont_resp_mouse* updates
            
            # if prac_wo_mix_cont_resp_mouse is starting this frame...
            if prac_wo_mix_cont_resp_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_wo_mix_cont_resp_mouse.frameNStart = frameN  # exact frame index
                prac_wo_mix_cont_resp_mouse.tStart = t  # local t and not account for scr refresh
                prac_wo_mix_cont_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_wo_mix_cont_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('prac_wo_mix_cont_resp_mouse.started', t)
                # update status
                prac_wo_mix_cont_resp_mouse.status = STARTED
                prevButtonState = prac_wo_mix_cont_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if prac_wo_mix_cont_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = prac_wo_mix_cont_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(prac_wo_mix_cont_resp_slider_shape, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(prac_wo_mix_cont_resp_mouse):
                                gotValidClick = True
                                prac_wo_mix_cont_resp_mouse.clicked_name.append(obj.name)
                                prac_wo_mix_cont_resp_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = prac_wo_mix_cont_resp_mouse.getPos()
                            prac_wo_mix_cont_resp_mouse.x.append(x)
                            prac_wo_mix_cont_resp_mouse.y.append(y)
                            buttons = prac_wo_mix_cont_resp_mouse.getPressed()
                            prac_wo_mix_cont_resp_mouse.leftButton.append(buttons[0])
                            prac_wo_mix_cont_resp_mouse.midButton.append(buttons[1])
                            prac_wo_mix_cont_resp_mouse.rightButton.append(buttons[2])
                            prac_wo_mix_cont_resp_mouse.time.append(prac_wo_mix_cont_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from prac_wo_mix_cont_resp_code
            if prac_wo_mix_cont_resp_slider_shape.contains(prac_wo_mix_cont_resp_mouse) and prac_wo_mix_cont_resp_mouse.getPos()[slider_orientation] != mouseRec[slider_orientation]:
                mouseRec = prac_wo_mix_cont_resp_mouse.getPos()
                prac_wo_mix_cont_resp_slider.markerPos = mouseRec[slider_orientation]/slider_width*(slider_ticks[-1]-slider_ticks[0])+(slider_ticks[0]+slider_ticks[-1])/2
                
                # (new)Record marker position only on left mouse click
                slider_data.append([round(prac_wo_mix_cont_resp_slider.markerPos, slider_decimals)])
            
            
            win.mouseVisible = True
            
            #只有左键结束
            if prac_wo_mix_cont_resp_mouse.isPressedIn(prac_wo_mix_cont_resp_slider_shape, buttons=[0]):
                continueRoutine = False
            
            else:
                continueRoutine = True
            
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
                prac_wo_mix_cont_resp.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_cont_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_cont_resp" ---
        for thisComponent in prac_wo_mix_cont_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_wo_mix_cont_resp
        prac_wo_mix_cont_resp.tStop = globalClock.getTime(format='float')
        prac_wo_mix_cont_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_wo_mix_cont_resp.stopped', prac_wo_mix_cont_resp.tStop)
        prac_wo_mix.addData('prac_wo_mix_cont_resp_slider.response', prac_wo_mix_cont_resp_slider.getRating())
        prac_wo_mix.addData('prac_wo_mix_cont_resp_slider.rt', prac_wo_mix_cont_resp_slider.getRT())
        # store data for prac_wo_mix (TrialHandler)
        prac_wo_mix.addData('prac_wo_mix_cont_resp_mouse.x', prac_wo_mix_cont_resp_mouse.x)
        prac_wo_mix.addData('prac_wo_mix_cont_resp_mouse.y', prac_wo_mix_cont_resp_mouse.y)
        prac_wo_mix.addData('prac_wo_mix_cont_resp_mouse.leftButton', prac_wo_mix_cont_resp_mouse.leftButton)
        prac_wo_mix.addData('prac_wo_mix_cont_resp_mouse.midButton', prac_wo_mix_cont_resp_mouse.midButton)
        prac_wo_mix.addData('prac_wo_mix_cont_resp_mouse.rightButton', prac_wo_mix_cont_resp_mouse.rightButton)
        prac_wo_mix.addData('prac_wo_mix_cont_resp_mouse.time', prac_wo_mix_cont_resp_mouse.time)
        prac_wo_mix.addData('prac_wo_mix_cont_resp_mouse.clicked_name', prac_wo_mix_cont_resp_mouse.clicked_name)
        # Run 'End Routine' code from prac_wo_mix_cont_resp_code
        
        win.mouseVisible = False
        
        # the Routine "prac_wo_mix_cont_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'prac_wo_mix'
    
    
    # --- Prepare to start Routine "prac_wo_end" ---
    # create an object to store info about Routine prac_wo_end
    prac_wo_end = data.Routine(
        name='prac_wo_end',
        components=[prac_wo_end_text, prac_wo_end_key_resp],
    )
    prac_wo_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for prac_wo_end_key_resp
    prac_wo_end_key_resp.keys = []
    prac_wo_end_key_resp.rt = []
    _prac_wo_end_key_resp_allKeys = []
    # store start times for prac_wo_end
    prac_wo_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prac_wo_end.tStart = globalClock.getTime(format='float')
    prac_wo_end.status = STARTED
    thisExp.addData('prac_wo_end.started', prac_wo_end.tStart)
    prac_wo_end.maxDuration = None
    # keep track of which components have finished
    prac_wo_endComponents = prac_wo_end.components
    for thisComponent in prac_wo_end.components:
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
    
    # --- Run Routine "prac_wo_end" ---
    prac_wo_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prac_wo_end_text* updates
        
        # if prac_wo_end_text is starting this frame...
        if prac_wo_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_wo_end_text.frameNStart = frameN  # exact frame index
            prac_wo_end_text.tStart = t  # local t and not account for scr refresh
            prac_wo_end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_wo_end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_wo_end_text.started')
            # update status
            prac_wo_end_text.status = STARTED
            prac_wo_end_text.setAutoDraw(True)
        
        # if prac_wo_end_text is active this frame...
        if prac_wo_end_text.status == STARTED:
            # update params
            pass
        
        # *prac_wo_end_key_resp* updates
        waitOnFlip = False
        
        # if prac_wo_end_key_resp is starting this frame...
        if prac_wo_end_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_wo_end_key_resp.frameNStart = frameN  # exact frame index
            prac_wo_end_key_resp.tStart = t  # local t and not account for scr refresh
            prac_wo_end_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_wo_end_key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            prac_wo_end_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(prac_wo_end_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(prac_wo_end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if prac_wo_end_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = prac_wo_end_key_resp.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _prac_wo_end_key_resp_allKeys.extend(theseKeys)
            if len(_prac_wo_end_key_resp_allKeys):
                prac_wo_end_key_resp.keys = _prac_wo_end_key_resp_allKeys[-1].name  # just the last key pressed
                prac_wo_end_key_resp.rt = _prac_wo_end_key_resp_allKeys[-1].rt
                prac_wo_end_key_resp.duration = _prac_wo_end_key_resp_allKeys[-1].duration
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
            prac_wo_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_wo_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_wo_end" ---
    for thisComponent in prac_wo_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prac_wo_end
    prac_wo_end.tStop = globalClock.getTime(format='float')
    prac_wo_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prac_wo_end.stopped', prac_wo_end.tStop)
    # check responses
    if prac_wo_end_key_resp.keys in ['', [], None]:  # No response was made
        prac_wo_end_key_resp.keys = None
    thisExp.addData('prac_wo_end_key_resp.keys',prac_wo_end_key_resp.keys)
    if prac_wo_end_key_resp.keys != None:  # we had a response
        thisExp.addData('prac_wo_end_key_resp.rt', prac_wo_end_key_resp.rt)
        thisExp.addData('prac_wo_end_key_resp.duration', prac_wo_end_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "prac_wo_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    wo_mix = data.TrialHandler2(
        name='wo_mix',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'pics - 4.5.xlsx', 
        selection='0'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(wo_mix)  # add the loop to the experiment
    thisWo_mix = wo_mix.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisWo_mix.rgb)
    if thisWo_mix != None:
        for paramName in thisWo_mix:
            globals()[paramName] = thisWo_mix[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisWo_mix in wo_mix:
        currentLoop = wo_mix
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisWo_mix.rgb)
        if thisWo_mix != None:
            for paramName in thisWo_mix:
                globals()[paramName] = thisWo_mix[paramName]
        
        # --- Prepare to start Routine "wo_mix_begin" ---
        # create an object to store info about Routine wo_mix_begin
        wo_mix_begin = data.Routine(
            name='wo_mix_begin',
            components=[wo_mix_begin_text, wo_mix_begin_key_resp],
        )
        wo_mix_begin.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for wo_mix_begin_key_resp
        wo_mix_begin_key_resp.keys = []
        wo_mix_begin_key_resp.rt = []
        _wo_mix_begin_key_resp_allKeys = []
        # store start times for wo_mix_begin
        wo_mix_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wo_mix_begin.tStart = globalClock.getTime(format='float')
        wo_mix_begin.status = STARTED
        thisExp.addData('wo_mix_begin.started', wo_mix_begin.tStart)
        wo_mix_begin.maxDuration = None
        # keep track of which components have finished
        wo_mix_beginComponents = wo_mix_begin.components
        for thisComponent in wo_mix_begin.components:
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
        
        # --- Run Routine "wo_mix_begin" ---
        # if trial has changed, end Routine now
        if isinstance(wo_mix, data.TrialHandler2) and thisWo_mix.thisN != wo_mix.thisTrial.thisN:
            continueRoutine = False
        wo_mix_begin.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *wo_mix_begin_text* updates
            
            # if wo_mix_begin_text is starting this frame...
            if wo_mix_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_begin_text.frameNStart = frameN  # exact frame index
                wo_mix_begin_text.tStart = t  # local t and not account for scr refresh
                wo_mix_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_begin_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_begin_text.started')
                # update status
                wo_mix_begin_text.status = STARTED
                wo_mix_begin_text.setAutoDraw(True)
            
            # if wo_mix_begin_text is active this frame...
            if wo_mix_begin_text.status == STARTED:
                # update params
                pass
            
            # *wo_mix_begin_key_resp* updates
            waitOnFlip = False
            
            # if wo_mix_begin_key_resp is starting this frame...
            if wo_mix_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_begin_key_resp.frameNStart = frameN  # exact frame index
                wo_mix_begin_key_resp.tStart = t  # local t and not account for scr refresh
                wo_mix_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_begin_key_resp.started')
                # update status
                wo_mix_begin_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(wo_mix_begin_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(wo_mix_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if wo_mix_begin_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = wo_mix_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _wo_mix_begin_key_resp_allKeys.extend(theseKeys)
                if len(_wo_mix_begin_key_resp_allKeys):
                    wo_mix_begin_key_resp.keys = _wo_mix_begin_key_resp_allKeys[-1].name  # just the last key pressed
                    wo_mix_begin_key_resp.rt = _wo_mix_begin_key_resp_allKeys[-1].rt
                    wo_mix_begin_key_resp.duration = _wo_mix_begin_key_resp_allKeys[-1].duration
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
                wo_mix_begin.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_begin.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_begin" ---
        for thisComponent in wo_mix_begin.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wo_mix_begin
        wo_mix_begin.tStop = globalClock.getTime(format='float')
        wo_mix_begin.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wo_mix_begin.stopped', wo_mix_begin.tStop)
        # check responses
        if wo_mix_begin_key_resp.keys in ['', [], None]:  # No response was made
            wo_mix_begin_key_resp.keys = None
        wo_mix.addData('wo_mix_begin_key_resp.keys',wo_mix_begin_key_resp.keys)
        if wo_mix_begin_key_resp.keys != None:  # we had a response
            wo_mix.addData('wo_mix_begin_key_resp.rt', wo_mix_begin_key_resp.rt)
            wo_mix.addData('wo_mix_begin_key_resp.duration', wo_mix_begin_key_resp.duration)
        # the Routine "wo_mix_begin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wo_mix_fix" ---
        # create an object to store info about Routine wo_mix_fix
        wo_mix_fix = data.Routine(
            name='wo_mix_fix',
            components=[wo_mix_fix_text],
        )
        wo_mix_fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from wo_mix_fix_Init_trial
        block_name = 'mix'
        
        # store start times for wo_mix_fix
        wo_mix_fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wo_mix_fix.tStart = globalClock.getTime(format='float')
        wo_mix_fix.status = STARTED
        thisExp.addData('wo_mix_fix.started', wo_mix_fix.tStart)
        wo_mix_fix.maxDuration = None
        # keep track of which components have finished
        wo_mix_fixComponents = wo_mix_fix.components
        for thisComponent in wo_mix_fix.components:
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
        
        # --- Run Routine "wo_mix_fix" ---
        # if trial has changed, end Routine now
        if isinstance(wo_mix, data.TrialHandler2) and thisWo_mix.thisN != wo_mix.thisTrial.thisN:
            continueRoutine = False
        wo_mix_fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *wo_mix_fix_text* updates
            
            # if wo_mix_fix_text is starting this frame...
            if wo_mix_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_fix_text.frameNStart = frameN  # exact frame index
                wo_mix_fix_text.tStart = t  # local t and not account for scr refresh
                wo_mix_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_fix_text.started')
                # update status
                wo_mix_fix_text.status = STARTED
                wo_mix_fix_text.setAutoDraw(True)
            
            # if wo_mix_fix_text is active this frame...
            if wo_mix_fix_text.status == STARTED:
                # update params
                pass
            
            # if wo_mix_fix_text is stopping this frame...
            if wo_mix_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wo_mix_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    wo_mix_fix_text.tStop = t  # not accounting for scr refresh
                    wo_mix_fix_text.tStopRefresh = tThisFlipGlobal  # on global time
                    wo_mix_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wo_mix_fix_text.stopped')
                    # update status
                    wo_mix_fix_text.status = FINISHED
                    wo_mix_fix_text.setAutoDraw(False)
            
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
                wo_mix_fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_fix" ---
        for thisComponent in wo_mix_fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wo_mix_fix
        wo_mix_fix.tStop = globalClock.getTime(format='float')
        wo_mix_fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wo_mix_fix.stopped', wo_mix_fix.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if wo_mix_fix.maxDurationReached:
            routineTimer.addTime(-wo_mix_fix.maxDuration)
        elif wo_mix_fix.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "wo_mix_face" ---
        # create an object to store info about Routine wo_mix_face
        wo_mix_face = data.Routine(
            name='wo_mix_face',
            components=[wo_mix_face_image],
        )
        wo_mix_face.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        wo_mix_face_image.setImage(mix)
        # store start times for wo_mix_face
        wo_mix_face.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wo_mix_face.tStart = globalClock.getTime(format='float')
        wo_mix_face.status = STARTED
        thisExp.addData('wo_mix_face.started', wo_mix_face.tStart)
        wo_mix_face.maxDuration = None
        # keep track of which components have finished
        wo_mix_faceComponents = wo_mix_face.components
        for thisComponent in wo_mix_face.components:
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
        
        # --- Run Routine "wo_mix_face" ---
        # if trial has changed, end Routine now
        if isinstance(wo_mix, data.TrialHandler2) and thisWo_mix.thisN != wo_mix.thisTrial.thisN:
            continueRoutine = False
        wo_mix_face.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *wo_mix_face_image* updates
            
            # if wo_mix_face_image is starting this frame...
            if wo_mix_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_face_image.frameNStart = frameN  # exact frame index
                wo_mix_face_image.tStart = t  # local t and not account for scr refresh
                wo_mix_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_face_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_face_image.started')
                # update status
                wo_mix_face_image.status = STARTED
                wo_mix_face_image.setAutoDraw(True)
            
            # if wo_mix_face_image is active this frame...
            if wo_mix_face_image.status == STARTED:
                # update params
                pass
            
            # if wo_mix_face_image is stopping this frame...
            if wo_mix_face_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wo_mix_face_image.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    wo_mix_face_image.tStop = t  # not accounting for scr refresh
                    wo_mix_face_image.tStopRefresh = tThisFlipGlobal  # on global time
                    wo_mix_face_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wo_mix_face_image.stopped')
                    # update status
                    wo_mix_face_image.status = FINISHED
                    wo_mix_face_image.setAutoDraw(False)
            
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
                wo_mix_face.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_face.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_face" ---
        for thisComponent in wo_mix_face.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wo_mix_face
        wo_mix_face.tStop = globalClock.getTime(format='float')
        wo_mix_face.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wo_mix_face.stopped', wo_mix_face.tStop)
        # Run 'End Routine' code from wo_mix_face_code
        trialID = trialID + 1
        
        thisExp.addData('trialID', trialID)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if wo_mix_face.maxDurationReached:
            routineTimer.addTime(-wo_mix_face.maxDuration)
        elif wo_mix_face.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "wo_mix_cate_resp" ---
        # create an object to store info about Routine wo_mix_cate_resp
        wo_mix_cate_resp = data.Routine(
            name='wo_mix_cate_resp',
            components=[wo_mix_cate_resp_prompt, wo_mix_cate_resp_rect_left, wo_mix_cate_resp_rect_right, wo_mix_cate_resp_left_calm, wo_mix_cate_resp_right_fear, wo_mix_cate_resp_mouse],
        )
        wo_mix_cate_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the wo_mix_cate_resp_mouse
        wo_mix_cate_resp_mouse.x = []
        wo_mix_cate_resp_mouse.y = []
        wo_mix_cate_resp_mouse.leftButton = []
        wo_mix_cate_resp_mouse.midButton = []
        wo_mix_cate_resp_mouse.rightButton = []
        wo_mix_cate_resp_mouse.time = []
        wo_mix_cate_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        wo_mix_cate_resp_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from wo_mix_cate_resp_code
        # 设置Rectangle的初始状态为不可见
        wo_mix_cate_resp_rect_left.opacity = 0
        wo_mix_cate_resp_rect_right.opacity = 0
        
        wo_mix_cate_resp_mouse.setPos((0, 0))
        
        ################################
        
        # 检查当前试次应该运行的例程
        if wo_mix.thisTrial['mix_type'] == 'wo_mix_cate_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        
        cate_resp = 'null'
        
        # store start times for wo_mix_cate_resp
        wo_mix_cate_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wo_mix_cate_resp.tStart = globalClock.getTime(format='float')
        wo_mix_cate_resp.status = STARTED
        thisExp.addData('wo_mix_cate_resp.started', wo_mix_cate_resp.tStart)
        wo_mix_cate_resp.maxDuration = None
        # keep track of which components have finished
        wo_mix_cate_respComponents = wo_mix_cate_resp.components
        for thisComponent in wo_mix_cate_resp.components:
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
        
        # --- Run Routine "wo_mix_cate_resp" ---
        # if trial has changed, end Routine now
        if isinstance(wo_mix, data.TrialHandler2) and thisWo_mix.thisN != wo_mix.thisTrial.thisN:
            continueRoutine = False
        wo_mix_cate_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *wo_mix_cate_resp_prompt* updates
            
            # if wo_mix_cate_resp_prompt is starting this frame...
            if wo_mix_cate_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cate_resp_prompt.frameNStart = frameN  # exact frame index
                wo_mix_cate_resp_prompt.tStart = t  # local t and not account for scr refresh
                wo_mix_cate_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cate_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cate_resp_prompt.started')
                # update status
                wo_mix_cate_resp_prompt.status = STARTED
                wo_mix_cate_resp_prompt.setAutoDraw(True)
            
            # if wo_mix_cate_resp_prompt is active this frame...
            if wo_mix_cate_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *wo_mix_cate_resp_rect_left* updates
            
            # if wo_mix_cate_resp_rect_left is starting this frame...
            if wo_mix_cate_resp_rect_left.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cate_resp_rect_left.frameNStart = frameN  # exact frame index
                wo_mix_cate_resp_rect_left.tStart = t  # local t and not account for scr refresh
                wo_mix_cate_resp_rect_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cate_resp_rect_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cate_resp_rect_left.started')
                # update status
                wo_mix_cate_resp_rect_left.status = STARTED
                wo_mix_cate_resp_rect_left.setAutoDraw(True)
            
            # if wo_mix_cate_resp_rect_left is active this frame...
            if wo_mix_cate_resp_rect_left.status == STARTED:
                # update params
                pass
            
            # *wo_mix_cate_resp_rect_right* updates
            
            # if wo_mix_cate_resp_rect_right is starting this frame...
            if wo_mix_cate_resp_rect_right.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cate_resp_rect_right.frameNStart = frameN  # exact frame index
                wo_mix_cate_resp_rect_right.tStart = t  # local t and not account for scr refresh
                wo_mix_cate_resp_rect_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cate_resp_rect_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cate_resp_rect_right.started')
                # update status
                wo_mix_cate_resp_rect_right.status = STARTED
                wo_mix_cate_resp_rect_right.setAutoDraw(True)
            
            # if wo_mix_cate_resp_rect_right is active this frame...
            if wo_mix_cate_resp_rect_right.status == STARTED:
                # update params
                pass
            
            # *wo_mix_cate_resp_left_calm* updates
            
            # if wo_mix_cate_resp_left_calm is starting this frame...
            if wo_mix_cate_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cate_resp_left_calm.frameNStart = frameN  # exact frame index
                wo_mix_cate_resp_left_calm.tStart = t  # local t and not account for scr refresh
                wo_mix_cate_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cate_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cate_resp_left_calm.started')
                # update status
                wo_mix_cate_resp_left_calm.status = STARTED
                wo_mix_cate_resp_left_calm.setAutoDraw(True)
            
            # if wo_mix_cate_resp_left_calm is active this frame...
            if wo_mix_cate_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *wo_mix_cate_resp_right_fear* updates
            
            # if wo_mix_cate_resp_right_fear is starting this frame...
            if wo_mix_cate_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cate_resp_right_fear.frameNStart = frameN  # exact frame index
                wo_mix_cate_resp_right_fear.tStart = t  # local t and not account for scr refresh
                wo_mix_cate_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cate_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cate_resp_right_fear.started')
                # update status
                wo_mix_cate_resp_right_fear.status = STARTED
                wo_mix_cate_resp_right_fear.setAutoDraw(True)
            
            # if wo_mix_cate_resp_right_fear is active this frame...
            if wo_mix_cate_resp_right_fear.status == STARTED:
                # update params
                pass
            # *wo_mix_cate_resp_mouse* updates
            
            # if wo_mix_cate_resp_mouse is starting this frame...
            if wo_mix_cate_resp_mouse.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cate_resp_mouse.frameNStart = frameN  # exact frame index
                wo_mix_cate_resp_mouse.tStart = t  # local t and not account for scr refresh
                wo_mix_cate_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cate_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cate_resp_mouse.started')
                # update status
                wo_mix_cate_resp_mouse.status = STARTED
                prevButtonState = wo_mix_cate_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if wo_mix_cate_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = wo_mix_cate_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([wo_mix_cate_resp_left_calm, wo_mix_cate_resp_right_fear], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(wo_mix_cate_resp_mouse):
                                gotValidClick = True
                                wo_mix_cate_resp_mouse.clicked_name.append(obj.name)
                                wo_mix_cate_resp_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = wo_mix_cate_resp_mouse.getPos()
                            wo_mix_cate_resp_mouse.x.append(x)
                            wo_mix_cate_resp_mouse.y.append(y)
                            buttons = wo_mix_cate_resp_mouse.getPressed()
                            wo_mix_cate_resp_mouse.leftButton.append(buttons[0])
                            wo_mix_cate_resp_mouse.midButton.append(buttons[1])
                            wo_mix_cate_resp_mouse.rightButton.append(buttons[2])
                            wo_mix_cate_resp_mouse.time.append(wo_mix_cate_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from wo_mix_cate_resp_code
            # 检测鼠标是否悬停在左边单词上方
            if wo_mix_cate_resp_left_calm.contains(wo_mix_cate_resp_mouse):
                wo_mix_cate_resp_rect_left.opacity = 1  # 设置左边Rectangle可见
            else:
                wo_mix_cate_resp_rect_left.opacity = 0  # 设置左边Rectangle不可见
            
            # 检测鼠标是否悬停在右边单词上方
            if wo_mix_cate_resp_right_fear.contains(wo_mix_cate_resp_mouse):
                wo_mix_cate_resp_rect_right.opacity = 1  # 设置右边Rectangle可见
            else:
                wo_mix_cate_resp_rect_right.opacity = 0  # 设置右边Rectangle不可见
            
            # 检测鼠标点击事件
            if wo_mix_cate_resp_mouse.getPressed()[0]:
                # 检测鼠标点击是否在左边单词上方
                if wo_mix_cate_resp_left_calm.contains(wo_mix_cate_resp_mouse):
                    cate_resp = 0  # 记录被试选择左边单词
                    continueRoutine = False
                # 检测鼠标点击是否在右边单词上方
                elif wo_mix_cate_resp_right_fear.contains(wo_mix_cate_resp_mouse):
                    cate_resp = 1  # 记录被试选择右边单词
                    continueRoutine = False
                else:
                    cate_resp = 'None'
                    continueRoutine = True
            else:
                cate_resp = 'None'
                continueRoutine = True
                
                
            win.mouseVisible = True
            
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
                wo_mix_cate_resp.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_cate_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_cate_resp" ---
        for thisComponent in wo_mix_cate_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wo_mix_cate_resp
        wo_mix_cate_resp.tStop = globalClock.getTime(format='float')
        wo_mix_cate_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wo_mix_cate_resp.stopped', wo_mix_cate_resp.tStop)
        # store data for wo_mix (TrialHandler)
        wo_mix.addData('wo_mix_cate_resp_mouse.x', wo_mix_cate_resp_mouse.x)
        wo_mix.addData('wo_mix_cate_resp_mouse.y', wo_mix_cate_resp_mouse.y)
        wo_mix.addData('wo_mix_cate_resp_mouse.leftButton', wo_mix_cate_resp_mouse.leftButton)
        wo_mix.addData('wo_mix_cate_resp_mouse.midButton', wo_mix_cate_resp_mouse.midButton)
        wo_mix.addData('wo_mix_cate_resp_mouse.rightButton', wo_mix_cate_resp_mouse.rightButton)
        wo_mix.addData('wo_mix_cate_resp_mouse.time', wo_mix_cate_resp_mouse.time)
        wo_mix.addData('wo_mix_cate_resp_mouse.clicked_name', wo_mix_cate_resp_mouse.clicked_name)
        # Run 'End Routine' code from wo_mix_cate_resp_code
        
        thisExp.addData('block_name', block_name)
        thisExp.addData('face_stim', mix)
        thisExp.addData('cate_resp', cate_resp)
        
        thisExp.addData('block_mix_type', mix_type)
        win.mouseVisible = False
        
        # the Routine "wo_mix_cate_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wo_mix_cont_resp" ---
        # create an object to store info about Routine wo_mix_cont_resp
        wo_mix_cont_resp = data.Routine(
            name='wo_mix_cont_resp',
            components=[wo_mix_cont_resp_prompt, wo_mix_cont_resp_slider_shape, wo_mix_cont_resp_slider, wo_mix_cont_resp_left_calm, wo_mix_cont_resp_right_fear, wo_mix_cont_resp_mouse],
        )
        wo_mix_cont_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        wo_mix_cont_resp_slider.reset()
        # setup some python lists for storing info about the wo_mix_cont_resp_mouse
        wo_mix_cont_resp_mouse.x = []
        wo_mix_cont_resp_mouse.y = []
        wo_mix_cont_resp_mouse.leftButton = []
        wo_mix_cont_resp_mouse.midButton = []
        wo_mix_cont_resp_mouse.rightButton = []
        wo_mix_cont_resp_mouse.time = []
        wo_mix_cont_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        wo_mix_cont_resp_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from wo_mix_cont_resp_code
        # Change the following parameters according to your own exp
        slider_width = 0.5
        slider_height = .05
        slider_orientation = 0
        #Failing to set ticks or labels in code
        slider_ticks=[0, 1]
        
        slider_granularity = .1
        slider_decimals = 2
        wo_mix_cont_resp_slider.decimals = 2
        # 获取Slider组件的标记（marker）
        marker = wo_mix_cont_resp_slider.marker
        # 将标记的形状更改为一个特定尺寸的方块
        marker.shape = 'square'
        marker.size = (0.01, 0.05)  # 设置方块的尺寸
        
        # record the current position of the mouse
        mouseRec=wo_mix_cont_resp_mouse.getPos()
        
        wo_mix_cont_resp_mouse.setPos((0, 0))
        
        ################################
        
        # 检查当前试次应该运行的例程
        if wo_mix.thisTrial['mix_type'] == 'wo_mix_cont_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        
        cont_resp = 'null'
        
        # store start times for wo_mix_cont_resp
        wo_mix_cont_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wo_mix_cont_resp.tStart = globalClock.getTime(format='float')
        wo_mix_cont_resp.status = STARTED
        thisExp.addData('wo_mix_cont_resp.started', wo_mix_cont_resp.tStart)
        wo_mix_cont_resp.maxDuration = None
        # keep track of which components have finished
        wo_mix_cont_respComponents = wo_mix_cont_resp.components
        for thisComponent in wo_mix_cont_resp.components:
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
        
        # --- Run Routine "wo_mix_cont_resp" ---
        # if trial has changed, end Routine now
        if isinstance(wo_mix, data.TrialHandler2) and thisWo_mix.thisN != wo_mix.thisTrial.thisN:
            continueRoutine = False
        wo_mix_cont_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *wo_mix_cont_resp_prompt* updates
            
            # if wo_mix_cont_resp_prompt is starting this frame...
            if wo_mix_cont_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cont_resp_prompt.frameNStart = frameN  # exact frame index
                wo_mix_cont_resp_prompt.tStart = t  # local t and not account for scr refresh
                wo_mix_cont_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cont_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cont_resp_prompt.started')
                # update status
                wo_mix_cont_resp_prompt.status = STARTED
                wo_mix_cont_resp_prompt.setAutoDraw(True)
            
            # if wo_mix_cont_resp_prompt is active this frame...
            if wo_mix_cont_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *wo_mix_cont_resp_slider_shape* updates
            
            # if wo_mix_cont_resp_slider_shape is starting this frame...
            if wo_mix_cont_resp_slider_shape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cont_resp_slider_shape.frameNStart = frameN  # exact frame index
                wo_mix_cont_resp_slider_shape.tStart = t  # local t and not account for scr refresh
                wo_mix_cont_resp_slider_shape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cont_resp_slider_shape, 'tStartRefresh')  # time at next scr refresh
                # update status
                wo_mix_cont_resp_slider_shape.status = STARTED
                wo_mix_cont_resp_slider_shape.setAutoDraw(True)
            
            # if wo_mix_cont_resp_slider_shape is active this frame...
            if wo_mix_cont_resp_slider_shape.status == STARTED:
                # update params
                pass
            
            # *wo_mix_cont_resp_slider* updates
            
            # if wo_mix_cont_resp_slider is starting this frame...
            if wo_mix_cont_resp_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cont_resp_slider.frameNStart = frameN  # exact frame index
                wo_mix_cont_resp_slider.tStart = t  # local t and not account for scr refresh
                wo_mix_cont_resp_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cont_resp_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cont_resp_slider.started')
                # update status
                wo_mix_cont_resp_slider.status = STARTED
                wo_mix_cont_resp_slider.setAutoDraw(True)
            
            # if wo_mix_cont_resp_slider is active this frame...
            if wo_mix_cont_resp_slider.status == STARTED:
                # update params
                pass
            
            # *wo_mix_cont_resp_left_calm* updates
            
            # if wo_mix_cont_resp_left_calm is starting this frame...
            if wo_mix_cont_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cont_resp_left_calm.frameNStart = frameN  # exact frame index
                wo_mix_cont_resp_left_calm.tStart = t  # local t and not account for scr refresh
                wo_mix_cont_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cont_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cont_resp_left_calm.started')
                # update status
                wo_mix_cont_resp_left_calm.status = STARTED
                wo_mix_cont_resp_left_calm.setAutoDraw(True)
            
            # if wo_mix_cont_resp_left_calm is active this frame...
            if wo_mix_cont_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *wo_mix_cont_resp_right_fear* updates
            
            # if wo_mix_cont_resp_right_fear is starting this frame...
            if wo_mix_cont_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cont_resp_right_fear.frameNStart = frameN  # exact frame index
                wo_mix_cont_resp_right_fear.tStart = t  # local t and not account for scr refresh
                wo_mix_cont_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cont_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_cont_resp_right_fear.started')
                # update status
                wo_mix_cont_resp_right_fear.status = STARTED
                wo_mix_cont_resp_right_fear.setAutoDraw(True)
            
            # if wo_mix_cont_resp_right_fear is active this frame...
            if wo_mix_cont_resp_right_fear.status == STARTED:
                # update params
                pass
            # *wo_mix_cont_resp_mouse* updates
            
            # if wo_mix_cont_resp_mouse is starting this frame...
            if wo_mix_cont_resp_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_cont_resp_mouse.frameNStart = frameN  # exact frame index
                wo_mix_cont_resp_mouse.tStart = t  # local t and not account for scr refresh
                wo_mix_cont_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_cont_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('wo_mix_cont_resp_mouse.started', t)
                # update status
                wo_mix_cont_resp_mouse.status = STARTED
                prevButtonState = wo_mix_cont_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if wo_mix_cont_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = wo_mix_cont_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(wo_mix_cont_resp_slider_shape, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(wo_mix_cont_resp_mouse):
                                gotValidClick = True
                                wo_mix_cont_resp_mouse.clicked_name.append(obj.name)
                                wo_mix_cont_resp_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = wo_mix_cont_resp_mouse.getPos()
                            wo_mix_cont_resp_mouse.x.append(x)
                            wo_mix_cont_resp_mouse.y.append(y)
                            buttons = wo_mix_cont_resp_mouse.getPressed()
                            wo_mix_cont_resp_mouse.leftButton.append(buttons[0])
                            wo_mix_cont_resp_mouse.midButton.append(buttons[1])
                            wo_mix_cont_resp_mouse.rightButton.append(buttons[2])
                            wo_mix_cont_resp_mouse.time.append(wo_mix_cont_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from wo_mix_cont_resp_code
            
            # Create a slider_shape in this routine with the same size as the slider
            # Check if mouse is moving within slider_shape, if so change marker position accordingly
            if wo_mix_cont_resp_slider_shape.contains(wo_mix_cont_resp_mouse):
                mouseRec = wo_mix_cont_resp_mouse.getPos()
                wo_mix_cont_resp_slider.markerPos = mouseRec[slider_orientation]/slider_width*(slider_ticks[-1]-slider_ticks[0])+(slider_ticks[0]+slider_ticks[-1])/2
                # (new)Record marker position only on left mouse click
                if wo_mix_cont_resp_mouse.getPressed()[0]:
                    cont_resp = round(wo_mix_cont_resp_slider.markerPos, slider_decimals)
                    continueRoutine = False
                else:
                    cont_resp = 'None'
                    continueRoutine = True
            else:
                cont_resp = 'None'
                continueRoutine = True
                    
            
            win.mouseVisible = True
            
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
                wo_mix_cont_resp.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_cont_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_cont_resp" ---
        for thisComponent in wo_mix_cont_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wo_mix_cont_resp
        wo_mix_cont_resp.tStop = globalClock.getTime(format='float')
        wo_mix_cont_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wo_mix_cont_resp.stopped', wo_mix_cont_resp.tStop)
        wo_mix.addData('wo_mix_cont_resp_slider.response', wo_mix_cont_resp_slider.getRating())
        wo_mix.addData('wo_mix_cont_resp_slider.rt', wo_mix_cont_resp_slider.getRT())
        # store data for wo_mix (TrialHandler)
        wo_mix.addData('wo_mix_cont_resp_mouse.x', wo_mix_cont_resp_mouse.x)
        wo_mix.addData('wo_mix_cont_resp_mouse.y', wo_mix_cont_resp_mouse.y)
        wo_mix.addData('wo_mix_cont_resp_mouse.leftButton', wo_mix_cont_resp_mouse.leftButton)
        wo_mix.addData('wo_mix_cont_resp_mouse.midButton', wo_mix_cont_resp_mouse.midButton)
        wo_mix.addData('wo_mix_cont_resp_mouse.rightButton', wo_mix_cont_resp_mouse.rightButton)
        wo_mix.addData('wo_mix_cont_resp_mouse.time', wo_mix_cont_resp_mouse.time)
        wo_mix.addData('wo_mix_cont_resp_mouse.clicked_name', wo_mix_cont_resp_mouse.clicked_name)
        # Run 'End Routine' code from wo_mix_cont_resp_code
        
        thisExp.addData('block_name', block_name)
        thisExp.addData('face_stim', mix)
        thisExp.addData('cont_resp',cont_resp)
        
        thisExp.addData('block_mix_type', mix_type)
        win.mouseVisible = False
        
        
        
        # the Routine "wo_mix_cont_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wo_mix_brea" ---
        # create an object to store info about Routine wo_mix_brea
        wo_mix_brea = data.Routine(
            name='wo_mix_brea',
            components=[wo_mix_brea_text, wo_mix_brea_key_resp],
        )
        wo_mix_brea.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for wo_mix_brea_key_resp
        wo_mix_brea_key_resp.keys = []
        wo_mix_brea_key_resp.rt = []
        _wo_mix_brea_key_resp_allKeys = []
        # Run 'Begin Routine' code from wo_mix_brea_code
        if trialID == 50:
            continueRoutine = True
        else:
            continueRoutine = False
        # store start times for wo_mix_brea
        wo_mix_brea.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wo_mix_brea.tStart = globalClock.getTime(format='float')
        wo_mix_brea.status = STARTED
        thisExp.addData('wo_mix_brea.started', wo_mix_brea.tStart)
        wo_mix_brea.maxDuration = None
        # keep track of which components have finished
        wo_mix_breaComponents = wo_mix_brea.components
        for thisComponent in wo_mix_brea.components:
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
        
        # --- Run Routine "wo_mix_brea" ---
        # if trial has changed, end Routine now
        if isinstance(wo_mix, data.TrialHandler2) and thisWo_mix.thisN != wo_mix.thisTrial.thisN:
            continueRoutine = False
        wo_mix_brea.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *wo_mix_brea_text* updates
            
            # if wo_mix_brea_text is starting this frame...
            if wo_mix_brea_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_brea_text.frameNStart = frameN  # exact frame index
                wo_mix_brea_text.tStart = t  # local t and not account for scr refresh
                wo_mix_brea_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_brea_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wo_mix_brea_text.started')
                # update status
                wo_mix_brea_text.status = STARTED
                wo_mix_brea_text.setAutoDraw(True)
            
            # if wo_mix_brea_text is active this frame...
            if wo_mix_brea_text.status == STARTED:
                # update params
                pass
            
            # *wo_mix_brea_key_resp* updates
            waitOnFlip = False
            
            # if wo_mix_brea_key_resp is starting this frame...
            if wo_mix_brea_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wo_mix_brea_key_resp.frameNStart = frameN  # exact frame index
                wo_mix_brea_key_resp.tStart = t  # local t and not account for scr refresh
                wo_mix_brea_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wo_mix_brea_key_resp, 'tStartRefresh')  # time at next scr refresh
                # update status
                wo_mix_brea_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(wo_mix_brea_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(wo_mix_brea_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if wo_mix_brea_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = wo_mix_brea_key_resp.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
                _wo_mix_brea_key_resp_allKeys.extend(theseKeys)
                if len(_wo_mix_brea_key_resp_allKeys):
                    wo_mix_brea_key_resp.keys = _wo_mix_brea_key_resp_allKeys[-1].name  # just the last key pressed
                    wo_mix_brea_key_resp.rt = _wo_mix_brea_key_resp_allKeys[-1].rt
                    wo_mix_brea_key_resp.duration = _wo_mix_brea_key_resp_allKeys[-1].duration
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
                wo_mix_brea.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_brea.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_brea" ---
        for thisComponent in wo_mix_brea.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wo_mix_brea
        wo_mix_brea.tStop = globalClock.getTime(format='float')
        wo_mix_brea.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wo_mix_brea.stopped', wo_mix_brea.tStop)
        # check responses
        if wo_mix_brea_key_resp.keys in ['', [], None]:  # No response was made
            wo_mix_brea_key_resp.keys = None
        wo_mix.addData('wo_mix_brea_key_resp.keys',wo_mix_brea_key_resp.keys)
        if wo_mix_brea_key_resp.keys != None:  # we had a response
            wo_mix.addData('wo_mix_brea_key_resp.rt', wo_mix_brea_key_resp.rt)
            wo_mix.addData('wo_mix_brea_key_resp.duration', wo_mix_brea_key_resp.duration)
        # the Routine "wo_mix_brea" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'wo_mix'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "formal_wo_end" ---
    # create an object to store info about Routine formal_wo_end
    formal_wo_end = data.Routine(
        name='formal_wo_end',
        components=[formal_wo_end_text, formal_wo_end_key_resp],
    )
    formal_wo_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for formal_wo_end_key_resp
    formal_wo_end_key_resp.keys = []
    formal_wo_end_key_resp.rt = []
    _formal_wo_end_key_resp_allKeys = []
    # store start times for formal_wo_end
    formal_wo_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    formal_wo_end.tStart = globalClock.getTime(format='float')
    formal_wo_end.status = STARTED
    thisExp.addData('formal_wo_end.started', formal_wo_end.tStart)
    formal_wo_end.maxDuration = None
    # keep track of which components have finished
    formal_wo_endComponents = formal_wo_end.components
    for thisComponent in formal_wo_end.components:
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
    
    # --- Run Routine "formal_wo_end" ---
    formal_wo_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *formal_wo_end_text* updates
        
        # if formal_wo_end_text is starting this frame...
        if formal_wo_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            formal_wo_end_text.frameNStart = frameN  # exact frame index
            formal_wo_end_text.tStart = t  # local t and not account for scr refresh
            formal_wo_end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(formal_wo_end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'formal_wo_end_text.started')
            # update status
            formal_wo_end_text.status = STARTED
            formal_wo_end_text.setAutoDraw(True)
        
        # if formal_wo_end_text is active this frame...
        if formal_wo_end_text.status == STARTED:
            # update params
            pass
        
        # *formal_wo_end_key_resp* updates
        waitOnFlip = False
        
        # if formal_wo_end_key_resp is starting this frame...
        if formal_wo_end_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            formal_wo_end_key_resp.frameNStart = frameN  # exact frame index
            formal_wo_end_key_resp.tStart = t  # local t and not account for scr refresh
            formal_wo_end_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(formal_wo_end_key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            formal_wo_end_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(formal_wo_end_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(formal_wo_end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if formal_wo_end_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = formal_wo_end_key_resp.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _formal_wo_end_key_resp_allKeys.extend(theseKeys)
            if len(_formal_wo_end_key_resp_allKeys):
                formal_wo_end_key_resp.keys = _formal_wo_end_key_resp_allKeys[-1].name  # just the last key pressed
                formal_wo_end_key_resp.rt = _formal_wo_end_key_resp_allKeys[-1].rt
                formal_wo_end_key_resp.duration = _formal_wo_end_key_resp_allKeys[-1].duration
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
            formal_wo_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in formal_wo_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "formal_wo_end" ---
    for thisComponent in formal_wo_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for formal_wo_end
    formal_wo_end.tStop = globalClock.getTime(format='float')
    formal_wo_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('formal_wo_end.stopped', formal_wo_end.tStop)
    # check responses
    if formal_wo_end_key_resp.keys in ['', [], None]:  # No response was made
        formal_wo_end_key_resp.keys = None
    thisExp.addData('formal_wo_end_key_resp.keys',formal_wo_end_key_resp.keys)
    if formal_wo_end_key_resp.keys != None:  # we had a response
        thisExp.addData('formal_wo_end_key_resp.rt', formal_wo_end_key_resp.rt)
        thisExp.addData('formal_wo_end_key_resp.duration', formal_wo_end_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "formal_wo_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "with_intro_1" ---
    # create an object to store info about Routine with_intro_1
    with_intro_1 = data.Routine(
        name='with_intro_1',
        components=[with_intro_1_text, with_intro_1_key_resp],
    )
    with_intro_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for with_intro_1_key_resp
    with_intro_1_key_resp.keys = []
    with_intro_1_key_resp.rt = []
    _with_intro_1_key_resp_allKeys = []
    # store start times for with_intro_1
    with_intro_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    with_intro_1.tStart = globalClock.getTime(format='float')
    with_intro_1.status = STARTED
    thisExp.addData('with_intro_1.started', with_intro_1.tStart)
    with_intro_1.maxDuration = None
    # keep track of which components have finished
    with_intro_1Components = with_intro_1.components
    for thisComponent in with_intro_1.components:
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
    
    # --- Run Routine "with_intro_1" ---
    with_intro_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *with_intro_1_text* updates
        
        # if with_intro_1_text is starting this frame...
        if with_intro_1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_1_text.frameNStart = frameN  # exact frame index
            with_intro_1_text.tStart = t  # local t and not account for scr refresh
            with_intro_1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_1_text.started')
            # update status
            with_intro_1_text.status = STARTED
            with_intro_1_text.setAutoDraw(True)
        
        # if with_intro_1_text is active this frame...
        if with_intro_1_text.status == STARTED:
            # update params
            pass
        
        # *with_intro_1_key_resp* updates
        waitOnFlip = False
        
        # if with_intro_1_key_resp is starting this frame...
        if with_intro_1_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_1_key_resp.frameNStart = frameN  # exact frame index
            with_intro_1_key_resp.tStart = t  # local t and not account for scr refresh
            with_intro_1_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_1_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_1_key_resp.started')
            # update status
            with_intro_1_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(with_intro_1_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(with_intro_1_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if with_intro_1_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = with_intro_1_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _with_intro_1_key_resp_allKeys.extend(theseKeys)
            if len(_with_intro_1_key_resp_allKeys):
                with_intro_1_key_resp.keys = _with_intro_1_key_resp_allKeys[-1].name  # just the last key pressed
                with_intro_1_key_resp.rt = _with_intro_1_key_resp_allKeys[-1].rt
                with_intro_1_key_resp.duration = _with_intro_1_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # Run 'Each Frame' code from with_intro_1_code
        win.mouseVisible = False
        
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
            with_intro_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_1" ---
    for thisComponent in with_intro_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for with_intro_1
    with_intro_1.tStop = globalClock.getTime(format='float')
    with_intro_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('with_intro_1.stopped', with_intro_1.tStop)
    # check responses
    if with_intro_1_key_resp.keys in ['', [], None]:  # No response was made
        with_intro_1_key_resp.keys = None
    thisExp.addData('with_intro_1_key_resp.keys',with_intro_1_key_resp.keys)
    if with_intro_1_key_resp.keys != None:  # we had a response
        thisExp.addData('with_intro_1_key_resp.rt', with_intro_1_key_resp.rt)
        thisExp.addData('with_intro_1_key_resp.duration', with_intro_1_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "with_intro_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "with_intro_2" ---
    # create an object to store info about Routine with_intro_2
    with_intro_2 = data.Routine(
        name='with_intro_2',
        components=[with_intro_2_text, with_intro_2_key_resp, with_intro_2_left_calm, with_intro_2_right_fear],
    )
    with_intro_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for with_intro_2_key_resp
    with_intro_2_key_resp.keys = []
    with_intro_2_key_resp.rt = []
    _with_intro_2_key_resp_allKeys = []
    # store start times for with_intro_2
    with_intro_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    with_intro_2.tStart = globalClock.getTime(format='float')
    with_intro_2.status = STARTED
    thisExp.addData('with_intro_2.started', with_intro_2.tStart)
    with_intro_2.maxDuration = None
    # keep track of which components have finished
    with_intro_2Components = with_intro_2.components
    for thisComponent in with_intro_2.components:
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
    
    # --- Run Routine "with_intro_2" ---
    with_intro_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *with_intro_2_text* updates
        
        # if with_intro_2_text is starting this frame...
        if with_intro_2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_2_text.frameNStart = frameN  # exact frame index
            with_intro_2_text.tStart = t  # local t and not account for scr refresh
            with_intro_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_2_text.started')
            # update status
            with_intro_2_text.status = STARTED
            with_intro_2_text.setAutoDraw(True)
        
        # if with_intro_2_text is active this frame...
        if with_intro_2_text.status == STARTED:
            # update params
            pass
        
        # *with_intro_2_key_resp* updates
        waitOnFlip = False
        
        # if with_intro_2_key_resp is starting this frame...
        if with_intro_2_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_2_key_resp.frameNStart = frameN  # exact frame index
            with_intro_2_key_resp.tStart = t  # local t and not account for scr refresh
            with_intro_2_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_2_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_2_key_resp.started')
            # update status
            with_intro_2_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(with_intro_2_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(with_intro_2_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if with_intro_2_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = with_intro_2_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _with_intro_2_key_resp_allKeys.extend(theseKeys)
            if len(_with_intro_2_key_resp_allKeys):
                with_intro_2_key_resp.keys = _with_intro_2_key_resp_allKeys[-1].name  # just the last key pressed
                with_intro_2_key_resp.rt = _with_intro_2_key_resp_allKeys[-1].rt
                with_intro_2_key_resp.duration = _with_intro_2_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *with_intro_2_left_calm* updates
        
        # if with_intro_2_left_calm is starting this frame...
        if with_intro_2_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_2_left_calm.frameNStart = frameN  # exact frame index
            with_intro_2_left_calm.tStart = t  # local t and not account for scr refresh
            with_intro_2_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_2_left_calm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_2_left_calm.started')
            # update status
            with_intro_2_left_calm.status = STARTED
            with_intro_2_left_calm.setAutoDraw(True)
        
        # if with_intro_2_left_calm is active this frame...
        if with_intro_2_left_calm.status == STARTED:
            # update params
            pass
        
        # *with_intro_2_right_fear* updates
        
        # if with_intro_2_right_fear is starting this frame...
        if with_intro_2_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_2_right_fear.frameNStart = frameN  # exact frame index
            with_intro_2_right_fear.tStart = t  # local t and not account for scr refresh
            with_intro_2_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_2_right_fear, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_2_right_fear.started')
            # update status
            with_intro_2_right_fear.status = STARTED
            with_intro_2_right_fear.setAutoDraw(True)
        
        # if with_intro_2_right_fear is active this frame...
        if with_intro_2_right_fear.status == STARTED:
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
            with_intro_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_2" ---
    for thisComponent in with_intro_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for with_intro_2
    with_intro_2.tStop = globalClock.getTime(format='float')
    with_intro_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('with_intro_2.stopped', with_intro_2.tStop)
    # check responses
    if with_intro_2_key_resp.keys in ['', [], None]:  # No response was made
        with_intro_2_key_resp.keys = None
    thisExp.addData('with_intro_2_key_resp.keys',with_intro_2_key_resp.keys)
    if with_intro_2_key_resp.keys != None:  # we had a response
        thisExp.addData('with_intro_2_key_resp.rt', with_intro_2_key_resp.rt)
        thisExp.addData('with_intro_2_key_resp.duration', with_intro_2_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "with_intro_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "with_intro_3" ---
    # create an object to store info about Routine with_intro_3
    with_intro_3 = data.Routine(
        name='with_intro_3',
        components=[with_intro_3_text, with_intro_3_key_resp, with_intro_3_image],
    )
    with_intro_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for with_intro_3_key_resp
    with_intro_3_key_resp.keys = []
    with_intro_3_key_resp.rt = []
    _with_intro_3_key_resp_allKeys = []
    # store start times for with_intro_3
    with_intro_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    with_intro_3.tStart = globalClock.getTime(format='float')
    with_intro_3.status = STARTED
    thisExp.addData('with_intro_3.started', with_intro_3.tStart)
    with_intro_3.maxDuration = None
    # keep track of which components have finished
    with_intro_3Components = with_intro_3.components
    for thisComponent in with_intro_3.components:
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
    
    # --- Run Routine "with_intro_3" ---
    with_intro_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *with_intro_3_text* updates
        
        # if with_intro_3_text is starting this frame...
        if with_intro_3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_3_text.frameNStart = frameN  # exact frame index
            with_intro_3_text.tStart = t  # local t and not account for scr refresh
            with_intro_3_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_3_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_3_text.started')
            # update status
            with_intro_3_text.status = STARTED
            with_intro_3_text.setAutoDraw(True)
        
        # if with_intro_3_text is active this frame...
        if with_intro_3_text.status == STARTED:
            # update params
            pass
        
        # *with_intro_3_key_resp* updates
        waitOnFlip = False
        
        # if with_intro_3_key_resp is starting this frame...
        if with_intro_3_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_3_key_resp.frameNStart = frameN  # exact frame index
            with_intro_3_key_resp.tStart = t  # local t and not account for scr refresh
            with_intro_3_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_3_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_3_key_resp.started')
            # update status
            with_intro_3_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(with_intro_3_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(with_intro_3_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if with_intro_3_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = with_intro_3_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _with_intro_3_key_resp_allKeys.extend(theseKeys)
            if len(_with_intro_3_key_resp_allKeys):
                with_intro_3_key_resp.keys = _with_intro_3_key_resp_allKeys[-1].name  # just the last key pressed
                with_intro_3_key_resp.rt = _with_intro_3_key_resp_allKeys[-1].rt
                with_intro_3_key_resp.duration = _with_intro_3_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *with_intro_3_image* updates
        
        # if with_intro_3_image is starting this frame...
        if with_intro_3_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_3_image.frameNStart = frameN  # exact frame index
            with_intro_3_image.tStart = t  # local t and not account for scr refresh
            with_intro_3_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_3_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_3_image.started')
            # update status
            with_intro_3_image.status = STARTED
            with_intro_3_image.setAutoDraw(True)
        
        # if with_intro_3_image is active this frame...
        if with_intro_3_image.status == STARTED:
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
            with_intro_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_3" ---
    for thisComponent in with_intro_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for with_intro_3
    with_intro_3.tStop = globalClock.getTime(format='float')
    with_intro_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('with_intro_3.stopped', with_intro_3.tStop)
    # check responses
    if with_intro_3_key_resp.keys in ['', [], None]:  # No response was made
        with_intro_3_key_resp.keys = None
    thisExp.addData('with_intro_3_key_resp.keys',with_intro_3_key_resp.keys)
    if with_intro_3_key_resp.keys != None:  # we had a response
        thisExp.addData('with_intro_3_key_resp.rt', with_intro_3_key_resp.rt)
        thisExp.addData('with_intro_3_key_resp.duration', with_intro_3_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "with_intro_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "with_intro_4" ---
    # create an object to store info about Routine with_intro_4
    with_intro_4 = data.Routine(
        name='with_intro_4',
        components=[with_intro_4_text, with_intro_4_key_resp, with_intro_4_rect_left, with_intro_4_left_calm, with_intro_4_right_fear],
    )
    with_intro_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for with_intro_4_key_resp
    with_intro_4_key_resp.keys = []
    with_intro_4_key_resp.rt = []
    _with_intro_4_key_resp_allKeys = []
    # store start times for with_intro_4
    with_intro_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    with_intro_4.tStart = globalClock.getTime(format='float')
    with_intro_4.status = STARTED
    thisExp.addData('with_intro_4.started', with_intro_4.tStart)
    with_intro_4.maxDuration = None
    # keep track of which components have finished
    with_intro_4Components = with_intro_4.components
    for thisComponent in with_intro_4.components:
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
    
    # --- Run Routine "with_intro_4" ---
    with_intro_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *with_intro_4_text* updates
        
        # if with_intro_4_text is starting this frame...
        if with_intro_4_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_4_text.frameNStart = frameN  # exact frame index
            with_intro_4_text.tStart = t  # local t and not account for scr refresh
            with_intro_4_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_4_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_4_text.started')
            # update status
            with_intro_4_text.status = STARTED
            with_intro_4_text.setAutoDraw(True)
        
        # if with_intro_4_text is active this frame...
        if with_intro_4_text.status == STARTED:
            # update params
            pass
        
        # *with_intro_4_key_resp* updates
        waitOnFlip = False
        
        # if with_intro_4_key_resp is starting this frame...
        if with_intro_4_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_4_key_resp.frameNStart = frameN  # exact frame index
            with_intro_4_key_resp.tStart = t  # local t and not account for scr refresh
            with_intro_4_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_4_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_4_key_resp.started')
            # update status
            with_intro_4_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(with_intro_4_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(with_intro_4_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if with_intro_4_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = with_intro_4_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _with_intro_4_key_resp_allKeys.extend(theseKeys)
            if len(_with_intro_4_key_resp_allKeys):
                with_intro_4_key_resp.keys = _with_intro_4_key_resp_allKeys[-1].name  # just the last key pressed
                with_intro_4_key_resp.rt = _with_intro_4_key_resp_allKeys[-1].rt
                with_intro_4_key_resp.duration = _with_intro_4_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *with_intro_4_rect_left* updates
        
        # if with_intro_4_rect_left is starting this frame...
        if with_intro_4_rect_left.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_4_rect_left.frameNStart = frameN  # exact frame index
            with_intro_4_rect_left.tStart = t  # local t and not account for scr refresh
            with_intro_4_rect_left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_4_rect_left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_4_rect_left.started')
            # update status
            with_intro_4_rect_left.status = STARTED
            with_intro_4_rect_left.setAutoDraw(True)
        
        # if with_intro_4_rect_left is active this frame...
        if with_intro_4_rect_left.status == STARTED:
            # update params
            pass
        
        # *with_intro_4_left_calm* updates
        
        # if with_intro_4_left_calm is starting this frame...
        if with_intro_4_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_4_left_calm.frameNStart = frameN  # exact frame index
            with_intro_4_left_calm.tStart = t  # local t and not account for scr refresh
            with_intro_4_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_4_left_calm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_4_left_calm.started')
            # update status
            with_intro_4_left_calm.status = STARTED
            with_intro_4_left_calm.setAutoDraw(True)
        
        # if with_intro_4_left_calm is active this frame...
        if with_intro_4_left_calm.status == STARTED:
            # update params
            pass
        
        # *with_intro_4_right_fear* updates
        
        # if with_intro_4_right_fear is starting this frame...
        if with_intro_4_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_4_right_fear.frameNStart = frameN  # exact frame index
            with_intro_4_right_fear.tStart = t  # local t and not account for scr refresh
            with_intro_4_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_4_right_fear, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_4_right_fear.started')
            # update status
            with_intro_4_right_fear.status = STARTED
            with_intro_4_right_fear.setAutoDraw(True)
        
        # if with_intro_4_right_fear is active this frame...
        if with_intro_4_right_fear.status == STARTED:
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
            with_intro_4.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_4" ---
    for thisComponent in with_intro_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for with_intro_4
    with_intro_4.tStop = globalClock.getTime(format='float')
    with_intro_4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('with_intro_4.stopped', with_intro_4.tStop)
    # check responses
    if with_intro_4_key_resp.keys in ['', [], None]:  # No response was made
        with_intro_4_key_resp.keys = None
    thisExp.addData('with_intro_4_key_resp.keys',with_intro_4_key_resp.keys)
    if with_intro_4_key_resp.keys != None:  # we had a response
        thisExp.addData('with_intro_4_key_resp.rt', with_intro_4_key_resp.rt)
        thisExp.addData('with_intro_4_key_resp.duration', with_intro_4_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "with_intro_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "with_intro_5" ---
    # create an object to store info about Routine with_intro_5
    with_intro_5 = data.Routine(
        name='with_intro_5',
        components=[with_intro_5_text, with_intro_5_key_resp, with_intro_5_left_calm, with_intro_5_right_fear, with_intro_5_cont_fix_line],
    )
    with_intro_5.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for with_intro_5_key_resp
    with_intro_5_key_resp.keys = []
    with_intro_5_key_resp.rt = []
    _with_intro_5_key_resp_allKeys = []
    # store start times for with_intro_5
    with_intro_5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    with_intro_5.tStart = globalClock.getTime(format='float')
    with_intro_5.status = STARTED
    thisExp.addData('with_intro_5.started', with_intro_5.tStart)
    with_intro_5.maxDuration = None
    # keep track of which components have finished
    with_intro_5Components = with_intro_5.components
    for thisComponent in with_intro_5.components:
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
    
    # --- Run Routine "with_intro_5" ---
    with_intro_5.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *with_intro_5_text* updates
        
        # if with_intro_5_text is starting this frame...
        if with_intro_5_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_5_text.frameNStart = frameN  # exact frame index
            with_intro_5_text.tStart = t  # local t and not account for scr refresh
            with_intro_5_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_5_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_5_text.started')
            # update status
            with_intro_5_text.status = STARTED
            with_intro_5_text.setAutoDraw(True)
        
        # if with_intro_5_text is active this frame...
        if with_intro_5_text.status == STARTED:
            # update params
            pass
        
        # *with_intro_5_key_resp* updates
        waitOnFlip = False
        
        # if with_intro_5_key_resp is starting this frame...
        if with_intro_5_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_5_key_resp.frameNStart = frameN  # exact frame index
            with_intro_5_key_resp.tStart = t  # local t and not account for scr refresh
            with_intro_5_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_5_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_5_key_resp.started')
            # update status
            with_intro_5_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(with_intro_5_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(with_intro_5_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if with_intro_5_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = with_intro_5_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _with_intro_5_key_resp_allKeys.extend(theseKeys)
            if len(_with_intro_5_key_resp_allKeys):
                with_intro_5_key_resp.keys = _with_intro_5_key_resp_allKeys[-1].name  # just the last key pressed
                with_intro_5_key_resp.rt = _with_intro_5_key_resp_allKeys[-1].rt
                with_intro_5_key_resp.duration = _with_intro_5_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *with_intro_5_left_calm* updates
        
        # if with_intro_5_left_calm is starting this frame...
        if with_intro_5_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_5_left_calm.frameNStart = frameN  # exact frame index
            with_intro_5_left_calm.tStart = t  # local t and not account for scr refresh
            with_intro_5_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_5_left_calm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_5_left_calm.started')
            # update status
            with_intro_5_left_calm.status = STARTED
            with_intro_5_left_calm.setAutoDraw(True)
        
        # if with_intro_5_left_calm is active this frame...
        if with_intro_5_left_calm.status == STARTED:
            # update params
            pass
        
        # *with_intro_5_right_fear* updates
        
        # if with_intro_5_right_fear is starting this frame...
        if with_intro_5_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_5_right_fear.frameNStart = frameN  # exact frame index
            with_intro_5_right_fear.tStart = t  # local t and not account for scr refresh
            with_intro_5_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_5_right_fear, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_5_right_fear.started')
            # update status
            with_intro_5_right_fear.status = STARTED
            with_intro_5_right_fear.setAutoDraw(True)
        
        # if with_intro_5_right_fear is active this frame...
        if with_intro_5_right_fear.status == STARTED:
            # update params
            pass
        
        # *with_intro_5_cont_fix_line* updates
        
        # if with_intro_5_cont_fix_line is starting this frame...
        if with_intro_5_cont_fix_line.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_5_cont_fix_line.frameNStart = frameN  # exact frame index
            with_intro_5_cont_fix_line.tStart = t  # local t and not account for scr refresh
            with_intro_5_cont_fix_line.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_5_cont_fix_line, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_5_cont_fix_line.started')
            # update status
            with_intro_5_cont_fix_line.status = STARTED
            with_intro_5_cont_fix_line.setAutoDraw(True)
        
        # if with_intro_5_cont_fix_line is active this frame...
        if with_intro_5_cont_fix_line.status == STARTED:
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
            with_intro_5.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_5.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_5" ---
    for thisComponent in with_intro_5.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for with_intro_5
    with_intro_5.tStop = globalClock.getTime(format='float')
    with_intro_5.tStopRefresh = tThisFlipGlobal
    thisExp.addData('with_intro_5.stopped', with_intro_5.tStop)
    # check responses
    if with_intro_5_key_resp.keys in ['', [], None]:  # No response was made
        with_intro_5_key_resp.keys = None
    thisExp.addData('with_intro_5_key_resp.keys',with_intro_5_key_resp.keys)
    if with_intro_5_key_resp.keys != None:  # we had a response
        thisExp.addData('with_intro_5_key_resp.rt', with_intro_5_key_resp.rt)
        thisExp.addData('with_intro_5_key_resp.duration', with_intro_5_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "with_intro_5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "with_intro_6" ---
    # create an object to store info about Routine with_intro_6
    with_intro_6 = data.Routine(
        name='with_intro_6',
        components=[with_intro_6_text, with_intro_6_key_resp, with_intro_6_image],
    )
    with_intro_6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for with_intro_6_key_resp
    with_intro_6_key_resp.keys = []
    with_intro_6_key_resp.rt = []
    _with_intro_6_key_resp_allKeys = []
    # store start times for with_intro_6
    with_intro_6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    with_intro_6.tStart = globalClock.getTime(format='float')
    with_intro_6.status = STARTED
    thisExp.addData('with_intro_6.started', with_intro_6.tStart)
    with_intro_6.maxDuration = None
    # keep track of which components have finished
    with_intro_6Components = with_intro_6.components
    for thisComponent in with_intro_6.components:
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
    
    # --- Run Routine "with_intro_6" ---
    with_intro_6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *with_intro_6_text* updates
        
        # if with_intro_6_text is starting this frame...
        if with_intro_6_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_6_text.frameNStart = frameN  # exact frame index
            with_intro_6_text.tStart = t  # local t and not account for scr refresh
            with_intro_6_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_6_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_6_text.started')
            # update status
            with_intro_6_text.status = STARTED
            with_intro_6_text.setAutoDraw(True)
        
        # if with_intro_6_text is active this frame...
        if with_intro_6_text.status == STARTED:
            # update params
            pass
        
        # *with_intro_6_key_resp* updates
        waitOnFlip = False
        
        # if with_intro_6_key_resp is starting this frame...
        if with_intro_6_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_6_key_resp.frameNStart = frameN  # exact frame index
            with_intro_6_key_resp.tStart = t  # local t and not account for scr refresh
            with_intro_6_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_6_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_6_key_resp.started')
            # update status
            with_intro_6_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(with_intro_6_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(with_intro_6_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if with_intro_6_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = with_intro_6_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _with_intro_6_key_resp_allKeys.extend(theseKeys)
            if len(_with_intro_6_key_resp_allKeys):
                with_intro_6_key_resp.keys = _with_intro_6_key_resp_allKeys[-1].name  # just the last key pressed
                with_intro_6_key_resp.rt = _with_intro_6_key_resp_allKeys[-1].rt
                with_intro_6_key_resp.duration = _with_intro_6_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *with_intro_6_image* updates
        
        # if with_intro_6_image is starting this frame...
        if with_intro_6_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_6_image.frameNStart = frameN  # exact frame index
            with_intro_6_image.tStart = t  # local t and not account for scr refresh
            with_intro_6_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_6_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_6_image.started')
            # update status
            with_intro_6_image.status = STARTED
            with_intro_6_image.setAutoDraw(True)
        
        # if with_intro_6_image is active this frame...
        if with_intro_6_image.status == STARTED:
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
            with_intro_6.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_6" ---
    for thisComponent in with_intro_6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for with_intro_6
    with_intro_6.tStop = globalClock.getTime(format='float')
    with_intro_6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('with_intro_6.stopped', with_intro_6.tStop)
    # check responses
    if with_intro_6_key_resp.keys in ['', [], None]:  # No response was made
        with_intro_6_key_resp.keys = None
    thisExp.addData('with_intro_6_key_resp.keys',with_intro_6_key_resp.keys)
    if with_intro_6_key_resp.keys != None:  # we had a response
        thisExp.addData('with_intro_6_key_resp.rt', with_intro_6_key_resp.rt)
        thisExp.addData('with_intro_6_key_resp.duration', with_intro_6_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "with_intro_6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "with_intro_7" ---
    # create an object to store info about Routine with_intro_7
    with_intro_7 = data.Routine(
        name='with_intro_7',
        components=[with_intro_7_text, with_intro_7_key_resp, with_intro_7_left_calm, with_intro_7_right_fear, with_intro_7_marker, with_intro_7_cont_fix_line],
    )
    with_intro_7.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for with_intro_7_key_resp
    with_intro_7_key_resp.keys = []
    with_intro_7_key_resp.rt = []
    _with_intro_7_key_resp_allKeys = []
    # store start times for with_intro_7
    with_intro_7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    with_intro_7.tStart = globalClock.getTime(format='float')
    with_intro_7.status = STARTED
    thisExp.addData('with_intro_7.started', with_intro_7.tStart)
    with_intro_7.maxDuration = None
    # keep track of which components have finished
    with_intro_7Components = with_intro_7.components
    for thisComponent in with_intro_7.components:
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
    
    # --- Run Routine "with_intro_7" ---
    with_intro_7.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *with_intro_7_text* updates
        
        # if with_intro_7_text is starting this frame...
        if with_intro_7_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_7_text.frameNStart = frameN  # exact frame index
            with_intro_7_text.tStart = t  # local t and not account for scr refresh
            with_intro_7_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_7_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_7_text.started')
            # update status
            with_intro_7_text.status = STARTED
            with_intro_7_text.setAutoDraw(True)
        
        # if with_intro_7_text is active this frame...
        if with_intro_7_text.status == STARTED:
            # update params
            pass
        
        # *with_intro_7_key_resp* updates
        waitOnFlip = False
        
        # if with_intro_7_key_resp is starting this frame...
        if with_intro_7_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_7_key_resp.frameNStart = frameN  # exact frame index
            with_intro_7_key_resp.tStart = t  # local t and not account for scr refresh
            with_intro_7_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_7_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_7_key_resp.started')
            # update status
            with_intro_7_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(with_intro_7_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(with_intro_7_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if with_intro_7_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = with_intro_7_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _with_intro_7_key_resp_allKeys.extend(theseKeys)
            if len(_with_intro_7_key_resp_allKeys):
                with_intro_7_key_resp.keys = _with_intro_7_key_resp_allKeys[-1].name  # just the last key pressed
                with_intro_7_key_resp.rt = _with_intro_7_key_resp_allKeys[-1].rt
                with_intro_7_key_resp.duration = _with_intro_7_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *with_intro_7_left_calm* updates
        
        # if with_intro_7_left_calm is starting this frame...
        if with_intro_7_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_7_left_calm.frameNStart = frameN  # exact frame index
            with_intro_7_left_calm.tStart = t  # local t and not account for scr refresh
            with_intro_7_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_7_left_calm, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_7_left_calm.started')
            # update status
            with_intro_7_left_calm.status = STARTED
            with_intro_7_left_calm.setAutoDraw(True)
        
        # if with_intro_7_left_calm is active this frame...
        if with_intro_7_left_calm.status == STARTED:
            # update params
            pass
        
        # *with_intro_7_right_fear* updates
        
        # if with_intro_7_right_fear is starting this frame...
        if with_intro_7_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_7_right_fear.frameNStart = frameN  # exact frame index
            with_intro_7_right_fear.tStart = t  # local t and not account for scr refresh
            with_intro_7_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_7_right_fear, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_7_right_fear.started')
            # update status
            with_intro_7_right_fear.status = STARTED
            with_intro_7_right_fear.setAutoDraw(True)
        
        # if with_intro_7_right_fear is active this frame...
        if with_intro_7_right_fear.status == STARTED:
            # update params
            pass
        
        # *with_intro_7_marker* updates
        
        # if with_intro_7_marker is starting this frame...
        if with_intro_7_marker.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_7_marker.frameNStart = frameN  # exact frame index
            with_intro_7_marker.tStart = t  # local t and not account for scr refresh
            with_intro_7_marker.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_7_marker, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_7_marker.started')
            # update status
            with_intro_7_marker.status = STARTED
            with_intro_7_marker.setAutoDraw(True)
        
        # if with_intro_7_marker is active this frame...
        if with_intro_7_marker.status == STARTED:
            # update params
            pass
        
        # *with_intro_7_cont_fix_line* updates
        
        # if with_intro_7_cont_fix_line is starting this frame...
        if with_intro_7_cont_fix_line.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_7_cont_fix_line.frameNStart = frameN  # exact frame index
            with_intro_7_cont_fix_line.tStart = t  # local t and not account for scr refresh
            with_intro_7_cont_fix_line.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_7_cont_fix_line, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_7_cont_fix_line.started')
            # update status
            with_intro_7_cont_fix_line.status = STARTED
            with_intro_7_cont_fix_line.setAutoDraw(True)
        
        # if with_intro_7_cont_fix_line is active this frame...
        if with_intro_7_cont_fix_line.status == STARTED:
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
            with_intro_7.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_7.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_7" ---
    for thisComponent in with_intro_7.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for with_intro_7
    with_intro_7.tStop = globalClock.getTime(format='float')
    with_intro_7.tStopRefresh = tThisFlipGlobal
    thisExp.addData('with_intro_7.stopped', with_intro_7.tStop)
    # check responses
    if with_intro_7_key_resp.keys in ['', [], None]:  # No response was made
        with_intro_7_key_resp.keys = None
    thisExp.addData('with_intro_7_key_resp.keys',with_intro_7_key_resp.keys)
    if with_intro_7_key_resp.keys != None:  # we had a response
        thisExp.addData('with_intro_7_key_resp.rt', with_intro_7_key_resp.rt)
        thisExp.addData('with_intro_7_key_resp.duration', with_intro_7_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "with_intro_7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "with_intro_8" ---
    # create an object to store info about Routine with_intro_8
    with_intro_8 = data.Routine(
        name='with_intro_8',
        components=[with_intro_8_text, with_intro_8_key_resp],
    )
    with_intro_8.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for with_intro_8_key_resp
    with_intro_8_key_resp.keys = []
    with_intro_8_key_resp.rt = []
    _with_intro_8_key_resp_allKeys = []
    # store start times for with_intro_8
    with_intro_8.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    with_intro_8.tStart = globalClock.getTime(format='float')
    with_intro_8.status = STARTED
    thisExp.addData('with_intro_8.started', with_intro_8.tStart)
    with_intro_8.maxDuration = None
    # keep track of which components have finished
    with_intro_8Components = with_intro_8.components
    for thisComponent in with_intro_8.components:
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
    
    # --- Run Routine "with_intro_8" ---
    with_intro_8.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *with_intro_8_text* updates
        
        # if with_intro_8_text is starting this frame...
        if with_intro_8_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_8_text.frameNStart = frameN  # exact frame index
            with_intro_8_text.tStart = t  # local t and not account for scr refresh
            with_intro_8_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_8_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_8_text.started')
            # update status
            with_intro_8_text.status = STARTED
            with_intro_8_text.setAutoDraw(True)
        
        # if with_intro_8_text is active this frame...
        if with_intro_8_text.status == STARTED:
            # update params
            pass
        
        # *with_intro_8_key_resp* updates
        waitOnFlip = False
        
        # if with_intro_8_key_resp is starting this frame...
        if with_intro_8_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            with_intro_8_key_resp.frameNStart = frameN  # exact frame index
            with_intro_8_key_resp.tStart = t  # local t and not account for scr refresh
            with_intro_8_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(with_intro_8_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'with_intro_8_key_resp.started')
            # update status
            with_intro_8_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(with_intro_8_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(with_intro_8_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if with_intro_8_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = with_intro_8_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _with_intro_8_key_resp_allKeys.extend(theseKeys)
            if len(_with_intro_8_key_resp_allKeys):
                with_intro_8_key_resp.keys = _with_intro_8_key_resp_allKeys[-1].name  # just the last key pressed
                with_intro_8_key_resp.rt = _with_intro_8_key_resp_allKeys[-1].rt
                with_intro_8_key_resp.duration = _with_intro_8_key_resp_allKeys[-1].duration
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
            with_intro_8.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_8.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_8" ---
    for thisComponent in with_intro_8.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for with_intro_8
    with_intro_8.tStop = globalClock.getTime(format='float')
    with_intro_8.tStopRefresh = tThisFlipGlobal
    thisExp.addData('with_intro_8.stopped', with_intro_8.tStop)
    # check responses
    if with_intro_8_key_resp.keys in ['', [], None]:  # No response was made
        with_intro_8_key_resp.keys = None
    thisExp.addData('with_intro_8_key_resp.keys',with_intro_8_key_resp.keys)
    if with_intro_8_key_resp.keys != None:  # we had a response
        thisExp.addData('with_intro_8_key_resp.rt', with_intro_8_key_resp.rt)
        thisExp.addData('with_intro_8_key_resp.duration', with_intro_8_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "with_intro_8" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "prac_with_begin" ---
    # create an object to store info about Routine prac_with_begin
    prac_with_begin = data.Routine(
        name='prac_with_begin',
        components=[prac_with_begin_text, prac_with_begin_key_resp],
    )
    prac_with_begin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for prac_with_begin_key_resp
    prac_with_begin_key_resp.keys = []
    prac_with_begin_key_resp.rt = []
    _prac_with_begin_key_resp_allKeys = []
    # store start times for prac_with_begin
    prac_with_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prac_with_begin.tStart = globalClock.getTime(format='float')
    prac_with_begin.status = STARTED
    thisExp.addData('prac_with_begin.started', prac_with_begin.tStart)
    prac_with_begin.maxDuration = None
    # keep track of which components have finished
    prac_with_beginComponents = prac_with_begin.components
    for thisComponent in prac_with_begin.components:
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
    
    # --- Run Routine "prac_with_begin" ---
    prac_with_begin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prac_with_begin_text* updates
        
        # if prac_with_begin_text is starting this frame...
        if prac_with_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            prac_with_begin_text.frameNStart = frameN  # exact frame index
            prac_with_begin_text.tStart = t  # local t and not account for scr refresh
            prac_with_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_with_begin_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_with_begin_text.started')
            # update status
            prac_with_begin_text.status = STARTED
            prac_with_begin_text.setAutoDraw(True)
        
        # if prac_with_begin_text is active this frame...
        if prac_with_begin_text.status == STARTED:
            # update params
            pass
        
        # *prac_with_begin_key_resp* updates
        waitOnFlip = False
        
        # if prac_with_begin_key_resp is starting this frame...
        if prac_with_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_with_begin_key_resp.frameNStart = frameN  # exact frame index
            prac_with_begin_key_resp.tStart = t  # local t and not account for scr refresh
            prac_with_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_with_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_with_begin_key_resp.started')
            # update status
            prac_with_begin_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(prac_with_begin_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(prac_with_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if prac_with_begin_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = prac_with_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _prac_with_begin_key_resp_allKeys.extend(theseKeys)
            if len(_prac_with_begin_key_resp_allKeys):
                prac_with_begin_key_resp.keys = _prac_with_begin_key_resp_allKeys[-1].name  # just the last key pressed
                prac_with_begin_key_resp.rt = _prac_with_begin_key_resp_allKeys[-1].rt
                prac_with_begin_key_resp.duration = _prac_with_begin_key_resp_allKeys[-1].duration
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
            prac_with_begin.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_with_begin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_with_begin" ---
    for thisComponent in prac_with_begin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prac_with_begin
    prac_with_begin.tStop = globalClock.getTime(format='float')
    prac_with_begin.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prac_with_begin.stopped', prac_with_begin.tStop)
    # check responses
    if prac_with_begin_key_resp.keys in ['', [], None]:  # No response was made
        prac_with_begin_key_resp.keys = None
    thisExp.addData('prac_with_begin_key_resp.keys',prac_with_begin_key_resp.keys)
    if prac_with_begin_key_resp.keys != None:  # we had a response
        thisExp.addData('prac_with_begin_key_resp.rt', prac_with_begin_key_resp.rt)
        thisExp.addData('prac_with_begin_key_resp.duration', prac_with_begin_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "prac_with_begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    prac_cate = data.TrialHandler2(
        name='prac_cate',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'pics - 4.5.xlsx', 
        selection='0'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(prac_cate)  # add the loop to the experiment
    thisPrac_cate = prac_cate.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPrac_cate.rgb)
    if thisPrac_cate != None:
        for paramName in thisPrac_cate:
            globals()[paramName] = thisPrac_cate[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPrac_cate in prac_cate:
        currentLoop = prac_cate
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_cate.rgb)
        if thisPrac_cate != None:
            for paramName in thisPrac_cate:
                globals()[paramName] = thisPrac_cate[paramName]
        
        # --- Prepare to start Routine "prac_cate_begin" ---
        # create an object to store info about Routine prac_cate_begin
        prac_cate_begin = data.Routine(
            name='prac_cate_begin',
            components=[prac_cate_begin_text, prac_cate_begin_key_resp],
        )
        prac_cate_begin.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for prac_cate_begin_key_resp
        prac_cate_begin_key_resp.keys = []
        prac_cate_begin_key_resp.rt = []
        _prac_cate_begin_key_resp_allKeys = []
        # store start times for prac_cate_begin
        prac_cate_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cate_begin.tStart = globalClock.getTime(format='float')
        prac_cate_begin.status = STARTED
        thisExp.addData('prac_cate_begin.started', prac_cate_begin.tStart)
        prac_cate_begin.maxDuration = None
        # keep track of which components have finished
        prac_cate_beginComponents = prac_cate_begin.components
        for thisComponent in prac_cate_begin.components:
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
        
        # --- Run Routine "prac_cate_begin" ---
        # if trial has changed, end Routine now
        if isinstance(prac_cate, data.TrialHandler2) and thisPrac_cate.thisN != prac_cate.thisTrial.thisN:
            continueRoutine = False
        prac_cate_begin.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_cate_begin_text* updates
            
            # if prac_cate_begin_text is starting this frame...
            if prac_cate_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_begin_text.frameNStart = frameN  # exact frame index
                prac_cate_begin_text.tStart = t  # local t and not account for scr refresh
                prac_cate_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_begin_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_begin_text.started')
                # update status
                prac_cate_begin_text.status = STARTED
                prac_cate_begin_text.setAutoDraw(True)
            
            # if prac_cate_begin_text is active this frame...
            if prac_cate_begin_text.status == STARTED:
                # update params
                pass
            
            # *prac_cate_begin_key_resp* updates
            waitOnFlip = False
            
            # if prac_cate_begin_key_resp is starting this frame...
            if prac_cate_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_begin_key_resp.frameNStart = frameN  # exact frame index
                prac_cate_begin_key_resp.tStart = t  # local t and not account for scr refresh
                prac_cate_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_begin_key_resp.started')
                # update status
                prac_cate_begin_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_cate_begin_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_cate_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_cate_begin_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_cate_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _prac_cate_begin_key_resp_allKeys.extend(theseKeys)
                if len(_prac_cate_begin_key_resp_allKeys):
                    prac_cate_begin_key_resp.keys = _prac_cate_begin_key_resp_allKeys[-1].name  # just the last key pressed
                    prac_cate_begin_key_resp.rt = _prac_cate_begin_key_resp_allKeys[-1].rt
                    prac_cate_begin_key_resp.duration = _prac_cate_begin_key_resp_allKeys[-1].duration
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
                prac_cate_begin.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cate_begin.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cate_begin" ---
        for thisComponent in prac_cate_begin.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cate_begin
        prac_cate_begin.tStop = globalClock.getTime(format='float')
        prac_cate_begin.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_cate_begin.stopped', prac_cate_begin.tStop)
        # check responses
        if prac_cate_begin_key_resp.keys in ['', [], None]:  # No response was made
            prac_cate_begin_key_resp.keys = None
        prac_cate.addData('prac_cate_begin_key_resp.keys',prac_cate_begin_key_resp.keys)
        if prac_cate_begin_key_resp.keys != None:  # we had a response
            prac_cate.addData('prac_cate_begin_key_resp.rt', prac_cate_begin_key_resp.rt)
            prac_cate.addData('prac_cate_begin_key_resp.duration', prac_cate_begin_key_resp.duration)
        # the Routine "prac_cate_begin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_cate_fix" ---
        # create an object to store info about Routine prac_cate_fix
        prac_cate_fix = data.Routine(
            name='prac_cate_fix',
            components=[prac_cate_fix_text, prac_cate_fix_left_calm, prac_cate_fix_right_fear],
        )
        prac_cate_fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for prac_cate_fix
        prac_cate_fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cate_fix.tStart = globalClock.getTime(format='float')
        prac_cate_fix.status = STARTED
        thisExp.addData('prac_cate_fix.started', prac_cate_fix.tStart)
        prac_cate_fix.maxDuration = None
        # keep track of which components have finished
        prac_cate_fixComponents = prac_cate_fix.components
        for thisComponent in prac_cate_fix.components:
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
        
        # --- Run Routine "prac_cate_fix" ---
        # if trial has changed, end Routine now
        if isinstance(prac_cate, data.TrialHandler2) and thisPrac_cate.thisN != prac_cate.thisTrial.thisN:
            continueRoutine = False
        prac_cate_fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_cate_fix_text* updates
            
            # if prac_cate_fix_text is starting this frame...
            if prac_cate_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_fix_text.frameNStart = frameN  # exact frame index
                prac_cate_fix_text.tStart = t  # local t and not account for scr refresh
                prac_cate_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_fix_text.started')
                # update status
                prac_cate_fix_text.status = STARTED
                prac_cate_fix_text.setAutoDraw(True)
            
            # if prac_cate_fix_text is active this frame...
            if prac_cate_fix_text.status == STARTED:
                # update params
                pass
            
            # if prac_cate_fix_text is stopping this frame...
            if prac_cate_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cate_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cate_fix_text.tStop = t  # not accounting for scr refresh
                    prac_cate_fix_text.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cate_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cate_fix_text.stopped')
                    # update status
                    prac_cate_fix_text.status = FINISHED
                    prac_cate_fix_text.setAutoDraw(False)
            
            # *prac_cate_fix_left_calm* updates
            
            # if prac_cate_fix_left_calm is starting this frame...
            if prac_cate_fix_left_calm.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_fix_left_calm.frameNStart = frameN  # exact frame index
                prac_cate_fix_left_calm.tStart = t  # local t and not account for scr refresh
                prac_cate_fix_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_fix_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_fix_left_calm.started')
                # update status
                prac_cate_fix_left_calm.status = STARTED
                prac_cate_fix_left_calm.setAutoDraw(True)
            
            # if prac_cate_fix_left_calm is active this frame...
            if prac_cate_fix_left_calm.status == STARTED:
                # update params
                pass
            
            # if prac_cate_fix_left_calm is stopping this frame...
            if prac_cate_fix_left_calm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cate_fix_left_calm.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cate_fix_left_calm.tStop = t  # not accounting for scr refresh
                    prac_cate_fix_left_calm.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cate_fix_left_calm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cate_fix_left_calm.stopped')
                    # update status
                    prac_cate_fix_left_calm.status = FINISHED
                    prac_cate_fix_left_calm.setAutoDraw(False)
            
            # *prac_cate_fix_right_fear* updates
            
            # if prac_cate_fix_right_fear is starting this frame...
            if prac_cate_fix_right_fear.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_fix_right_fear.frameNStart = frameN  # exact frame index
                prac_cate_fix_right_fear.tStart = t  # local t and not account for scr refresh
                prac_cate_fix_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_fix_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_fix_right_fear.started')
                # update status
                prac_cate_fix_right_fear.status = STARTED
                prac_cate_fix_right_fear.setAutoDraw(True)
            
            # if prac_cate_fix_right_fear is active this frame...
            if prac_cate_fix_right_fear.status == STARTED:
                # update params
                pass
            
            # if prac_cate_fix_right_fear is stopping this frame...
            if prac_cate_fix_right_fear.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cate_fix_right_fear.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cate_fix_right_fear.tStop = t  # not accounting for scr refresh
                    prac_cate_fix_right_fear.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cate_fix_right_fear.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cate_fix_right_fear.stopped')
                    # update status
                    prac_cate_fix_right_fear.status = FINISHED
                    prac_cate_fix_right_fear.setAutoDraw(False)
            
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
                prac_cate_fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cate_fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cate_fix" ---
        for thisComponent in prac_cate_fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cate_fix
        prac_cate_fix.tStop = globalClock.getTime(format='float')
        prac_cate_fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_cate_fix.stopped', prac_cate_fix.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_cate_fix.maxDurationReached:
            routineTimer.addTime(-prac_cate_fix.maxDuration)
        elif prac_cate_fix.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "prac_cate_face" ---
        # create an object to store info about Routine prac_cate_face
        prac_cate_face = data.Routine(
            name='prac_cate_face',
            components=[prac_cate_face_image],
        )
        prac_cate_face.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_cate_face_image.setImage(S2praccate1cont2)
        # store start times for prac_cate_face
        prac_cate_face.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cate_face.tStart = globalClock.getTime(format='float')
        prac_cate_face.status = STARTED
        prac_cate_face.maxDuration = None
        # keep track of which components have finished
        prac_cate_faceComponents = prac_cate_face.components
        for thisComponent in prac_cate_face.components:
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
        
        # --- Run Routine "prac_cate_face" ---
        # if trial has changed, end Routine now
        if isinstance(prac_cate, data.TrialHandler2) and thisPrac_cate.thisN != prac_cate.thisTrial.thisN:
            continueRoutine = False
        prac_cate_face.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_cate_face_image* updates
            
            # if prac_cate_face_image is starting this frame...
            if prac_cate_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_face_image.frameNStart = frameN  # exact frame index
                prac_cate_face_image.tStart = t  # local t and not account for scr refresh
                prac_cate_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_face_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_face_image.started')
                # update status
                prac_cate_face_image.status = STARTED
                prac_cate_face_image.setAutoDraw(True)
            
            # if prac_cate_face_image is active this frame...
            if prac_cate_face_image.status == STARTED:
                # update params
                pass
            
            # if prac_cate_face_image is stopping this frame...
            if prac_cate_face_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cate_face_image.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cate_face_image.tStop = t  # not accounting for scr refresh
                    prac_cate_face_image.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cate_face_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cate_face_image.stopped')
                    # update status
                    prac_cate_face_image.status = FINISHED
                    prac_cate_face_image.setAutoDraw(False)
            
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
                prac_cate_face.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cate_face.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cate_face" ---
        for thisComponent in prac_cate_face.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cate_face
        prac_cate_face.tStop = globalClock.getTime(format='float')
        prac_cate_face.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_cate_face.maxDurationReached:
            routineTimer.addTime(-prac_cate_face.maxDuration)
        elif prac_cate_face.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "prac_cate_resp" ---
        # create an object to store info about Routine prac_cate_resp
        prac_cate_resp = data.Routine(
            name='prac_cate_resp',
            components=[prac_cate_resp_prompt, prac_cate_resp_rect_left, prac_cate_resp_rect_right, prac_cate_resp_left_calm, prac_cate_resp_right_fear, prac_cate_resp_mouse],
        )
        prac_cate_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the prac_cate_resp_mouse
        prac_cate_resp_mouse.x = []
        prac_cate_resp_mouse.y = []
        prac_cate_resp_mouse.leftButton = []
        prac_cate_resp_mouse.midButton = []
        prac_cate_resp_mouse.rightButton = []
        prac_cate_resp_mouse.time = []
        prac_cate_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from prac_cate_resp_code
        # 设置Rectangle的初始状态为不可见
        prac_cate_resp_rect_left.opacity = 0
        prac_cate_resp_rect_right.opacity = 0
        
        # 在您的循环之前，设置鼠标的初始位置（在屏幕中心）
        prac_cate_resp_mouse.setPos((0, 0))
        
        # store start times for prac_cate_resp
        prac_cate_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cate_resp.tStart = globalClock.getTime(format='float')
        prac_cate_resp.status = STARTED
        thisExp.addData('prac_cate_resp.started', prac_cate_resp.tStart)
        prac_cate_resp.maxDuration = None
        # keep track of which components have finished
        prac_cate_respComponents = prac_cate_resp.components
        for thisComponent in prac_cate_resp.components:
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
        
        # --- Run Routine "prac_cate_resp" ---
        # if trial has changed, end Routine now
        if isinstance(prac_cate, data.TrialHandler2) and thisPrac_cate.thisN != prac_cate.thisTrial.thisN:
            continueRoutine = False
        prac_cate_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_cate_resp_prompt* updates
            
            # if prac_cate_resp_prompt is starting this frame...
            if prac_cate_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_resp_prompt.frameNStart = frameN  # exact frame index
                prac_cate_resp_prompt.tStart = t  # local t and not account for scr refresh
                prac_cate_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_resp_prompt.started')
                # update status
                prac_cate_resp_prompt.status = STARTED
                prac_cate_resp_prompt.setAutoDraw(True)
            
            # if prac_cate_resp_prompt is active this frame...
            if prac_cate_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *prac_cate_resp_rect_left* updates
            
            # if prac_cate_resp_rect_left is starting this frame...
            if prac_cate_resp_rect_left.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_resp_rect_left.frameNStart = frameN  # exact frame index
                prac_cate_resp_rect_left.tStart = t  # local t and not account for scr refresh
                prac_cate_resp_rect_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_resp_rect_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_resp_rect_left.started')
                # update status
                prac_cate_resp_rect_left.status = STARTED
                prac_cate_resp_rect_left.setAutoDraw(True)
            
            # if prac_cate_resp_rect_left is active this frame...
            if prac_cate_resp_rect_left.status == STARTED:
                # update params
                pass
            
            # *prac_cate_resp_rect_right* updates
            
            # if prac_cate_resp_rect_right is starting this frame...
            if prac_cate_resp_rect_right.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_resp_rect_right.frameNStart = frameN  # exact frame index
                prac_cate_resp_rect_right.tStart = t  # local t and not account for scr refresh
                prac_cate_resp_rect_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_resp_rect_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_resp_rect_right.started')
                # update status
                prac_cate_resp_rect_right.status = STARTED
                prac_cate_resp_rect_right.setAutoDraw(True)
            
            # if prac_cate_resp_rect_right is active this frame...
            if prac_cate_resp_rect_right.status == STARTED:
                # update params
                pass
            
            # *prac_cate_resp_left_calm* updates
            
            # if prac_cate_resp_left_calm is starting this frame...
            if prac_cate_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_resp_left_calm.frameNStart = frameN  # exact frame index
                prac_cate_resp_left_calm.tStart = t  # local t and not account for scr refresh
                prac_cate_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_resp_left_calm.started')
                # update status
                prac_cate_resp_left_calm.status = STARTED
                prac_cate_resp_left_calm.setAutoDraw(True)
            
            # if prac_cate_resp_left_calm is active this frame...
            if prac_cate_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *prac_cate_resp_right_fear* updates
            
            # if prac_cate_resp_right_fear is starting this frame...
            if prac_cate_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_resp_right_fear.frameNStart = frameN  # exact frame index
                prac_cate_resp_right_fear.tStart = t  # local t and not account for scr refresh
                prac_cate_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cate_resp_right_fear.started')
                # update status
                prac_cate_resp_right_fear.status = STARTED
                prac_cate_resp_right_fear.setAutoDraw(True)
            
            # if prac_cate_resp_right_fear is active this frame...
            if prac_cate_resp_right_fear.status == STARTED:
                # update params
                pass
            # *prac_cate_resp_mouse* updates
            
            # if prac_cate_resp_mouse is starting this frame...
            if prac_cate_resp_mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cate_resp_mouse.frameNStart = frameN  # exact frame index
                prac_cate_resp_mouse.tStart = t  # local t and not account for scr refresh
                prac_cate_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cate_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('prac_cate_resp_mouse.started', t)
                # update status
                prac_cate_resp_mouse.status = STARTED
                prac_cate_resp_mouse.mouseClock.reset()
                prevButtonState = prac_cate_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if prac_cate_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = prac_cate_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([prac_cate_resp_left_calm, prac_cate_resp_right_fear], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(prac_cate_resp_mouse):
                                gotValidClick = True
                                prac_cate_resp_mouse.clicked_name.append(obj.name)
                        if not gotValidClick:
                            prac_cate_resp_mouse.clicked_name.append(None)
                        x, y = prac_cate_resp_mouse.getPos()
                        prac_cate_resp_mouse.x.append(x)
                        prac_cate_resp_mouse.y.append(y)
                        buttons = prac_cate_resp_mouse.getPressed()
                        prac_cate_resp_mouse.leftButton.append(buttons[0])
                        prac_cate_resp_mouse.midButton.append(buttons[1])
                        prac_cate_resp_mouse.rightButton.append(buttons[2])
                        prac_cate_resp_mouse.time.append(prac_cate_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from prac_cate_resp_code
            # 检测鼠标是否悬停在左边单词上方
            if prac_cate_resp_left_calm.contains(prac_cate_resp_mouse):
                prac_cate_resp_rect_left.opacity = 1  # 设置左边Rectangle可见
            else:
                prac_cate_resp_rect_left.opacity = 0  # 设置左边Rectangle不可见
            
            # 检测鼠标是否悬停在右边单词上方
            if prac_cate_resp_right_fear.contains(prac_cate_resp_mouse):
                prac_cate_resp_rect_right.opacity = 1  # 设置右边Rectangle可见
            else:
                prac_cate_resp_rect_right.opacity = 0  # 设置右边Rectangle不可见
            
            
            
            #mouse.setVisible(True)  # 显示鼠标
            win.mouseVisible = True
            
            if prac_cate_resp_mouse.isPressedIn(prac_cate_resp_left_calm, buttons=[0]):
                continueRoutine = False
            elif prac_cate_resp_mouse.isPressedIn(prac_cate_resp_right_fear, buttons=[0]):
                continueRoutine = False
            else:
                continueRoutine = True
            
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
                prac_cate_resp.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cate_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cate_resp" ---
        for thisComponent in prac_cate_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cate_resp
        prac_cate_resp.tStop = globalClock.getTime(format='float')
        prac_cate_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_cate_resp.stopped', prac_cate_resp.tStop)
        # store data for prac_cate (TrialHandler)
        prac_cate.addData('prac_cate_resp_mouse.x', prac_cate_resp_mouse.x)
        prac_cate.addData('prac_cate_resp_mouse.y', prac_cate_resp_mouse.y)
        prac_cate.addData('prac_cate_resp_mouse.leftButton', prac_cate_resp_mouse.leftButton)
        prac_cate.addData('prac_cate_resp_mouse.midButton', prac_cate_resp_mouse.midButton)
        prac_cate.addData('prac_cate_resp_mouse.rightButton', prac_cate_resp_mouse.rightButton)
        prac_cate.addData('prac_cate_resp_mouse.time', prac_cate_resp_mouse.time)
        prac_cate.addData('prac_cate_resp_mouse.clicked_name', prac_cate_resp_mouse.clicked_name)
        # Run 'End Routine' code from prac_cate_resp_code
        win.mouseVisible = False
        # the Routine "prac_cate_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'prac_cate'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "prac_with_begin_2" ---
    # create an object to store info about Routine prac_with_begin_2
    prac_with_begin_2 = data.Routine(
        name='prac_with_begin_2',
        components=[prac_with_begin_2_text, prac_with_begin_2_key_resp],
    )
    prac_with_begin_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for prac_with_begin_2_key_resp
    prac_with_begin_2_key_resp.keys = []
    prac_with_begin_2_key_resp.rt = []
    _prac_with_begin_2_key_resp_allKeys = []
    # store start times for prac_with_begin_2
    prac_with_begin_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prac_with_begin_2.tStart = globalClock.getTime(format='float')
    prac_with_begin_2.status = STARTED
    thisExp.addData('prac_with_begin_2.started', prac_with_begin_2.tStart)
    prac_with_begin_2.maxDuration = None
    # keep track of which components have finished
    prac_with_begin_2Components = prac_with_begin_2.components
    for thisComponent in prac_with_begin_2.components:
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
    
    # --- Run Routine "prac_with_begin_2" ---
    prac_with_begin_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prac_with_begin_2_text* updates
        
        # if prac_with_begin_2_text is starting this frame...
        if prac_with_begin_2_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            prac_with_begin_2_text.frameNStart = frameN  # exact frame index
            prac_with_begin_2_text.tStart = t  # local t and not account for scr refresh
            prac_with_begin_2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_with_begin_2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_with_begin_2_text.started')
            # update status
            prac_with_begin_2_text.status = STARTED
            prac_with_begin_2_text.setAutoDraw(True)
        
        # if prac_with_begin_2_text is active this frame...
        if prac_with_begin_2_text.status == STARTED:
            # update params
            pass
        
        # *prac_with_begin_2_key_resp* updates
        waitOnFlip = False
        
        # if prac_with_begin_2_key_resp is starting this frame...
        if prac_with_begin_2_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_with_begin_2_key_resp.frameNStart = frameN  # exact frame index
            prac_with_begin_2_key_resp.tStart = t  # local t and not account for scr refresh
            prac_with_begin_2_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_with_begin_2_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_with_begin_2_key_resp.started')
            # update status
            prac_with_begin_2_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(prac_with_begin_2_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(prac_with_begin_2_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if prac_with_begin_2_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = prac_with_begin_2_key_resp.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _prac_with_begin_2_key_resp_allKeys.extend(theseKeys)
            if len(_prac_with_begin_2_key_resp_allKeys):
                prac_with_begin_2_key_resp.keys = _prac_with_begin_2_key_resp_allKeys[-1].name  # just the last key pressed
                prac_with_begin_2_key_resp.rt = _prac_with_begin_2_key_resp_allKeys[-1].rt
                prac_with_begin_2_key_resp.duration = _prac_with_begin_2_key_resp_allKeys[-1].duration
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
            prac_with_begin_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_with_begin_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_with_begin_2" ---
    for thisComponent in prac_with_begin_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prac_with_begin_2
    prac_with_begin_2.tStop = globalClock.getTime(format='float')
    prac_with_begin_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prac_with_begin_2.stopped', prac_with_begin_2.tStop)
    # check responses
    if prac_with_begin_2_key_resp.keys in ['', [], None]:  # No response was made
        prac_with_begin_2_key_resp.keys = None
    thisExp.addData('prac_with_begin_2_key_resp.keys',prac_with_begin_2_key_resp.keys)
    if prac_with_begin_2_key_resp.keys != None:  # we had a response
        thisExp.addData('prac_with_begin_2_key_resp.rt', prac_with_begin_2_key_resp.rt)
        thisExp.addData('prac_with_begin_2_key_resp.duration', prac_with_begin_2_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "prac_with_begin_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    prac_cont = data.TrialHandler2(
        name='prac_cont',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'pics - 4.5.xlsx', 
        selection='0'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(prac_cont)  # add the loop to the experiment
    thisPrac_cont = prac_cont.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPrac_cont.rgb)
    if thisPrac_cont != None:
        for paramName in thisPrac_cont:
            globals()[paramName] = thisPrac_cont[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPrac_cont in prac_cont:
        currentLoop = prac_cont
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_cont.rgb)
        if thisPrac_cont != None:
            for paramName in thisPrac_cont:
                globals()[paramName] = thisPrac_cont[paramName]
        
        # --- Prepare to start Routine "prac_cont_begin" ---
        # create an object to store info about Routine prac_cont_begin
        prac_cont_begin = data.Routine(
            name='prac_cont_begin',
            components=[prac_cont_begin_text, prac_cont_begin_key_resp],
        )
        prac_cont_begin.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for prac_cont_begin_key_resp
        prac_cont_begin_key_resp.keys = []
        prac_cont_begin_key_resp.rt = []
        _prac_cont_begin_key_resp_allKeys = []
        # store start times for prac_cont_begin
        prac_cont_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cont_begin.tStart = globalClock.getTime(format='float')
        prac_cont_begin.status = STARTED
        thisExp.addData('prac_cont_begin.started', prac_cont_begin.tStart)
        prac_cont_begin.maxDuration = None
        # keep track of which components have finished
        prac_cont_beginComponents = prac_cont_begin.components
        for thisComponent in prac_cont_begin.components:
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
        
        # --- Run Routine "prac_cont_begin" ---
        # if trial has changed, end Routine now
        if isinstance(prac_cont, data.TrialHandler2) and thisPrac_cont.thisN != prac_cont.thisTrial.thisN:
            continueRoutine = False
        prac_cont_begin.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_cont_begin_text* updates
            
            # if prac_cont_begin_text is starting this frame...
            if prac_cont_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_begin_text.frameNStart = frameN  # exact frame index
                prac_cont_begin_text.tStart = t  # local t and not account for scr refresh
                prac_cont_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_begin_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_begin_text.started')
                # update status
                prac_cont_begin_text.status = STARTED
                prac_cont_begin_text.setAutoDraw(True)
            
            # if prac_cont_begin_text is active this frame...
            if prac_cont_begin_text.status == STARTED:
                # update params
                pass
            
            # *prac_cont_begin_key_resp* updates
            waitOnFlip = False
            
            # if prac_cont_begin_key_resp is starting this frame...
            if prac_cont_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_begin_key_resp.frameNStart = frameN  # exact frame index
                prac_cont_begin_key_resp.tStart = t  # local t and not account for scr refresh
                prac_cont_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_begin_key_resp.started')
                # update status
                prac_cont_begin_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_cont_begin_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_cont_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_cont_begin_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_cont_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _prac_cont_begin_key_resp_allKeys.extend(theseKeys)
                if len(_prac_cont_begin_key_resp_allKeys):
                    prac_cont_begin_key_resp.keys = _prac_cont_begin_key_resp_allKeys[-1].name  # just the last key pressed
                    prac_cont_begin_key_resp.rt = _prac_cont_begin_key_resp_allKeys[-1].rt
                    prac_cont_begin_key_resp.duration = _prac_cont_begin_key_resp_allKeys[-1].duration
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
                prac_cont_begin.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cont_begin.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cont_begin" ---
        for thisComponent in prac_cont_begin.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cont_begin
        prac_cont_begin.tStop = globalClock.getTime(format='float')
        prac_cont_begin.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_cont_begin.stopped', prac_cont_begin.tStop)
        # check responses
        if prac_cont_begin_key_resp.keys in ['', [], None]:  # No response was made
            prac_cont_begin_key_resp.keys = None
        prac_cont.addData('prac_cont_begin_key_resp.keys',prac_cont_begin_key_resp.keys)
        if prac_cont_begin_key_resp.keys != None:  # we had a response
            prac_cont.addData('prac_cont_begin_key_resp.rt', prac_cont_begin_key_resp.rt)
            prac_cont.addData('prac_cont_begin_key_resp.duration', prac_cont_begin_key_resp.duration)
        # the Routine "prac_cont_begin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_cont_fix" ---
        # create an object to store info about Routine prac_cont_fix
        prac_cont_fix = data.Routine(
            name='prac_cont_fix',
            components=[prac_cont_fix_text, prac_cont_fix_left_calm, prac_cont_fix_right_fear, prac_cont_fix_line],
        )
        prac_cont_fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for prac_cont_fix
        prac_cont_fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cont_fix.tStart = globalClock.getTime(format='float')
        prac_cont_fix.status = STARTED
        thisExp.addData('prac_cont_fix.started', prac_cont_fix.tStart)
        prac_cont_fix.maxDuration = None
        # keep track of which components have finished
        prac_cont_fixComponents = prac_cont_fix.components
        for thisComponent in prac_cont_fix.components:
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
        
        # --- Run Routine "prac_cont_fix" ---
        # if trial has changed, end Routine now
        if isinstance(prac_cont, data.TrialHandler2) and thisPrac_cont.thisN != prac_cont.thisTrial.thisN:
            continueRoutine = False
        prac_cont_fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_cont_fix_text* updates
            
            # if prac_cont_fix_text is starting this frame...
            if prac_cont_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_fix_text.frameNStart = frameN  # exact frame index
                prac_cont_fix_text.tStart = t  # local t and not account for scr refresh
                prac_cont_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_fix_text.started')
                # update status
                prac_cont_fix_text.status = STARTED
                prac_cont_fix_text.setAutoDraw(True)
            
            # if prac_cont_fix_text is active this frame...
            if prac_cont_fix_text.status == STARTED:
                # update params
                pass
            
            # if prac_cont_fix_text is stopping this frame...
            if prac_cont_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cont_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cont_fix_text.tStop = t  # not accounting for scr refresh
                    prac_cont_fix_text.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cont_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont_fix_text.stopped')
                    # update status
                    prac_cont_fix_text.status = FINISHED
                    prac_cont_fix_text.setAutoDraw(False)
            
            # *prac_cont_fix_left_calm* updates
            
            # if prac_cont_fix_left_calm is starting this frame...
            if prac_cont_fix_left_calm.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_fix_left_calm.frameNStart = frameN  # exact frame index
                prac_cont_fix_left_calm.tStart = t  # local t and not account for scr refresh
                prac_cont_fix_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_fix_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_fix_left_calm.started')
                # update status
                prac_cont_fix_left_calm.status = STARTED
                prac_cont_fix_left_calm.setAutoDraw(True)
            
            # if prac_cont_fix_left_calm is active this frame...
            if prac_cont_fix_left_calm.status == STARTED:
                # update params
                pass
            
            # if prac_cont_fix_left_calm is stopping this frame...
            if prac_cont_fix_left_calm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cont_fix_left_calm.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cont_fix_left_calm.tStop = t  # not accounting for scr refresh
                    prac_cont_fix_left_calm.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cont_fix_left_calm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont_fix_left_calm.stopped')
                    # update status
                    prac_cont_fix_left_calm.status = FINISHED
                    prac_cont_fix_left_calm.setAutoDraw(False)
            
            # *prac_cont_fix_right_fear* updates
            
            # if prac_cont_fix_right_fear is starting this frame...
            if prac_cont_fix_right_fear.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_fix_right_fear.frameNStart = frameN  # exact frame index
                prac_cont_fix_right_fear.tStart = t  # local t and not account for scr refresh
                prac_cont_fix_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_fix_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_fix_right_fear.started')
                # update status
                prac_cont_fix_right_fear.status = STARTED
                prac_cont_fix_right_fear.setAutoDraw(True)
            
            # if prac_cont_fix_right_fear is active this frame...
            if prac_cont_fix_right_fear.status == STARTED:
                # update params
                pass
            
            # if prac_cont_fix_right_fear is stopping this frame...
            if prac_cont_fix_right_fear.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cont_fix_right_fear.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cont_fix_right_fear.tStop = t  # not accounting for scr refresh
                    prac_cont_fix_right_fear.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cont_fix_right_fear.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont_fix_right_fear.stopped')
                    # update status
                    prac_cont_fix_right_fear.status = FINISHED
                    prac_cont_fix_right_fear.setAutoDraw(False)
            
            # *prac_cont_fix_line* updates
            
            # if prac_cont_fix_line is starting this frame...
            if prac_cont_fix_line.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_fix_line.frameNStart = frameN  # exact frame index
                prac_cont_fix_line.tStart = t  # local t and not account for scr refresh
                prac_cont_fix_line.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_fix_line, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_fix_line.started')
                # update status
                prac_cont_fix_line.status = STARTED
                prac_cont_fix_line.setAutoDraw(True)
            
            # if prac_cont_fix_line is active this frame...
            if prac_cont_fix_line.status == STARTED:
                # update params
                pass
            
            # if prac_cont_fix_line is stopping this frame...
            if prac_cont_fix_line.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cont_fix_line.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cont_fix_line.tStop = t  # not accounting for scr refresh
                    prac_cont_fix_line.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cont_fix_line.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont_fix_line.stopped')
                    # update status
                    prac_cont_fix_line.status = FINISHED
                    prac_cont_fix_line.setAutoDraw(False)
            
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
                prac_cont_fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cont_fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cont_fix" ---
        for thisComponent in prac_cont_fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cont_fix
        prac_cont_fix.tStop = globalClock.getTime(format='float')
        prac_cont_fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_cont_fix.stopped', prac_cont_fix.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_cont_fix.maxDurationReached:
            routineTimer.addTime(-prac_cont_fix.maxDuration)
        elif prac_cont_fix.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "prac_cont_face" ---
        # create an object to store info about Routine prac_cont_face
        prac_cont_face = data.Routine(
            name='prac_cont_face',
            components=[prac_cont_face_image],
        )
        prac_cont_face.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_cont_face_image.setImage(S2praccont1cate2)
        # store start times for prac_cont_face
        prac_cont_face.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cont_face.tStart = globalClock.getTime(format='float')
        prac_cont_face.status = STARTED
        thisExp.addData('prac_cont_face.started', prac_cont_face.tStart)
        prac_cont_face.maxDuration = None
        # keep track of which components have finished
        prac_cont_faceComponents = prac_cont_face.components
        for thisComponent in prac_cont_face.components:
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
        
        # --- Run Routine "prac_cont_face" ---
        # if trial has changed, end Routine now
        if isinstance(prac_cont, data.TrialHandler2) and thisPrac_cont.thisN != prac_cont.thisTrial.thisN:
            continueRoutine = False
        prac_cont_face.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_cont_face_image* updates
            
            # if prac_cont_face_image is starting this frame...
            if prac_cont_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_face_image.frameNStart = frameN  # exact frame index
                prac_cont_face_image.tStart = t  # local t and not account for scr refresh
                prac_cont_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_face_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_face_image.started')
                # update status
                prac_cont_face_image.status = STARTED
                prac_cont_face_image.setAutoDraw(True)
            
            # if prac_cont_face_image is active this frame...
            if prac_cont_face_image.status == STARTED:
                # update params
                pass
            
            # if prac_cont_face_image is stopping this frame...
            if prac_cont_face_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_cont_face_image.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_cont_face_image.tStop = t  # not accounting for scr refresh
                    prac_cont_face_image.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_cont_face_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cont_face_image.stopped')
                    # update status
                    prac_cont_face_image.status = FINISHED
                    prac_cont_face_image.setAutoDraw(False)
            
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
                prac_cont_face.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cont_face.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cont_face" ---
        for thisComponent in prac_cont_face.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cont_face
        prac_cont_face.tStop = globalClock.getTime(format='float')
        prac_cont_face.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_cont_face.stopped', prac_cont_face.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_cont_face.maxDurationReached:
            routineTimer.addTime(-prac_cont_face.maxDuration)
        elif prac_cont_face.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "prac_cont_resp" ---
        # create an object to store info about Routine prac_cont_resp
        prac_cont_resp = data.Routine(
            name='prac_cont_resp',
            components=[prac_cont_resp_prompt, prac_cont_resp_slider_shape, prac_cont_resp_slider, prac_cont_resp_left_calm, prac_cont_resp_right_fear, prac_cont_resp_mouse],
        )
        prac_cont_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_cont_resp_slider.reset()
        # setup some python lists for storing info about the prac_cont_resp_mouse
        prac_cont_resp_mouse.x = []
        prac_cont_resp_mouse.y = []
        prac_cont_resp_mouse.leftButton = []
        prac_cont_resp_mouse.midButton = []
        prac_cont_resp_mouse.rightButton = []
        prac_cont_resp_mouse.time = []
        prac_cont_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from prac_cont_resp_code
        # Change the following parameters according to your own exp
        slider_width = 0.5
        slider_height = .05
        slider_orientation = 0
        #Failing to set ticks or labels in code
        slider_ticks=[0, 1]
        
        slider_granularity = 0
        slider_decimals = 2
        
        prac_cont_resp_slider.decimals = 2
        
        # 获取Slider组件的标记（marker）
        marker = prac_cont_resp_slider.marker
        # 将标记的形状更改为一个特定尺寸的方块
        marker.shape = 'square'
        marker.size = (0.01, 0.05)  # 设置方块的尺寸
        
        # create an empty list to store data
        slider_data=[]
        
        # record the current position of the mouse
        mouseRec=prac_cont_resp_mouse.getPos()
        
        prac_cont_resp_mouse.setPos((0, 0))
        # store start times for prac_cont_resp
        prac_cont_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_cont_resp.tStart = globalClock.getTime(format='float')
        prac_cont_resp.status = STARTED
        thisExp.addData('prac_cont_resp.started', prac_cont_resp.tStart)
        prac_cont_resp.maxDuration = None
        # keep track of which components have finished
        prac_cont_respComponents = prac_cont_resp.components
        for thisComponent in prac_cont_resp.components:
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
        
        # --- Run Routine "prac_cont_resp" ---
        # if trial has changed, end Routine now
        if isinstance(prac_cont, data.TrialHandler2) and thisPrac_cont.thisN != prac_cont.thisTrial.thisN:
            continueRoutine = False
        prac_cont_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_cont_resp_prompt* updates
            
            # if prac_cont_resp_prompt is starting this frame...
            if prac_cont_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_resp_prompt.frameNStart = frameN  # exact frame index
                prac_cont_resp_prompt.tStart = t  # local t and not account for scr refresh
                prac_cont_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_resp_prompt.started')
                # update status
                prac_cont_resp_prompt.status = STARTED
                prac_cont_resp_prompt.setAutoDraw(True)
            
            # if prac_cont_resp_prompt is active this frame...
            if prac_cont_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *prac_cont_resp_slider_shape* updates
            
            # if prac_cont_resp_slider_shape is starting this frame...
            if prac_cont_resp_slider_shape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_resp_slider_shape.frameNStart = frameN  # exact frame index
                prac_cont_resp_slider_shape.tStart = t  # local t and not account for scr refresh
                prac_cont_resp_slider_shape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_resp_slider_shape, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_resp_slider_shape.started')
                # update status
                prac_cont_resp_slider_shape.status = STARTED
                prac_cont_resp_slider_shape.setAutoDraw(True)
            
            # if prac_cont_resp_slider_shape is active this frame...
            if prac_cont_resp_slider_shape.status == STARTED:
                # update params
                pass
            
            # *prac_cont_resp_slider* updates
            
            # if prac_cont_resp_slider is starting this frame...
            if prac_cont_resp_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_resp_slider.frameNStart = frameN  # exact frame index
                prac_cont_resp_slider.tStart = t  # local t and not account for scr refresh
                prac_cont_resp_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_resp_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_resp_slider.started')
                # update status
                prac_cont_resp_slider.status = STARTED
                prac_cont_resp_slider.setAutoDraw(True)
            
            # if prac_cont_resp_slider is active this frame...
            if prac_cont_resp_slider.status == STARTED:
                # update params
                pass
            
            # *prac_cont_resp_left_calm* updates
            
            # if prac_cont_resp_left_calm is starting this frame...
            if prac_cont_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_resp_left_calm.frameNStart = frameN  # exact frame index
                prac_cont_resp_left_calm.tStart = t  # local t and not account for scr refresh
                prac_cont_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_resp_left_calm.started')
                # update status
                prac_cont_resp_left_calm.status = STARTED
                prac_cont_resp_left_calm.setAutoDraw(True)
            
            # if prac_cont_resp_left_calm is active this frame...
            if prac_cont_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *prac_cont_resp_right_fear* updates
            
            # if prac_cont_resp_right_fear is starting this frame...
            if prac_cont_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_resp_right_fear.frameNStart = frameN  # exact frame index
                prac_cont_resp_right_fear.tStart = t  # local t and not account for scr refresh
                prac_cont_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_cont_resp_right_fear.started')
                # update status
                prac_cont_resp_right_fear.status = STARTED
                prac_cont_resp_right_fear.setAutoDraw(True)
            
            # if prac_cont_resp_right_fear is active this frame...
            if prac_cont_resp_right_fear.status == STARTED:
                # update params
                pass
            # *prac_cont_resp_mouse* updates
            
            # if prac_cont_resp_mouse is starting this frame...
            if prac_cont_resp_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_cont_resp_mouse.frameNStart = frameN  # exact frame index
                prac_cont_resp_mouse.tStart = t  # local t and not account for scr refresh
                prac_cont_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_cont_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('prac_cont_resp_mouse.started', t)
                # update status
                prac_cont_resp_mouse.status = STARTED
                prac_cont_resp_mouse.mouseClock.reset()
                prevButtonState = prac_cont_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if prac_cont_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = prac_cont_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(prac_cont_resp_slider_shape, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(prac_cont_resp_mouse):
                                gotValidClick = True
                                prac_cont_resp_mouse.clicked_name.append(obj.name)
                        if not gotValidClick:
                            prac_cont_resp_mouse.clicked_name.append(None)
                        x, y = prac_cont_resp_mouse.getPos()
                        prac_cont_resp_mouse.x.append(x)
                        prac_cont_resp_mouse.y.append(y)
                        buttons = prac_cont_resp_mouse.getPressed()
                        prac_cont_resp_mouse.leftButton.append(buttons[0])
                        prac_cont_resp_mouse.midButton.append(buttons[1])
                        prac_cont_resp_mouse.rightButton.append(buttons[2])
                        prac_cont_resp_mouse.time.append(prac_cont_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from prac_cont_resp_code
            
            # Create a slider_shape in this routine with the same size as the slider
            # Check if mouse is moving within slider_shape, if so change marker position accordingly
            if prac_cont_resp_slider_shape.contains(prac_cont_resp_mouse) and prac_cont_resp_mouse.getPos()[slider_orientation] != mouseRec[slider_orientation]:
                mouseRec = prac_cont_resp_mouse.getPos()
                prac_cont_resp_slider.markerPos = mouseRec[slider_orientation]/slider_width*(slider_ticks[-1]-slider_ticks[0])+(slider_ticks[0]+slider_ticks[-1])/2
                
                # (new)Record marker position only on left mouse click
                slider_data.append([round(prac_cont_resp_slider.markerPos, slider_decimals)])
            
            
            win.mouseVisible = True
            
            #只有左键结束
            if prac_cont_resp_mouse.isPressedIn(prac_cont_resp_slider_shape, buttons=[0]):
                continueRoutine = False
            
            else:
                continueRoutine = True
            
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
                prac_cont_resp.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_cont_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_cont_resp" ---
        for thisComponent in prac_cont_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_cont_resp
        prac_cont_resp.tStop = globalClock.getTime(format='float')
        prac_cont_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_cont_resp.stopped', prac_cont_resp.tStop)
        prac_cont.addData('prac_cont_resp_slider.response', prac_cont_resp_slider.getRating())
        prac_cont.addData('prac_cont_resp_slider.rt', prac_cont_resp_slider.getRT())
        # store data for prac_cont (TrialHandler)
        prac_cont.addData('prac_cont_resp_mouse.x', prac_cont_resp_mouse.x)
        prac_cont.addData('prac_cont_resp_mouse.y', prac_cont_resp_mouse.y)
        prac_cont.addData('prac_cont_resp_mouse.leftButton', prac_cont_resp_mouse.leftButton)
        prac_cont.addData('prac_cont_resp_mouse.midButton', prac_cont_resp_mouse.midButton)
        prac_cont.addData('prac_cont_resp_mouse.rightButton', prac_cont_resp_mouse.rightButton)
        prac_cont.addData('prac_cont_resp_mouse.time', prac_cont_resp_mouse.time)
        prac_cont.addData('prac_cont_resp_mouse.clicked_name', prac_cont_resp_mouse.clicked_name)
        # Run 'End Routine' code from prac_cont_resp_code
        win.mouseVisible = False
        # the Routine "prac_cont_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'prac_cont'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "prac_end" ---
    # create an object to store info about Routine prac_end
    prac_end = data.Routine(
        name='prac_end',
        components=[prac_end_text, prac_end_key_resp],
    )
    prac_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for prac_end_key_resp
    prac_end_key_resp.keys = []
    prac_end_key_resp.rt = []
    _prac_end_key_resp_allKeys = []
    # store start times for prac_end
    prac_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prac_end.tStart = globalClock.getTime(format='float')
    prac_end.status = STARTED
    thisExp.addData('prac_end.started', prac_end.tStart)
    prac_end.maxDuration = None
    # keep track of which components have finished
    prac_endComponents = prac_end.components
    for thisComponent in prac_end.components:
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
    
    # --- Run Routine "prac_end" ---
    prac_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prac_end_text* updates
        
        # if prac_end_text is starting this frame...
        if prac_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_end_text.frameNStart = frameN  # exact frame index
            prac_end_text.tStart = t  # local t and not account for scr refresh
            prac_end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_end_text.started')
            # update status
            prac_end_text.status = STARTED
            prac_end_text.setAutoDraw(True)
        
        # if prac_end_text is active this frame...
        if prac_end_text.status == STARTED:
            # update params
            pass
        
        # *prac_end_key_resp* updates
        waitOnFlip = False
        
        # if prac_end_key_resp is starting this frame...
        if prac_end_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_end_key_resp.frameNStart = frameN  # exact frame index
            prac_end_key_resp.tStart = t  # local t and not account for scr refresh
            prac_end_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_end_key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            prac_end_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(prac_end_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(prac_end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if prac_end_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = prac_end_key_resp.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _prac_end_key_resp_allKeys.extend(theseKeys)
            if len(_prac_end_key_resp_allKeys):
                prac_end_key_resp.keys = _prac_end_key_resp_allKeys[-1].name  # just the last key pressed
                prac_end_key_resp.rt = _prac_end_key_resp_allKeys[-1].rt
                prac_end_key_resp.duration = _prac_end_key_resp_allKeys[-1].duration
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
            prac_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_end" ---
    for thisComponent in prac_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prac_end
    prac_end.tStop = globalClock.getTime(format='float')
    prac_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prac_end.stopped', prac_end.tStop)
    # check responses
    if prac_end_key_resp.keys in ['', [], None]:  # No response was made
        prac_end_key_resp.keys = None
    thisExp.addData('prac_end_key_resp.keys',prac_end_key_resp.keys)
    if prac_end_key_resp.keys != None:  # we had a response
        thisExp.addData('prac_end_key_resp.rt', prac_end_key_resp.rt)
        thisExp.addData('prac_end_key_resp.duration', prac_end_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "prac_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal_cate = data.TrialHandler2(
        name='formal_cate',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'pics - 4.5.xlsx', 
        selection='0'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(formal_cate)  # add the loop to the experiment
    thisFormal_cate = formal_cate.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFormal_cate.rgb)
    if thisFormal_cate != None:
        for paramName in thisFormal_cate:
            globals()[paramName] = thisFormal_cate[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFormal_cate in formal_cate:
        currentLoop = formal_cate
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFormal_cate.rgb)
        if thisFormal_cate != None:
            for paramName in thisFormal_cate:
                globals()[paramName] = thisFormal_cate[paramName]
        
        # --- Prepare to start Routine "formal_cate_begin" ---
        # create an object to store info about Routine formal_cate_begin
        formal_cate_begin = data.Routine(
            name='formal_cate_begin',
            components=[formal_cate_begin_text, formal_cate_begin_key_resp],
        )
        formal_cate_begin.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for formal_cate_begin_key_resp
        formal_cate_begin_key_resp.keys = []
        formal_cate_begin_key_resp.rt = []
        _formal_cate_begin_key_resp_allKeys = []
        # store start times for formal_cate_begin
        formal_cate_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_cate_begin.tStart = globalClock.getTime(format='float')
        formal_cate_begin.status = STARTED
        thisExp.addData('formal_cate_begin.started', formal_cate_begin.tStart)
        formal_cate_begin.maxDuration = None
        # keep track of which components have finished
        formal_cate_beginComponents = formal_cate_begin.components
        for thisComponent in formal_cate_begin.components:
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
        
        # --- Run Routine "formal_cate_begin" ---
        # if trial has changed, end Routine now
        if isinstance(formal_cate, data.TrialHandler2) and thisFormal_cate.thisN != formal_cate.thisTrial.thisN:
            continueRoutine = False
        formal_cate_begin.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_cate_begin_text* updates
            
            # if formal_cate_begin_text is starting this frame...
            if formal_cate_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_begin_text.frameNStart = frameN  # exact frame index
                formal_cate_begin_text.tStart = t  # local t and not account for scr refresh
                formal_cate_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_begin_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_begin_text.started')
                # update status
                formal_cate_begin_text.status = STARTED
                formal_cate_begin_text.setAutoDraw(True)
            
            # if formal_cate_begin_text is active this frame...
            if formal_cate_begin_text.status == STARTED:
                # update params
                pass
            
            # *formal_cate_begin_key_resp* updates
            waitOnFlip = False
            
            # if formal_cate_begin_key_resp is starting this frame...
            if formal_cate_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_begin_key_resp.frameNStart = frameN  # exact frame index
                formal_cate_begin_key_resp.tStart = t  # local t and not account for scr refresh
                formal_cate_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_begin_key_resp.started')
                # update status
                formal_cate_begin_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(formal_cate_begin_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(formal_cate_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if formal_cate_begin_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = formal_cate_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _formal_cate_begin_key_resp_allKeys.extend(theseKeys)
                if len(_formal_cate_begin_key_resp_allKeys):
                    formal_cate_begin_key_resp.keys = _formal_cate_begin_key_resp_allKeys[-1].name  # just the last key pressed
                    formal_cate_begin_key_resp.rt = _formal_cate_begin_key_resp_allKeys[-1].rt
                    formal_cate_begin_key_resp.duration = _formal_cate_begin_key_resp_allKeys[-1].duration
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
                formal_cate_begin.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_cate_begin.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_cate_begin" ---
        for thisComponent in formal_cate_begin.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_cate_begin
        formal_cate_begin.tStop = globalClock.getTime(format='float')
        formal_cate_begin.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_cate_begin.stopped', formal_cate_begin.tStop)
        # check responses
        if formal_cate_begin_key_resp.keys in ['', [], None]:  # No response was made
            formal_cate_begin_key_resp.keys = None
        formal_cate.addData('formal_cate_begin_key_resp.keys',formal_cate_begin_key_resp.keys)
        if formal_cate_begin_key_resp.keys != None:  # we had a response
            formal_cate.addData('formal_cate_begin_key_resp.rt', formal_cate_begin_key_resp.rt)
            formal_cate.addData('formal_cate_begin_key_resp.duration', formal_cate_begin_key_resp.duration)
        # the Routine "formal_cate_begin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "formal_cate_fix" ---
        # create an object to store info about Routine formal_cate_fix
        formal_cate_fix = data.Routine(
            name='formal_cate_fix',
            components=[formal_cate_fix_text, formal_cate_fix_left_calm, formal_cate_fix_right_fear],
        )
        formal_cate_fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal_cate_fix_Init_trial
        block_name = 'categorical'
        
        # store start times for formal_cate_fix
        formal_cate_fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_cate_fix.tStart = globalClock.getTime(format='float')
        formal_cate_fix.status = STARTED
        thisExp.addData('formal_cate_fix.started', formal_cate_fix.tStart)
        formal_cate_fix.maxDuration = None
        # keep track of which components have finished
        formal_cate_fixComponents = formal_cate_fix.components
        for thisComponent in formal_cate_fix.components:
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
        
        # --- Run Routine "formal_cate_fix" ---
        # if trial has changed, end Routine now
        if isinstance(formal_cate, data.TrialHandler2) and thisFormal_cate.thisN != formal_cate.thisTrial.thisN:
            continueRoutine = False
        formal_cate_fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_cate_fix_text* updates
            
            # if formal_cate_fix_text is starting this frame...
            if formal_cate_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_fix_text.frameNStart = frameN  # exact frame index
                formal_cate_fix_text.tStart = t  # local t and not account for scr refresh
                formal_cate_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_fix_text.started')
                # update status
                formal_cate_fix_text.status = STARTED
                formal_cate_fix_text.setAutoDraw(True)
            
            # if formal_cate_fix_text is active this frame...
            if formal_cate_fix_text.status == STARTED:
                # update params
                pass
            
            # if formal_cate_fix_text is stopping this frame...
            if formal_cate_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cate_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cate_fix_text.tStop = t  # not accounting for scr refresh
                    formal_cate_fix_text.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cate_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cate_fix_text.stopped')
                    # update status
                    formal_cate_fix_text.status = FINISHED
                    formal_cate_fix_text.setAutoDraw(False)
            
            # *formal_cate_fix_left_calm* updates
            
            # if formal_cate_fix_left_calm is starting this frame...
            if formal_cate_fix_left_calm.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_fix_left_calm.frameNStart = frameN  # exact frame index
                formal_cate_fix_left_calm.tStart = t  # local t and not account for scr refresh
                formal_cate_fix_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_fix_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_fix_left_calm.started')
                # update status
                formal_cate_fix_left_calm.status = STARTED
                formal_cate_fix_left_calm.setAutoDraw(True)
            
            # if formal_cate_fix_left_calm is active this frame...
            if formal_cate_fix_left_calm.status == STARTED:
                # update params
                pass
            
            # if formal_cate_fix_left_calm is stopping this frame...
            if formal_cate_fix_left_calm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cate_fix_left_calm.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cate_fix_left_calm.tStop = t  # not accounting for scr refresh
                    formal_cate_fix_left_calm.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cate_fix_left_calm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cate_fix_left_calm.stopped')
                    # update status
                    formal_cate_fix_left_calm.status = FINISHED
                    formal_cate_fix_left_calm.setAutoDraw(False)
            
            # *formal_cate_fix_right_fear* updates
            
            # if formal_cate_fix_right_fear is starting this frame...
            if formal_cate_fix_right_fear.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_fix_right_fear.frameNStart = frameN  # exact frame index
                formal_cate_fix_right_fear.tStart = t  # local t and not account for scr refresh
                formal_cate_fix_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_fix_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_fix_right_fear.started')
                # update status
                formal_cate_fix_right_fear.status = STARTED
                formal_cate_fix_right_fear.setAutoDraw(True)
            
            # if formal_cate_fix_right_fear is active this frame...
            if formal_cate_fix_right_fear.status == STARTED:
                # update params
                pass
            
            # if formal_cate_fix_right_fear is stopping this frame...
            if formal_cate_fix_right_fear.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cate_fix_right_fear.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cate_fix_right_fear.tStop = t  # not accounting for scr refresh
                    formal_cate_fix_right_fear.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cate_fix_right_fear.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cate_fix_right_fear.stopped')
                    # update status
                    formal_cate_fix_right_fear.status = FINISHED
                    formal_cate_fix_right_fear.setAutoDraw(False)
            
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
                formal_cate_fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_cate_fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_cate_fix" ---
        for thisComponent in formal_cate_fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_cate_fix
        formal_cate_fix.tStop = globalClock.getTime(format='float')
        formal_cate_fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_cate_fix.stopped', formal_cate_fix.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal_cate_fix.maxDurationReached:
            routineTimer.addTime(-formal_cate_fix.maxDuration)
        elif formal_cate_fix.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "formal_cate_face" ---
        # create an object to store info about Routine formal_cate_face
        formal_cate_face = data.Routine(
            name='formal_cate_face',
            components=[formal_cate_face_image],
        )
        formal_cate_face.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        formal_cate_face_image.setImage(S2cate1cont2)
        # store start times for formal_cate_face
        formal_cate_face.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_cate_face.tStart = globalClock.getTime(format='float')
        formal_cate_face.status = STARTED
        formal_cate_face.maxDuration = None
        # keep track of which components have finished
        formal_cate_faceComponents = formal_cate_face.components
        for thisComponent in formal_cate_face.components:
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
        
        # --- Run Routine "formal_cate_face" ---
        # if trial has changed, end Routine now
        if isinstance(formal_cate, data.TrialHandler2) and thisFormal_cate.thisN != formal_cate.thisTrial.thisN:
            continueRoutine = False
        formal_cate_face.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_cate_face_image* updates
            
            # if formal_cate_face_image is starting this frame...
            if formal_cate_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_face_image.frameNStart = frameN  # exact frame index
                formal_cate_face_image.tStart = t  # local t and not account for scr refresh
                formal_cate_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_face_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_face_image.started')
                # update status
                formal_cate_face_image.status = STARTED
                formal_cate_face_image.setAutoDraw(True)
            
            # if formal_cate_face_image is active this frame...
            if formal_cate_face_image.status == STARTED:
                # update params
                pass
            
            # if formal_cate_face_image is stopping this frame...
            if formal_cate_face_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cate_face_image.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cate_face_image.tStop = t  # not accounting for scr refresh
                    formal_cate_face_image.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cate_face_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cate_face_image.stopped')
                    # update status
                    formal_cate_face_image.status = FINISHED
                    formal_cate_face_image.setAutoDraw(False)
            
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
                formal_cate_face.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_cate_face.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_cate_face" ---
        for thisComponent in formal_cate_face.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_cate_face
        formal_cate_face.tStop = globalClock.getTime(format='float')
        formal_cate_face.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal_cate_face.maxDurationReached:
            routineTimer.addTime(-formal_cate_face.maxDuration)
        elif formal_cate_face.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "formal_cate_resp" ---
        # create an object to store info about Routine formal_cate_resp
        formal_cate_resp = data.Routine(
            name='formal_cate_resp',
            components=[formal_cate_resp_prompt, formal_cate_resp_rect_left, formal_cate_resp_rect_right, formal_cate_resp_left_calm, formal_cate_resp_right_fear, formal_cate_resp_mouse],
        )
        formal_cate_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the formal_cate_resp_mouse
        formal_cate_resp_mouse.x = []
        formal_cate_resp_mouse.y = []
        formal_cate_resp_mouse.leftButton = []
        formal_cate_resp_mouse.midButton = []
        formal_cate_resp_mouse.rightButton = []
        formal_cate_resp_mouse.time = []
        formal_cate_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        formal_cate_resp_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from formal_cate_resp_code
        # 设置Rectangle的初始状态为不可见
        formal_cate_resp_rect_left.opacity = 0
        formal_cate_resp_rect_right.opacity = 0
        
        formal_cate_resp_mouse.setPos((0, 0))
        # store start times for formal_cate_resp
        formal_cate_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_cate_resp.tStart = globalClock.getTime(format='float')
        formal_cate_resp.status = STARTED
        thisExp.addData('formal_cate_resp.started', formal_cate_resp.tStart)
        formal_cate_resp.maxDuration = None
        # keep track of which components have finished
        formal_cate_respComponents = formal_cate_resp.components
        for thisComponent in formal_cate_resp.components:
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
        
        # --- Run Routine "formal_cate_resp" ---
        # if trial has changed, end Routine now
        if isinstance(formal_cate, data.TrialHandler2) and thisFormal_cate.thisN != formal_cate.thisTrial.thisN:
            continueRoutine = False
        formal_cate_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_cate_resp_prompt* updates
            
            # if formal_cate_resp_prompt is starting this frame...
            if formal_cate_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_resp_prompt.frameNStart = frameN  # exact frame index
                formal_cate_resp_prompt.tStart = t  # local t and not account for scr refresh
                formal_cate_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_resp_prompt.started')
                # update status
                formal_cate_resp_prompt.status = STARTED
                formal_cate_resp_prompt.setAutoDraw(True)
            
            # if formal_cate_resp_prompt is active this frame...
            if formal_cate_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *formal_cate_resp_rect_left* updates
            
            # if formal_cate_resp_rect_left is starting this frame...
            if formal_cate_resp_rect_left.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_resp_rect_left.frameNStart = frameN  # exact frame index
                formal_cate_resp_rect_left.tStart = t  # local t and not account for scr refresh
                formal_cate_resp_rect_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_resp_rect_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_resp_rect_left.started')
                # update status
                formal_cate_resp_rect_left.status = STARTED
                formal_cate_resp_rect_left.setAutoDraw(True)
            
            # if formal_cate_resp_rect_left is active this frame...
            if formal_cate_resp_rect_left.status == STARTED:
                # update params
                pass
            
            # *formal_cate_resp_rect_right* updates
            
            # if formal_cate_resp_rect_right is starting this frame...
            if formal_cate_resp_rect_right.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_resp_rect_right.frameNStart = frameN  # exact frame index
                formal_cate_resp_rect_right.tStart = t  # local t and not account for scr refresh
                formal_cate_resp_rect_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_resp_rect_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_resp_rect_right.started')
                # update status
                formal_cate_resp_rect_right.status = STARTED
                formal_cate_resp_rect_right.setAutoDraw(True)
            
            # if formal_cate_resp_rect_right is active this frame...
            if formal_cate_resp_rect_right.status == STARTED:
                # update params
                pass
            
            # *formal_cate_resp_left_calm* updates
            
            # if formal_cate_resp_left_calm is starting this frame...
            if formal_cate_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_resp_left_calm.frameNStart = frameN  # exact frame index
                formal_cate_resp_left_calm.tStart = t  # local t and not account for scr refresh
                formal_cate_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_resp_left_calm.started')
                # update status
                formal_cate_resp_left_calm.status = STARTED
                formal_cate_resp_left_calm.setAutoDraw(True)
            
            # if formal_cate_resp_left_calm is active this frame...
            if formal_cate_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *formal_cate_resp_right_fear* updates
            
            # if formal_cate_resp_right_fear is starting this frame...
            if formal_cate_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_resp_right_fear.frameNStart = frameN  # exact frame index
                formal_cate_resp_right_fear.tStart = t  # local t and not account for scr refresh
                formal_cate_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_resp_right_fear.started')
                # update status
                formal_cate_resp_right_fear.status = STARTED
                formal_cate_resp_right_fear.setAutoDraw(True)
            
            # if formal_cate_resp_right_fear is active this frame...
            if formal_cate_resp_right_fear.status == STARTED:
                # update params
                pass
            # *formal_cate_resp_mouse* updates
            
            # if formal_cate_resp_mouse is starting this frame...
            if formal_cate_resp_mouse.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cate_resp_mouse.frameNStart = frameN  # exact frame index
                formal_cate_resp_mouse.tStart = t  # local t and not account for scr refresh
                formal_cate_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cate_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cate_resp_mouse.started')
                # update status
                formal_cate_resp_mouse.status = STARTED
                prevButtonState = formal_cate_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if formal_cate_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = formal_cate_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([formal_cate_resp_left_calm, formal_cate_resp_right_fear], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(formal_cate_resp_mouse):
                                gotValidClick = True
                                formal_cate_resp_mouse.clicked_name.append(obj.name)
                                formal_cate_resp_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = formal_cate_resp_mouse.getPos()
                            formal_cate_resp_mouse.x.append(x)
                            formal_cate_resp_mouse.y.append(y)
                            buttons = formal_cate_resp_mouse.getPressed()
                            formal_cate_resp_mouse.leftButton.append(buttons[0])
                            formal_cate_resp_mouse.midButton.append(buttons[1])
                            formal_cate_resp_mouse.rightButton.append(buttons[2])
                            formal_cate_resp_mouse.time.append(formal_cate_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from formal_cate_resp_code
            # 检测鼠标是否悬停在左边单词上方
            if formal_cate_resp_left_calm.contains(formal_cate_resp_mouse):
                formal_cate_resp_rect_left.opacity = 1  # 设置左边Rectangle可见
            else:
                formal_cate_resp_rect_left.opacity = 0  # 设置左边Rectangle不可见
            
            # 检测鼠标是否悬停在右边单词上方
            if formal_cate_resp_right_fear.contains(formal_cate_resp_mouse):
                formal_cate_resp_rect_right.opacity = 1  # 设置右边Rectangle可见
            else:
                formal_cate_resp_rect_right.opacity = 0  # 设置右边Rectangle不可见
            
            # 检测鼠标点击事件
            if formal_cate_resp_mouse.getPressed()[0]:
                # 检测鼠标点击是否在左边单词上方
                if formal_cate_resp_left_calm.contains(formal_cate_resp_mouse):
                    cate_resp = 0  # 记录被试选择左边单词
                    continueRoutine = False
            
                # 检测鼠标点击是否在右边单词上方
                elif formal_cate_resp_right_fear.contains(formal_cate_resp_mouse):
                    cate_resp = 1  # 记录被试选择右边单词
                    continueRoutine = False
            
                else:
                    cate_resp = 'None'
                    continueRoutine = True
            else:
                cate_resp = 'None'
                continueRoutine = True
                
            win.mouseVisible = True
            
            
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
                formal_cate_resp.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_cate_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_cate_resp" ---
        for thisComponent in formal_cate_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_cate_resp
        formal_cate_resp.tStop = globalClock.getTime(format='float')
        formal_cate_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_cate_resp.stopped', formal_cate_resp.tStop)
        # store data for formal_cate (TrialHandler)
        formal_cate.addData('formal_cate_resp_mouse.x', formal_cate_resp_mouse.x)
        formal_cate.addData('formal_cate_resp_mouse.y', formal_cate_resp_mouse.y)
        formal_cate.addData('formal_cate_resp_mouse.leftButton', formal_cate_resp_mouse.leftButton)
        formal_cate.addData('formal_cate_resp_mouse.midButton', formal_cate_resp_mouse.midButton)
        formal_cate.addData('formal_cate_resp_mouse.rightButton', formal_cate_resp_mouse.rightButton)
        formal_cate.addData('formal_cate_resp_mouse.time', formal_cate_resp_mouse.time)
        formal_cate.addData('formal_cate_resp_mouse.clicked_name', formal_cate_resp_mouse.clicked_name)
        # Run 'End Routine' code from formal_cate_resp_code
        trialID = trialID + 1
        
        thisExp.addData('block_name', block_name)
        thisExp.addData('trialID', trialID)
        thisExp.addData('face_stim', S2cate1cont2)
        thisExp.addData('cate_resp', cate_resp)
        
        win.mouseVisible = False
        
        # the Routine "formal_cate_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'formal_cate'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "brea" ---
    # create an object to store info about Routine brea
    brea = data.Routine(
        name='brea',
        components=[brea_text, brea_key_resp],
    )
    brea.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for brea_key_resp
    brea_key_resp.keys = []
    brea_key_resp.rt = []
    _brea_key_resp_allKeys = []
    # store start times for brea
    brea.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    brea.tStart = globalClock.getTime(format='float')
    brea.status = STARTED
    thisExp.addData('brea.started', brea.tStart)
    brea.maxDuration = None
    # keep track of which components have finished
    breaComponents = brea.components
    for thisComponent in brea.components:
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
    
    # --- Run Routine "brea" ---
    brea.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *brea_text* updates
        
        # if brea_text is starting this frame...
        if brea_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            brea_text.frameNStart = frameN  # exact frame index
            brea_text.tStart = t  # local t and not account for scr refresh
            brea_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(brea_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'brea_text.started')
            # update status
            brea_text.status = STARTED
            brea_text.setAutoDraw(True)
        
        # if brea_text is active this frame...
        if brea_text.status == STARTED:
            # update params
            pass
        
        # *brea_key_resp* updates
        waitOnFlip = False
        
        # if brea_key_resp is starting this frame...
        if brea_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            brea_key_resp.frameNStart = frameN  # exact frame index
            brea_key_resp.tStart = t  # local t and not account for scr refresh
            brea_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(brea_key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            brea_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(brea_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(brea_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if brea_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = brea_key_resp.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _brea_key_resp_allKeys.extend(theseKeys)
            if len(_brea_key_resp_allKeys):
                brea_key_resp.keys = _brea_key_resp_allKeys[-1].name  # just the last key pressed
                brea_key_resp.rt = _brea_key_resp_allKeys[-1].rt
                brea_key_resp.duration = _brea_key_resp_allKeys[-1].duration
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
            brea.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in brea.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "brea" ---
    for thisComponent in brea.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for brea
    brea.tStop = globalClock.getTime(format='float')
    brea.tStopRefresh = tThisFlipGlobal
    thisExp.addData('brea.stopped', brea.tStop)
    # check responses
    if brea_key_resp.keys in ['', [], None]:  # No response was made
        brea_key_resp.keys = None
    thisExp.addData('brea_key_resp.keys',brea_key_resp.keys)
    if brea_key_resp.keys != None:  # we had a response
        thisExp.addData('brea_key_resp.rt', brea_key_resp.rt)
        thisExp.addData('brea_key_resp.duration', brea_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "brea" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal_cont = data.TrialHandler2(
        name='formal_cont',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'pics - 4.5.xlsx', 
        selection='0:50'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(formal_cont)  # add the loop to the experiment
    thisFormal_cont = formal_cont.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFormal_cont.rgb)
    if thisFormal_cont != None:
        for paramName in thisFormal_cont:
            globals()[paramName] = thisFormal_cont[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFormal_cont in formal_cont:
        currentLoop = formal_cont
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFormal_cont.rgb)
        if thisFormal_cont != None:
            for paramName in thisFormal_cont:
                globals()[paramName] = thisFormal_cont[paramName]
        
        # --- Prepare to start Routine "formal_cont_begin" ---
        # create an object to store info about Routine formal_cont_begin
        formal_cont_begin = data.Routine(
            name='formal_cont_begin',
            components=[formal_cont_begin_text, formal_cont_begin_key_resp],
        )
        formal_cont_begin.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for formal_cont_begin_key_resp
        formal_cont_begin_key_resp.keys = []
        formal_cont_begin_key_resp.rt = []
        _formal_cont_begin_key_resp_allKeys = []
        # store start times for formal_cont_begin
        formal_cont_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_cont_begin.tStart = globalClock.getTime(format='float')
        formal_cont_begin.status = STARTED
        thisExp.addData('formal_cont_begin.started', formal_cont_begin.tStart)
        formal_cont_begin.maxDuration = None
        # keep track of which components have finished
        formal_cont_beginComponents = formal_cont_begin.components
        for thisComponent in formal_cont_begin.components:
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
        
        # --- Run Routine "formal_cont_begin" ---
        # if trial has changed, end Routine now
        if isinstance(formal_cont, data.TrialHandler2) and thisFormal_cont.thisN != formal_cont.thisTrial.thisN:
            continueRoutine = False
        formal_cont_begin.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_cont_begin_text* updates
            
            # if formal_cont_begin_text is starting this frame...
            if formal_cont_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_begin_text.frameNStart = frameN  # exact frame index
                formal_cont_begin_text.tStart = t  # local t and not account for scr refresh
                formal_cont_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_begin_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_begin_text.started')
                # update status
                formal_cont_begin_text.status = STARTED
                formal_cont_begin_text.setAutoDraw(True)
            
            # if formal_cont_begin_text is active this frame...
            if formal_cont_begin_text.status == STARTED:
                # update params
                pass
            
            # *formal_cont_begin_key_resp* updates
            waitOnFlip = False
            
            # if formal_cont_begin_key_resp is starting this frame...
            if formal_cont_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_begin_key_resp.frameNStart = frameN  # exact frame index
                formal_cont_begin_key_resp.tStart = t  # local t and not account for scr refresh
                formal_cont_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_begin_key_resp.started')
                # update status
                formal_cont_begin_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(formal_cont_begin_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(formal_cont_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if formal_cont_begin_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = formal_cont_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _formal_cont_begin_key_resp_allKeys.extend(theseKeys)
                if len(_formal_cont_begin_key_resp_allKeys):
                    formal_cont_begin_key_resp.keys = _formal_cont_begin_key_resp_allKeys[-1].name  # just the last key pressed
                    formal_cont_begin_key_resp.rt = _formal_cont_begin_key_resp_allKeys[-1].rt
                    formal_cont_begin_key_resp.duration = _formal_cont_begin_key_resp_allKeys[-1].duration
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
                formal_cont_begin.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_cont_begin.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_cont_begin" ---
        for thisComponent in formal_cont_begin.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_cont_begin
        formal_cont_begin.tStop = globalClock.getTime(format='float')
        formal_cont_begin.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_cont_begin.stopped', formal_cont_begin.tStop)
        # check responses
        if formal_cont_begin_key_resp.keys in ['', [], None]:  # No response was made
            formal_cont_begin_key_resp.keys = None
        formal_cont.addData('formal_cont_begin_key_resp.keys',formal_cont_begin_key_resp.keys)
        if formal_cont_begin_key_resp.keys != None:  # we had a response
            formal_cont.addData('formal_cont_begin_key_resp.rt', formal_cont_begin_key_resp.rt)
            formal_cont.addData('formal_cont_begin_key_resp.duration', formal_cont_begin_key_resp.duration)
        # the Routine "formal_cont_begin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "formal_cont_fix" ---
        # create an object to store info about Routine formal_cont_fix
        formal_cont_fix = data.Routine(
            name='formal_cont_fix',
            components=[formal_cont_fix_text, formal_cont_fix_left_calm, formal_cont_fix_right_fear, formal_cont_fix_line],
        )
        formal_cont_fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal_cont_fix_Init_trial
        block_name = 'continuous'
        
        # store start times for formal_cont_fix
        formal_cont_fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_cont_fix.tStart = globalClock.getTime(format='float')
        formal_cont_fix.status = STARTED
        thisExp.addData('formal_cont_fix.started', formal_cont_fix.tStart)
        formal_cont_fix.maxDuration = None
        # keep track of which components have finished
        formal_cont_fixComponents = formal_cont_fix.components
        for thisComponent in formal_cont_fix.components:
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
        
        # --- Run Routine "formal_cont_fix" ---
        # if trial has changed, end Routine now
        if isinstance(formal_cont, data.TrialHandler2) and thisFormal_cont.thisN != formal_cont.thisTrial.thisN:
            continueRoutine = False
        formal_cont_fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_cont_fix_text* updates
            
            # if formal_cont_fix_text is starting this frame...
            if formal_cont_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_fix_text.frameNStart = frameN  # exact frame index
                formal_cont_fix_text.tStart = t  # local t and not account for scr refresh
                formal_cont_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_fix_text.started')
                # update status
                formal_cont_fix_text.status = STARTED
                formal_cont_fix_text.setAutoDraw(True)
            
            # if formal_cont_fix_text is active this frame...
            if formal_cont_fix_text.status == STARTED:
                # update params
                pass
            
            # if formal_cont_fix_text is stopping this frame...
            if formal_cont_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cont_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cont_fix_text.tStop = t  # not accounting for scr refresh
                    formal_cont_fix_text.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cont_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cont_fix_text.stopped')
                    # update status
                    formal_cont_fix_text.status = FINISHED
                    formal_cont_fix_text.setAutoDraw(False)
            
            # *formal_cont_fix_left_calm* updates
            
            # if formal_cont_fix_left_calm is starting this frame...
            if formal_cont_fix_left_calm.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_fix_left_calm.frameNStart = frameN  # exact frame index
                formal_cont_fix_left_calm.tStart = t  # local t and not account for scr refresh
                formal_cont_fix_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_fix_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_fix_left_calm.started')
                # update status
                formal_cont_fix_left_calm.status = STARTED
                formal_cont_fix_left_calm.setAutoDraw(True)
            
            # if formal_cont_fix_left_calm is active this frame...
            if formal_cont_fix_left_calm.status == STARTED:
                # update params
                pass
            
            # if formal_cont_fix_left_calm is stopping this frame...
            if formal_cont_fix_left_calm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cont_fix_left_calm.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cont_fix_left_calm.tStop = t  # not accounting for scr refresh
                    formal_cont_fix_left_calm.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cont_fix_left_calm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cont_fix_left_calm.stopped')
                    # update status
                    formal_cont_fix_left_calm.status = FINISHED
                    formal_cont_fix_left_calm.setAutoDraw(False)
            
            # *formal_cont_fix_right_fear* updates
            
            # if formal_cont_fix_right_fear is starting this frame...
            if formal_cont_fix_right_fear.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_fix_right_fear.frameNStart = frameN  # exact frame index
                formal_cont_fix_right_fear.tStart = t  # local t and not account for scr refresh
                formal_cont_fix_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_fix_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_fix_right_fear.started')
                # update status
                formal_cont_fix_right_fear.status = STARTED
                formal_cont_fix_right_fear.setAutoDraw(True)
            
            # if formal_cont_fix_right_fear is active this frame...
            if formal_cont_fix_right_fear.status == STARTED:
                # update params
                pass
            
            # if formal_cont_fix_right_fear is stopping this frame...
            if formal_cont_fix_right_fear.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cont_fix_right_fear.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cont_fix_right_fear.tStop = t  # not accounting for scr refresh
                    formal_cont_fix_right_fear.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cont_fix_right_fear.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cont_fix_right_fear.stopped')
                    # update status
                    formal_cont_fix_right_fear.status = FINISHED
                    formal_cont_fix_right_fear.setAutoDraw(False)
            
            # *formal_cont_fix_line* updates
            
            # if formal_cont_fix_line is starting this frame...
            if formal_cont_fix_line.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_fix_line.frameNStart = frameN  # exact frame index
                formal_cont_fix_line.tStart = t  # local t and not account for scr refresh
                formal_cont_fix_line.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_fix_line, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_fix_line.started')
                # update status
                formal_cont_fix_line.status = STARTED
                formal_cont_fix_line.setAutoDraw(True)
            
            # if formal_cont_fix_line is active this frame...
            if formal_cont_fix_line.status == STARTED:
                # update params
                pass
            
            # if formal_cont_fix_line is stopping this frame...
            if formal_cont_fix_line.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cont_fix_line.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cont_fix_line.tStop = t  # not accounting for scr refresh
                    formal_cont_fix_line.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cont_fix_line.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cont_fix_line.stopped')
                    # update status
                    formal_cont_fix_line.status = FINISHED
                    formal_cont_fix_line.setAutoDraw(False)
            
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
                formal_cont_fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_cont_fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_cont_fix" ---
        for thisComponent in formal_cont_fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_cont_fix
        formal_cont_fix.tStop = globalClock.getTime(format='float')
        formal_cont_fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_cont_fix.stopped', formal_cont_fix.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal_cont_fix.maxDurationReached:
            routineTimer.addTime(-formal_cont_fix.maxDuration)
        elif formal_cont_fix.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "formal_cont_face" ---
        # create an object to store info about Routine formal_cont_face
        formal_cont_face = data.Routine(
            name='formal_cont_face',
            components=[formal_cont_face_image],
        )
        formal_cont_face.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        formal_cont_face_image.setImage(S2cate2cont1)
        # store start times for formal_cont_face
        formal_cont_face.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_cont_face.tStart = globalClock.getTime(format='float')
        formal_cont_face.status = STARTED
        thisExp.addData('formal_cont_face.started', formal_cont_face.tStart)
        formal_cont_face.maxDuration = None
        # keep track of which components have finished
        formal_cont_faceComponents = formal_cont_face.components
        for thisComponent in formal_cont_face.components:
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
        
        # --- Run Routine "formal_cont_face" ---
        # if trial has changed, end Routine now
        if isinstance(formal_cont, data.TrialHandler2) and thisFormal_cont.thisN != formal_cont.thisTrial.thisN:
            continueRoutine = False
        formal_cont_face.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_cont_face_image* updates
            
            # if formal_cont_face_image is starting this frame...
            if formal_cont_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_face_image.frameNStart = frameN  # exact frame index
                formal_cont_face_image.tStart = t  # local t and not account for scr refresh
                formal_cont_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_face_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_face_image.started')
                # update status
                formal_cont_face_image.status = STARTED
                formal_cont_face_image.setAutoDraw(True)
            
            # if formal_cont_face_image is active this frame...
            if formal_cont_face_image.status == STARTED:
                # update params
                pass
            
            # if formal_cont_face_image is stopping this frame...
            if formal_cont_face_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_cont_face_image.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    formal_cont_face_image.tStop = t  # not accounting for scr refresh
                    formal_cont_face_image.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_cont_face_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_cont_face_image.stopped')
                    # update status
                    formal_cont_face_image.status = FINISHED
                    formal_cont_face_image.setAutoDraw(False)
            
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
                formal_cont_face.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_cont_face.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_cont_face" ---
        for thisComponent in formal_cont_face.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_cont_face
        formal_cont_face.tStop = globalClock.getTime(format='float')
        formal_cont_face.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_cont_face.stopped', formal_cont_face.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal_cont_face.maxDurationReached:
            routineTimer.addTime(-formal_cont_face.maxDuration)
        elif formal_cont_face.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "formal_cont_resp" ---
        # create an object to store info about Routine formal_cont_resp
        formal_cont_resp = data.Routine(
            name='formal_cont_resp',
            components=[formal_cont_resp_prompt, formal_cont_resp_slider_shape, formal_cont_resp_slider, formal_cont_resp_left_calm, formal_cont_resp_right_fear, formal_cont_resp_mouse],
        )
        formal_cont_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        formal_cont_resp_slider.reset()
        # setup some python lists for storing info about the formal_cont_resp_mouse
        formal_cont_resp_mouse.x = []
        formal_cont_resp_mouse.y = []
        formal_cont_resp_mouse.leftButton = []
        formal_cont_resp_mouse.midButton = []
        formal_cont_resp_mouse.rightButton = []
        formal_cont_resp_mouse.time = []
        formal_cont_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        formal_cont_resp_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from formal_cont_resp_code
        # Change the following parameters according to your own exp
        slider_width = 0.5
        slider_height = .05
        slider_orientation = 0
        #Failing to set ticks or labels in code
        slider_ticks=[0, 1]
        
        slider_granularity = .1
        slider_decimals = 2
        formal_cont_resp_slider.decimals = 2
        # 获取Slider组件的标记（marker）
        marker = formal_cont_resp_slider.marker
        # 将标记的形状更改为一个特定尺寸的方块
        marker.shape = 'square'
        marker.size = (0.01, 0.05)  # 设置方块的尺寸
        # Begin Routine
        
        # record the current position of the mouse
        mouseRec=formal_cont_resp_mouse.getPos()
        
        formal_cont_resp_mouse.setPos((0, 0))
        # store start times for formal_cont_resp
        formal_cont_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_cont_resp.tStart = globalClock.getTime(format='float')
        formal_cont_resp.status = STARTED
        thisExp.addData('formal_cont_resp.started', formal_cont_resp.tStart)
        formal_cont_resp.maxDuration = None
        # keep track of which components have finished
        formal_cont_respComponents = formal_cont_resp.components
        for thisComponent in formal_cont_resp.components:
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
        
        # --- Run Routine "formal_cont_resp" ---
        # if trial has changed, end Routine now
        if isinstance(formal_cont, data.TrialHandler2) and thisFormal_cont.thisN != formal_cont.thisTrial.thisN:
            continueRoutine = False
        formal_cont_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_cont_resp_prompt* updates
            
            # if formal_cont_resp_prompt is starting this frame...
            if formal_cont_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_resp_prompt.frameNStart = frameN  # exact frame index
                formal_cont_resp_prompt.tStart = t  # local t and not account for scr refresh
                formal_cont_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_resp_prompt.started')
                # update status
                formal_cont_resp_prompt.status = STARTED
                formal_cont_resp_prompt.setAutoDraw(True)
            
            # if formal_cont_resp_prompt is active this frame...
            if formal_cont_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *formal_cont_resp_slider_shape* updates
            
            # if formal_cont_resp_slider_shape is starting this frame...
            if formal_cont_resp_slider_shape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_resp_slider_shape.frameNStart = frameN  # exact frame index
                formal_cont_resp_slider_shape.tStart = t  # local t and not account for scr refresh
                formal_cont_resp_slider_shape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_resp_slider_shape, 'tStartRefresh')  # time at next scr refresh
                # update status
                formal_cont_resp_slider_shape.status = STARTED
                formal_cont_resp_slider_shape.setAutoDraw(True)
            
            # if formal_cont_resp_slider_shape is active this frame...
            if formal_cont_resp_slider_shape.status == STARTED:
                # update params
                pass
            
            # *formal_cont_resp_slider* updates
            
            # if formal_cont_resp_slider is starting this frame...
            if formal_cont_resp_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_resp_slider.frameNStart = frameN  # exact frame index
                formal_cont_resp_slider.tStart = t  # local t and not account for scr refresh
                formal_cont_resp_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_resp_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_resp_slider.started')
                # update status
                formal_cont_resp_slider.status = STARTED
                formal_cont_resp_slider.setAutoDraw(True)
            
            # if formal_cont_resp_slider is active this frame...
            if formal_cont_resp_slider.status == STARTED:
                # update params
                pass
            
            # *formal_cont_resp_left_calm* updates
            
            # if formal_cont_resp_left_calm is starting this frame...
            if formal_cont_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_resp_left_calm.frameNStart = frameN  # exact frame index
                formal_cont_resp_left_calm.tStart = t  # local t and not account for scr refresh
                formal_cont_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_resp_left_calm.started')
                # update status
                formal_cont_resp_left_calm.status = STARTED
                formal_cont_resp_left_calm.setAutoDraw(True)
            
            # if formal_cont_resp_left_calm is active this frame...
            if formal_cont_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *formal_cont_resp_right_fear* updates
            
            # if formal_cont_resp_right_fear is starting this frame...
            if formal_cont_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_resp_right_fear.frameNStart = frameN  # exact frame index
                formal_cont_resp_right_fear.tStart = t  # local t and not account for scr refresh
                formal_cont_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'formal_cont_resp_right_fear.started')
                # update status
                formal_cont_resp_right_fear.status = STARTED
                formal_cont_resp_right_fear.setAutoDraw(True)
            
            # if formal_cont_resp_right_fear is active this frame...
            if formal_cont_resp_right_fear.status == STARTED:
                # update params
                pass
            # *formal_cont_resp_mouse* updates
            
            # if formal_cont_resp_mouse is starting this frame...
            if formal_cont_resp_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_cont_resp_mouse.frameNStart = frameN  # exact frame index
                formal_cont_resp_mouse.tStart = t  # local t and not account for scr refresh
                formal_cont_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(formal_cont_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('formal_cont_resp_mouse.started', t)
                # update status
                formal_cont_resp_mouse.status = STARTED
                prevButtonState = formal_cont_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if formal_cont_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = formal_cont_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(formal_cont_resp_slider_shape, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(formal_cont_resp_mouse):
                                gotValidClick = True
                                formal_cont_resp_mouse.clicked_name.append(obj.name)
                                formal_cont_resp_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = formal_cont_resp_mouse.getPos()
                            formal_cont_resp_mouse.x.append(x)
                            formal_cont_resp_mouse.y.append(y)
                            buttons = formal_cont_resp_mouse.getPressed()
                            formal_cont_resp_mouse.leftButton.append(buttons[0])
                            formal_cont_resp_mouse.midButton.append(buttons[1])
                            formal_cont_resp_mouse.rightButton.append(buttons[2])
                            formal_cont_resp_mouse.time.append(formal_cont_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from formal_cont_resp_code
            
            # Create a slider_shape in this routine with the same size as the slider
            # Check if mouse is moving within slider_shape, if so change marker position accordingly
            if formal_cont_resp_slider_shape.contains(formal_cont_resp_mouse):
                mouseRec = formal_cont_resp_mouse.getPos()
                formal_cont_resp_slider.markerPos = mouseRec[slider_orientation]/slider_width*(slider_ticks[-1]-slider_ticks[0])+(slider_ticks[0]+slider_ticks[-1])/2
                # (new)Record marker position only on left mouse click
                if formal_cont_resp_mouse.getPressed()[0]:
                    cont_resp = round(formal_cont_resp_slider.markerPos, slider_decimals)
                    continueRoutine = False
                else:
                    cont_resp = 'None'
                    continueRoutine = True
            else:
                cont_resp = 'None'
                continueRoutine = True
            
            win.mouseVisible = True
            
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
                formal_cont_resp.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_cont_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_cont_resp" ---
        for thisComponent in formal_cont_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_cont_resp
        formal_cont_resp.tStop = globalClock.getTime(format='float')
        formal_cont_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_cont_resp.stopped', formal_cont_resp.tStop)
        formal_cont.addData('formal_cont_resp_slider.response', formal_cont_resp_slider.getRating())
        formal_cont.addData('formal_cont_resp_slider.rt', formal_cont_resp_slider.getRT())
        # store data for formal_cont (TrialHandler)
        formal_cont.addData('formal_cont_resp_mouse.x', formal_cont_resp_mouse.x)
        formal_cont.addData('formal_cont_resp_mouse.y', formal_cont_resp_mouse.y)
        formal_cont.addData('formal_cont_resp_mouse.leftButton', formal_cont_resp_mouse.leftButton)
        formal_cont.addData('formal_cont_resp_mouse.midButton', formal_cont_resp_mouse.midButton)
        formal_cont.addData('formal_cont_resp_mouse.rightButton', formal_cont_resp_mouse.rightButton)
        formal_cont.addData('formal_cont_resp_mouse.time', formal_cont_resp_mouse.time)
        formal_cont.addData('formal_cont_resp_mouse.clicked_name', formal_cont_resp_mouse.clicked_name)
        # Run 'End Routine' code from formal_cont_resp_code
        trialID = trialID + 1
        
        thisExp.addData('block_name', block_name)
        thisExp.addData('trialID', trialID)
        thisExp.addData('face_stim', S2cate2cont1)
        
        #thisExp.addData('All Responses',slider_data)
        thisExp.addData('cont_resp',cont_resp)
        
        print('cont_resp',cont_resp)
        
        win.mouseVisible = False
        
        # the Routine "formal_cont_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'formal_cont'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end_exp" ---
    # create an object to store info about Routine end_exp
    end_exp = data.Routine(
        name='end_exp',
        components=[end_exp_text],
    )
    end_exp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end_exp
    end_exp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_exp.tStart = globalClock.getTime(format='float')
    end_exp.status = STARTED
    thisExp.addData('end_exp.started', end_exp.tStart)
    end_exp.maxDuration = None
    # keep track of which components have finished
    end_expComponents = end_exp.components
    for thisComponent in end_exp.components:
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
    
    # --- Run Routine "end_exp" ---
    end_exp.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_exp_text* updates
        
        # if end_exp_text is starting this frame...
        if end_exp_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_exp_text.frameNStart = frameN  # exact frame index
            end_exp_text.tStart = t  # local t and not account for scr refresh
            end_exp_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_exp_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_exp_text.started')
            # update status
            end_exp_text.status = STARTED
            end_exp_text.setAutoDraw(True)
        
        # if end_exp_text is active this frame...
        if end_exp_text.status == STARTED:
            # update params
            pass
        
        # if end_exp_text is stopping this frame...
        if end_exp_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_exp_text.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                end_exp_text.tStop = t  # not accounting for scr refresh
                end_exp_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_exp_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_exp_text.stopped')
                # update status
                end_exp_text.status = FINISHED
                end_exp_text.setAutoDraw(False)
        
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
            end_exp.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_exp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_exp" ---
    for thisComponent in end_exp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_exp
    end_exp.tStop = globalClock.getTime(format='float')
    end_exp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_exp.stopped', end_exp.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end_exp.maxDurationReached:
        routineTimer.addTime(-end_exp.maxDuration)
    elif end_exp.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
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
