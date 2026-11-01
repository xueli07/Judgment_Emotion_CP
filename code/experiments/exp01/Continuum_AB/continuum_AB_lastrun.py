#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 十二月 06, 2025, at 19:17
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
expName = 'Continuum_AB'  # from the Builder filename that created this script
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
    if deviceManager.getDevice('instr6_key_resp') is None:
        # initialise instr6_key_resp
        instr6_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr6_key_resp',
        )
    if deviceManager.getDevice('instr7_key_resp') is None:
        # initialise instr7_key_resp
        instr7_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr7_key_resp',
        )
    if deviceManager.getDevice('instr8_key_resp') is None:
        # initialise instr8_key_resp
        instr8_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr8_key_resp',
        )
    if deviceManager.getDevice('instr9_key_resp') is None:
        # initialise instr9_key_resp
        instr9_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr9_key_resp',
        )
    if deviceManager.getDevice('instr10_key_resp') is None:
        # initialise instr10_key_resp
        instr10_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr10_key_resp',
        )
    if deviceManager.getDevice('instr11_key_resp') is None:
        # initialise instr11_key_resp
        instr11_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr11_key_resp',
        )
    if deviceManager.getDevice('prac_begin_key_resp') is None:
        # initialise prac_begin_key_resp
        prac_begin_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_begin_key_resp',
        )
    if deviceManager.getDevice('prac_begin_f_key_resp') is None:
        # initialise prac_begin_f_key_resp
        prac_begin_f_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_begin_f_key_resp',
        )
    if deviceManager.getDevice('prac_end_key_resp') is None:
        # initialise prac_end_key_resp
        prac_end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_end_key_resp',
        )
    if deviceManager.getDevice('form_begin_m_key_resp') is None:
        # initialise form_begin_m_key_resp
        form_begin_m_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='form_begin_m_key_resp',
        )
    if deviceManager.getDevice('form_begin_f_key_resp') is None:
        # initialise form_begin_f_key_resp
        form_begin_f_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='form_begin_f_key_resp',
        )
    if deviceManager.getDevice('form_break_key_resp') is None:
        # initialise form_break_key_resp
        form_break_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='form_break_key_resp',
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
        text='情绪识别实验指导语\n',
        font='Arial',
        pos=(0, 0.18), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr1_center_text = visual.TextStim(win=win, name='instr1_center_text',
        text='您好，欢迎参加本实验！\n\n这是一个情绪识别实验，您一共需要完成2个任务，\n\n整个实验过程大约需要40分钟，\n\n请您仔细阅读后面的指导语。\n\n按“空格”键查看更多说明\n',
        font='Arial',
        pos=(0, -0.05), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr1_key_resp = keyboard.Keyboard(deviceName='instr1_key_resp')
    
    # --- Initialize components for Routine "instr2" ---
    instr2_center_text = visual.TextStim(win=win, name='instr2_center_text',
        text='情绪表达往往是复杂和微妙的，很少是绝对单一的情绪。\n\n就像音量调节一样，我们很少将音量调到绝对的"开"或"关"，\n\n而是根据需要在两个极端之间连续调节。\n\n同样地，真实生活中的情绪表达也很少是极端的纯开心或纯悲伤，\n\n而是存在着丰富的程度差异。\n\n一个人可能同时带着一丝忧伤的微笑，或者在开心中夹杂着一点担忧。\n\n正是这种程度上的连续变化，让人类的情绪表达如此细腻和真实。\n\n\n按“空格”键继续\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr2_key_resp = keyboard.Keyboard(deviceName='instr2_key_resp')
    
    # --- Initialize components for Routine "instr3" ---
    instr3_center_text = visual.TextStim(win=win, name='instr3_center_text',
        text='在第一个任务中，\n\n我们想了解人们是如何感知这种连续变化的情绪程度的。\n\n在任务中，屏幕上会依次呈现一系列人物面孔图片，\n\n【您的任务】：在每张面孔出现后，评估每张面孔的情绪程度。\n\n\n按“空格”键继续\n\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr3_key_resp = keyboard.Keyboard(deviceName='instr3_key_resp')
    
    # --- Initialize components for Routine "instr4" ---
    instr4_top_text = visual.TextStim(win=win, name='instr4_top_text',
        text='【评估方式】：如下所示，您将使用一条代表情绪变化的水平直线进行评估\n',
        font='Arial',
        pos=(-0.35, 0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr4_center_text = visual.TextStim(win=win, name='instr4_center_text',
        text='【操作方式】：\n\n直线左端代表100%开心（0%悲伤），即“完全开心，没有一点悲伤”，\n\n直线右端代表100%悲伤（0%开心），即“完全悲伤，没有一点开心”,\n\n直线的中间代表两种情绪不同程度的混合，\n\n请用鼠标在直线上点击，来表示您认为该面孔的情绪程度。',
        font='Arial',
        pos=(-0.35, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr4_bottom_text = visual.TextStim(win=win, name='instr4_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    instr4_left_text = visual.TextStim(win=win, name='instr4_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    instr4_right_text = visual.TextStim(win=win, name='instr4_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    instr4_slider = visual.Slider(win=win, name='instr4_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, 0.1), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    instr4_mouse = event.Mouse(win=win)
    x, y = [None, None]
    instr4_mouse.mouseClock = core.Clock()
    instr4_key_resp = keyboard.Keyboard(deviceName='instr4_key_resp')
    
    # --- Initialize components for Routine "instr5" ---
    instr5_top_text = visual.TextStim(win=win, name='instr5_top_text',
        text='【示例理解】：接下来，我们将通过几个例子来学习如何使用它，\n\n首先，我们来看一些情绪比较明显的例子。\n\n',
        font='Arial',
        pos=(-0.35, 0.36), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr5_center_text = visual.TextStim(win=win, name='instr5_center_text',
        text='【操作方式】：\n\n对于这样一张表现出强烈、纯粹开心的面孔，您应该将鼠标拖动到最左端。\n\n请注意：正式实验中，第一次鼠标点击后，即会自动进入下一张面孔的判断。\n',
        font='Arial',
        pos=(-0.35, -0.33), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr5_bottom_text = visual.TextStim(win=win, name='instr5_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    instr5_left_text = visual.TextStim(win=win, name='instr5_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    instr5_right_text = visual.TextStim(win=win, name='instr5_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    instr5_image = visual.ImageStim(
        win=win,
        name='instr5_image', 
        image='prac_training_faces/m_0_m1.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    instr5_slider = visual.Slider(win=win, name='instr5_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, -0.1), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    instr5_mouse = event.Mouse(win=win)
    x, y = [None, None]
    instr5_mouse.mouseClock = core.Clock()
    instr5_key_resp = keyboard.Keyboard(deviceName='instr5_key_resp')
    
    # --- Initialize components for Routine "instr6" ---
    instr6_top_text = visual.TextStim(win=win, name='instr6_top_text',
        text='【示例理解】：接下来，我们将通过几个例子来学习如何使用它，\n\n首先，我们来看一些情绪比较明显的例子。\n\n',
        font='Arial',
        pos=(-0.35, 0.36), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr6_center_text = visual.TextStim(win=win, name='instr6_center_text',
        text='【操作方式】：\n\n对于这样一张表现出强烈、纯粹悲伤的面孔，您应该将鼠标拖动到最右端。\n\n这些情况下，请您果断地使用直线的两端。\n',
        font='Arial',
        pos=(-0.35, -0.33), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr6_bottom_text = visual.TextStim(win=win, name='instr6_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    instr6_left_text = visual.TextStim(win=win, name='instr6_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    instr6_right_text = visual.TextStim(win=win, name='instr6_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    instr6_image = visual.ImageStim(
        win=win,
        name='instr6_image', 
        image='prac_training_faces/m_100_m1.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    instr6_slider = visual.Slider(win=win, name='instr6_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, -0.1), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    instr6_mouse = event.Mouse(win=win)
    x, y = [None, None]
    instr6_mouse.mouseClock = core.Clock()
    instr6_key_resp = keyboard.Keyboard(deviceName='instr6_key_resp')
    
    # --- Initialize components for Routine "instr7" ---
    instr7_top_text = visual.TextStim(win=win, name='instr7_top_text',
        text='【示例理解】：上述例子是为了向您展示什么是纯粹的‘开心’和‘悲伤’，从而帮助您理解这两种情绪的的含义。\n\n然而，很多面孔的情绪并不那么极端，而是两种情绪不同程度的混合。',
        font='Arial',
        pos=(-0.35, 0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr7_center_text = visual.TextStim(win=win, name='instr7_center_text',
        text='【操作方式】：\n\n请看这张面孔。它可能既不完全是开心，也不完全是悲伤，而是两种情绪都有一些。\n\n有些人可能觉得它更像一种“苦乐参半”或“略带伤感的微笑”。\n\n现在，请根据您对这张面孔的【第一感觉】，拖动滑块到直线上您认为最匹配的位置。',
        font='Arial',
        pos=(-0.35, -0.3), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr7_bottom_text = visual.TextStim(win=win, name='instr7_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    instr7_left_text = visual.TextStim(win=win, name='instr7_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    instr7_right_text = visual.TextStim(win=win, name='instr7_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    instr7_image = visual.ImageStim(
        win=win,
        name='instr7_image', 
        image='prac_training_faces/m_50_m1.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    instr7_slider = visual.Slider(win=win, name='instr7_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, -0.1), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    instr7_mouse = event.Mouse(win=win)
    x, y = [None, None]
    instr7_mouse.mouseClock = core.Clock()
    instr7_key_resp = keyboard.Keyboard(deviceName='instr7_key_resp')
    
    # --- Initialize components for Routine "instr8" ---
    instr8_top_text = visual.TextStim(win=win, name='instr8_top_text',
        text='【示例理解】：然而，很多面孔的情绪并不那么极端，而是两种情绪不同程度的混合。\n',
        font='Arial',
        pos=(-0.35, 0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr8_center_text = visual.TextStim(win=win, name='instr8_center_text',
        text='【操作方式】：\n\n很好！正如您所体验的，对于这种混合情绪，不同的人可能会将鼠标放在中间区域的不同位置，这完全是正常的。\n\n【请记住】：我们不需要您进行精确计算，只需要您相信自己的直觉。没有标准答案。',
        font='Arial',
        pos=(-0.35, -0.28), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr8_bottom_text = visual.TextStim(win=win, name='instr8_bottom_text',
        text='按“空格”键继续',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    instr8_left_text = visual.TextStim(win=win, name='instr8_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    instr8_right_text = visual.TextStim(win=win, name='instr8_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    instr8_image = visual.ImageStim(
        win=win,
        name='instr8_image', 
        image='prac_training_faces/m_50_m1.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    instr8_slider = visual.Slider(win=win, name='instr8_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, -0.1), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    instr8_mouse = event.Mouse(win=win)
    x, y = [None, None]
    instr8_mouse.mouseClock = core.Clock()
    instr8_key_resp = keyboard.Keyboard(deviceName='instr8_key_resp')
    
    # --- Initialize components for Routine "instr9" ---
    instr9_center_text = visual.TextStim(win=win, name='instr9_center_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr9_bottom_text = visual.TextStim(win=win, name='instr9_bottom_text',
        text='按“空格”键开始练习实验',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr9_key_resp = keyboard.Keyboard(deviceName='instr9_key_resp')
    
    # --- Initialize components for Routine "instr10" ---
    instr10_center_text = visual.TextStim(win=win, name='instr10_center_text',
        text='【练习实验说明】：\n\n在练习中，系统会提供实时反馈，即显示情绪百分比，\n\n如：70%开心（30%悲伤），旨在帮助您建构对情绪连续体的理解，\n\n即：从纯开心到纯悲伤是一个渐变的连续体，\n\n直线中间的不同位置对应着不同的情绪程度。\n\n【您的目标】：通过练习，逐渐形成对情绪连续性的直觉理解，\n\n而非刻意记住任何数字或位置。\n\n【请注意】：这个实时提示仅用于练习试次。\n\n正式实验开始后，提示将会消失。\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr10_bottom_text = visual.TextStim(win=win, name='instr10_bottom_text',
        text='按“空格”键查看更多练习实验说明',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr10_key_resp = keyboard.Keyboard(deviceName='instr10_key_resp')
    
    # --- Initialize components for Routine "instr11" ---
    instr11_center_text = visual.TextStim(win=win, name='instr11_center_text',
        text='正式实验开始前，您将先进行22次练习，以熟悉任务流程。\n\n如果您已完全理解任务要求，按"空格"键开始练习，\n\n如果还有任何疑问，请联系主试。\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr11_bottom_text = visual.TextStim(win=win, name='instr11_bottom_text',
        text='按“空格”键开始练习实验',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr11_key_resp = keyboard.Keyboard(deviceName='instr11_key_resp')
    # Run 'Begin Experiment' code from instr11_code
    # 以下代码是关于初始化关键变量
    
    #######
    # 练习实验的全局计数器和关键变量
    # 练习重新开始的次数计数
    prac_cont_block = 1
    # 所有试次的连续计数
    prac_cont_trial = 0
    # 记录当前试次的性别类型
    prac_current_gender = ''
    # 记录前一个试次的性别，用于检测mini-block转换
    prac_previous_gender = None
    
    
    ########
    # 以下是正式实验的全局计数和关键变量
    # block计数
    form_cont_block = 1
    # 所有试次的计数
    form_cont_trial = 0
    # 记录当前试次的性别类型
    form_current_gender = ''
    # 记录前一个试次的性别，用于检测mini-block转换
    form_previous_gender = None
    
    
    # --- Initialize components for Routine "prac_begin_m" ---
    prac_begin_text = visual.TextStim(win=win, name='prac_begin_text',
        text='按“空格”键开始',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_begin_key_resp = keyboard.Keyboard(deviceName='prac_begin_key_resp')
    
    # --- Initialize components for Routine "prac_cross_m" ---
    prac_cross_m_text = visual.TextStim(win=win, name='prac_cross_m_text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_resp_m" ---
    prac_resp_m_face_image = visual.ImageStim(
        win=win,
        name='prac_resp_m_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac_resp_m_left_text = visual.TextStim(win=win, name='prac_resp_m_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_resp_m_right_text = visual.TextStim(win=win, name='prac_resp_m_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_resp_m_slider = visual.Slider(win=win, name='prac_resp_m_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, -0.25), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    prac_resp_m_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_resp_m_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prac_blank_m" ---
    prac_blank_text = visual.TextStim(win=win, name='prac_blank_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_break_m" ---
    prac_m_break_text = visual.TextStim(win=win, name='prac_m_break_text',
        text='现在休息10秒',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_begin_f" ---
    prac_begin_f_text = visual.TextStim(win=win, name='prac_begin_f_text',
        text='按“空格”键开始',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_begin_f_key_resp = keyboard.Keyboard(deviceName='prac_begin_f_key_resp')
    
    # --- Initialize components for Routine "prac_cross_f" ---
    prac_cross_f_text = visual.TextStim(win=win, name='prac_cross_f_text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_resp_f" ---
    prac_resp_f_face_image = visual.ImageStim(
        win=win,
        name='prac_resp_f_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac_resp_f_left_text = visual.TextStim(win=win, name='prac_resp_f_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_resp_f_right_text = visual.TextStim(win=win, name='prac_resp_f_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_resp_f_slider = visual.Slider(win=win, name='prac_resp_f_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, -0.25), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    prac_resp_f_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_resp_f_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prac_blank_f" ---
    prac_blank_f_text = visual.TextStim(win=win, name='prac_blank_f_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_end" ---
    prac_end_center_text = visual.TextStim(win=win, name='prac_end_center_text',
        text='恭喜！您已经完成所有练习试次，接下来将开始正式实验。\n\n该任务总共分为3个部分，每完成一部分后会有休息时间。您可以休息1分钟，感觉准备好了就继续。\n\n整个任务大约需要24分钟。\n\n\n【请注意】：\n\n1.正式实验中没有情绪程度的实时提示，请您完全依靠自己的直觉进行评估，答案本身没有对错。\n\n2.实验结束后，我们会检查您实验的认真程度，其与您的被试费挂钩，请您务必认真作答。\n\n\n如果仍需练习，请按“P”键返回练习。如果仍有疑问，请联系主试。\n\n如果您已完全理解任务要求，请空格键开始正式实验。',
        font='Arial',
        pos=(-0.21, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_end_bottom_text = visual.TextStim(win=win, name='prac_end_bottom_text',
        text='按“空格”键开始正式实验',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_end_key_resp = keyboard.Keyboard(deviceName='prac_end_key_resp')
    # Run 'Begin Experiment' code from prac_end_code
    # 初始化变量，只在实验开始时运行一次
    needPractice = True
    
    # --- Initialize components for Routine "form_begin_m" ---
    form_begin_m_text = visual.TextStim(win=win, name='form_begin_m_text',
        text='按“空格”键开始',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    form_begin_m_key_resp = keyboard.Keyboard(deviceName='form_begin_m_key_resp')
    
    # --- Initialize components for Routine "form_cross_m" ---
    form_cross_m_text = visual.TextStim(win=win, name='form_cross_m_text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "form_resp_m" ---
    form_resp_m_face_image = visual.ImageStim(
        win=win,
        name='form_resp_m_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    form_resp_m_left_text = visual.TextStim(win=win, name='form_resp_m_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    form_resp_m_right_text = visual.TextStim(win=win, name='form_resp_m_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    form_resp_m_slider = visual.Slider(win=win, name='form_resp_m_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, -0.25), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    form_resp_m_mouse = event.Mouse(win=win)
    x, y = [None, None]
    form_resp_m_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "form_blank_m" ---
    form_blank_m_text = visual.TextStim(win=win, name='form_blank_m_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "form_break_m" ---
    form_break_m_text = visual.TextStim(win=win, name='form_break_m_text',
        text='现在休息10秒',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "form_begin_f" ---
    form_begin_f_text = visual.TextStim(win=win, name='form_begin_f_text',
        text='按“空格”键开始',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    form_begin_f_key_resp = keyboard.Keyboard(deviceName='form_begin_f_key_resp')
    
    # --- Initialize components for Routine "form_cross_f" ---
    form_cross_f_text = visual.TextStim(win=win, name='form_cross_f_text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "form_resp_f" ---
    form_resp_f_face_image = visual.ImageStim(
        win=win,
        name='form_resp_f_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    form_resp_f_left_text = visual.TextStim(win=win, name='form_resp_f_left_text',
        text='100%开心\n(0%悲伤)',
        font='Arial',
        pos=(-0.5, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    form_resp_f_right_text = visual.TextStim(win=win, name='form_resp_f_right_text',
        text='100%悲伤\n(0%开心)',
        font='Arial',
        pos=(0.5, -0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    form_resp_f_slider = visual.Slider(win=win, name='form_resp_f_slider',
        startValue=0.5, size=(0.8, 0.05), pos=(0, -0.25), units=win.units,
        labels=None, ticks=[0,1], granularity=0.01,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, -1.0000, -1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    form_resp_f_mouse = event.Mouse(win=win)
    x, y = [None, None]
    form_resp_f_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "form_blank_f" ---
    form_blank_f_text = visual.TextStim(win=win, name='form_blank_f_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "form_break_f" ---
    form_break_f_text = visual.TextStim(win=win, name='form_break_f_text',
        text='现在休息10秒',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "form_break" ---
    form_break_center_text = visual.TextStim(win=win, name='form_break_center_text',
        text='现在休息1分钟\n\n1分钟后，如果您感觉准备好了，您需要按“空格”键继续实验',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    form_break_bottom_text = visual.TextStim(win=win, name='form_break_bottom_text',
        text='按“空格”键继续实验',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    form_break_key_resp = keyboard.Keyboard(deviceName='form_break_key_resp')
    
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
        image='prac_training_faces/m_10_f1.png', mask=None, anchor='center',
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
        image='prac_training_faces/m_10_f1.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.15, -0.1), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    xab_instr2_face_b_image = visual.ImageStim(
        win=win,
        name='xab_instr2_face_b_image', 
        image='prac_training_faces/m_10_f1.png', mask=None, anchor='center',
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
    pracl_xab_blank1_text = visual.TextStim(win=win, name='pracl_xab_blank1_text',
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
        components=[instr3_center_text, instr3_key_resp],
    )
    instr3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
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
        
        # *instr3_center_text* updates
        
        # if instr3_center_text is starting this frame...
        if instr3_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr3_center_text.frameNStart = frameN  # exact frame index
            instr3_center_text.tStart = t  # local t and not account for scr refresh
            instr3_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr3_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr3_center_text.started')
            # update status
            instr3_center_text.status = STARTED
            instr3_center_text.setAutoDraw(True)
        
        # if instr3_center_text is active this frame...
        if instr3_center_text.status == STARTED:
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
        components=[instr4_top_text, instr4_center_text, instr4_bottom_text, instr4_left_text, instr4_right_text, instr4_slider, instr4_mouse, instr4_key_resp],
    )
    instr4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instr4_slider.reset()
    # setup some python lists for storing info about the instr4_mouse
    instr4_mouse.x = []
    instr4_mouse.y = []
    instr4_mouse.leftButton = []
    instr4_mouse.midButton = []
    instr4_mouse.rightButton = []
    instr4_mouse.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for instr4_key_resp
    instr4_key_resp.keys = []
    instr4_key_resp.rt = []
    _instr4_key_resp_allKeys = []
    # Run 'Begin Routine' code from instr4_code
    # 控制文字的左对齐
    instr4_center_text.alignText = 'left'  # 设置左对齐
    instr4_top_text.alignText = 'left'
    
    # 显示鼠标可见
    win.mouseVisible = True
    
    # 初始化鼠标位置在slider中间
    instr4_mouse.setPos([0, 0.1])
    
    # 设置slider的marker外观
    # marker就是鼠标在直线上的光标
    instr4_slider.marker.size = (0.02, 0.08)  # 设置marker大小（宽度，高度）
    instr4_slider.marker.shape = 'square'     # 设置为方形
    instr4_slider.marker.fillColor = 'white'    # 设置颜色
    
    # 记录slider宽度，用于后续计算
    slider_width = 0.8  
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
        
        # *instr4_top_text* updates
        
        # if instr4_top_text is starting this frame...
        if instr4_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr4_top_text.frameNStart = frameN  # exact frame index
            instr4_top_text.tStart = t  # local t and not account for scr refresh
            instr4_top_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr4_top_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr4_top_text.started')
            # update status
            instr4_top_text.status = STARTED
            instr4_top_text.setAutoDraw(True)
        
        # if instr4_top_text is active this frame...
        if instr4_top_text.status == STARTED:
            # update params
            pass
        
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
        
        # *instr4_left_text* updates
        
        # if instr4_left_text is starting this frame...
        if instr4_left_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr4_left_text.frameNStart = frameN  # exact frame index
            instr4_left_text.tStart = t  # local t and not account for scr refresh
            instr4_left_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr4_left_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr4_left_text.started')
            # update status
            instr4_left_text.status = STARTED
            instr4_left_text.setAutoDraw(True)
        
        # if instr4_left_text is active this frame...
        if instr4_left_text.status == STARTED:
            # update params
            pass
        
        # *instr4_right_text* updates
        
        # if instr4_right_text is starting this frame...
        if instr4_right_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr4_right_text.frameNStart = frameN  # exact frame index
            instr4_right_text.tStart = t  # local t and not account for scr refresh
            instr4_right_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr4_right_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr4_right_text.started')
            # update status
            instr4_right_text.status = STARTED
            instr4_right_text.setAutoDraw(True)
        
        # if instr4_right_text is active this frame...
        if instr4_right_text.status == STARTED:
            # update params
            pass
        
        # *instr4_slider* updates
        
        # if instr4_slider is starting this frame...
        if instr4_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr4_slider.frameNStart = frameN  # exact frame index
            instr4_slider.tStart = t  # local t and not account for scr refresh
            instr4_slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr4_slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr4_slider.started')
            # update status
            instr4_slider.status = STARTED
            instr4_slider.setAutoDraw(True)
        
        # if instr4_slider is active this frame...
        if instr4_slider.status == STARTED:
            # update params
            pass
        # *instr4_mouse* updates
        
        # if instr4_mouse is starting this frame...
        if instr4_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr4_mouse.frameNStart = frameN  # exact frame index
            instr4_mouse.tStart = t  # local t and not account for scr refresh
            instr4_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr4_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('instr4_mouse.started', t)
            # update status
            instr4_mouse.status = STARTED
            instr4_mouse.mouseClock.reset()
            prevButtonState = instr4_mouse.getPressed()  # if button is down already this ISN'T a new click
        if instr4_mouse.status == STARTED:  # only update if started and not finished!
            buttons = instr4_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
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
        # Run 'Each Frame' code from instr4_code
        # 获取当前鼠标位置
        mouse_pos = instr4_mouse.getPos()
        mouse_x, mouse_y = mouse_pos
        
        # 基于slider的位置和大小，计算slider的边界
        slider_center_x = 0      # slider的中心x坐标
        slider_center_y = 0.1    # slider的中心y坐标
        slider_left = slider_center_x - slider_width/2    # 左边界
        slider_right = slider_center_x + slider_width/2   # 右边界
        slider_bottom = slider_center_y - 0.025          # 下边界
        slider_top = slider_center_y + 0.025             # 上边界
        
        # 检查鼠标是否在slider区域内（四个边界条件）
        mouse_in_slider = (slider_left <= mouse_x <= slider_right and 
                           slider_bottom <= mouse_y <= slider_top)
                           
        if mouse_in_slider:
            # 当鼠标在slider上时，让marker跟随鼠标移动
            
            # 将鼠标x坐标转换为slider的0-1范围
            # 这个计算将屏幕坐标映射到评分坐标
            slider_value = (mouse_x - slider_left) / slider_width
            
            # 限制数值在有效范围内，防止超出边界
            slider_value = max(0, min(1, slider_value))
            
            # 设置marker位置，这会让光标跟随鼠标移动
            instr4_slider.markerPos = slider_value
            
        
        
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
    thisExp.addData('instr4_slider.response', instr4_slider.getRating())
    thisExp.addData('instr4_slider.rt', instr4_slider.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('instr4_mouse.x', instr4_mouse.x)
    thisExp.addData('instr4_mouse.y', instr4_mouse.y)
    thisExp.addData('instr4_mouse.leftButton', instr4_mouse.leftButton)
    thisExp.addData('instr4_mouse.midButton', instr4_mouse.midButton)
    thisExp.addData('instr4_mouse.rightButton', instr4_mouse.rightButton)
    thisExp.addData('instr4_mouse.time', instr4_mouse.time)
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
    
    # --- Prepare to start Routine "instr5" ---
    # create an object to store info about Routine instr5
    instr5 = data.Routine(
        name='instr5',
        components=[instr5_top_text, instr5_center_text, instr5_bottom_text, instr5_left_text, instr5_right_text, instr5_image, instr5_slider, instr5_mouse, instr5_key_resp],
    )
    instr5.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instr5_slider.reset()
    # setup some python lists for storing info about the instr5_mouse
    instr5_mouse.x = []
    instr5_mouse.y = []
    instr5_mouse.leftButton = []
    instr5_mouse.midButton = []
    instr5_mouse.rightButton = []
    instr5_mouse.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for instr5_key_resp
    instr5_key_resp.keys = []
    instr5_key_resp.rt = []
    _instr5_key_resp_allKeys = []
    # Run 'Begin Routine' code from instr5_code
    # 控制文字的左对齐
    instr5_center_text.alignText = 'left'
    instr5_top_text.alignText = 'left'
    
    # 显示鼠标可见
    win.mouseVisible = True
    
    # 初始化鼠标位置在slider中间
    instr5_mouse.setPos([0, -0.1])
    
    # 设置slider的marker外观
    instr5_slider.marker.size = (0.02, 0.08)  # 设置marker大小（宽度，高度）
    instr5_slider.marker.shape = 'square'     # 设置为方形
    instr5_slider.marker.fillColor = 'white'   # 设置颜色
    
    # 记录slider宽度，用于后续计算
    slider_width = 0.8 
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
        
        # *instr5_center_text* updates
        
        # if instr5_center_text is starting this frame...
        if instr5_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr5_center_text.frameNStart = frameN  # exact frame index
            instr5_center_text.tStart = t  # local t and not account for scr refresh
            instr5_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr5_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr5_center_text.started')
            # update status
            instr5_center_text.status = STARTED
            instr5_center_text.setAutoDraw(True)
        
        # if instr5_center_text is active this frame...
        if instr5_center_text.status == STARTED:
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
        
        # *instr5_left_text* updates
        
        # if instr5_left_text is starting this frame...
        if instr5_left_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr5_left_text.frameNStart = frameN  # exact frame index
            instr5_left_text.tStart = t  # local t and not account for scr refresh
            instr5_left_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr5_left_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr5_left_text.started')
            # update status
            instr5_left_text.status = STARTED
            instr5_left_text.setAutoDraw(True)
        
        # if instr5_left_text is active this frame...
        if instr5_left_text.status == STARTED:
            # update params
            pass
        
        # *instr5_right_text* updates
        
        # if instr5_right_text is starting this frame...
        if instr5_right_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr5_right_text.frameNStart = frameN  # exact frame index
            instr5_right_text.tStart = t  # local t and not account for scr refresh
            instr5_right_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr5_right_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr5_right_text.started')
            # update status
            instr5_right_text.status = STARTED
            instr5_right_text.setAutoDraw(True)
        
        # if instr5_right_text is active this frame...
        if instr5_right_text.status == STARTED:
            # update params
            pass
        
        # *instr5_image* updates
        
        # if instr5_image is starting this frame...
        if instr5_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr5_image.frameNStart = frameN  # exact frame index
            instr5_image.tStart = t  # local t and not account for scr refresh
            instr5_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr5_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr5_image.started')
            # update status
            instr5_image.status = STARTED
            instr5_image.setAutoDraw(True)
        
        # if instr5_image is active this frame...
        if instr5_image.status == STARTED:
            # update params
            pass
        
        # *instr5_slider* updates
        
        # if instr5_slider is starting this frame...
        if instr5_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr5_slider.frameNStart = frameN  # exact frame index
            instr5_slider.tStart = t  # local t and not account for scr refresh
            instr5_slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr5_slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr5_slider.started')
            # update status
            instr5_slider.status = STARTED
            instr5_slider.setAutoDraw(True)
        
        # if instr5_slider is active this frame...
        if instr5_slider.status == STARTED:
            # update params
            pass
        # *instr5_mouse* updates
        
        # if instr5_mouse is starting this frame...
        if instr5_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr5_mouse.frameNStart = frameN  # exact frame index
            instr5_mouse.tStart = t  # local t and not account for scr refresh
            instr5_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr5_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('instr5_mouse.started', t)
            # update status
            instr5_mouse.status = STARTED
            instr5_mouse.mouseClock.reset()
            prevButtonState = instr5_mouse.getPressed()  # if button is down already this ISN'T a new click
        if instr5_mouse.status == STARTED:  # only update if started and not finished!
            buttons = instr5_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
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
        # Run 'Each Frame' code from instr5_code
        # 获取当前鼠标位置
        mouse_pos = instr5_mouse.getPos()
        mouse_x, mouse_y = mouse_pos
        
        # 基于slider的位置和大小，计算slider的边界
        slider_center_x = 0      # slider的中心x坐标
        slider_center_y = -0.1    # slider的中心y坐标
        slider_left = slider_center_x - slider_width/2    # 左边界
        slider_right = slider_center_x + slider_width/2   # 右边界
        slider_bottom = slider_center_y - 0.025          # 下边界
        slider_top = slider_center_y + 0.025             # 上边界
        
        # 检查鼠标是否在slider区域内（四个边界条件）
        mouse_in_slider = (slider_left <= mouse_x <= slider_right and
                           slider_bottom <= mouse_y <= slider_top)
        
        if mouse_in_slider:
            # 当鼠标在slider上时，让marker跟随鼠标移动
            # 将鼠标x坐标转换为slider的0-1范围
            slider_value = (mouse_x - slider_left) / slider_width
        
            # 限制数值在有效范围内，防止超出边界
            slider_value = max(0, min(1, slider_value))
        
            # 设置marker位置，这会让光标跟随鼠标移动
            instr5_slider.markerPos = slider_value
        
            # 检测鼠标点击，点击后结束当前routine
            #if instr5_mouse.getPressed()[0]:
                #continueRoutine = False
        
        
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
    thisExp.addData('instr5_slider.response', instr5_slider.getRating())
    thisExp.addData('instr5_slider.rt', instr5_slider.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('instr5_mouse.x', instr5_mouse.x)
    thisExp.addData('instr5_mouse.y', instr5_mouse.y)
    thisExp.addData('instr5_mouse.leftButton', instr5_mouse.leftButton)
    thisExp.addData('instr5_mouse.midButton', instr5_mouse.midButton)
    thisExp.addData('instr5_mouse.rightButton', instr5_mouse.rightButton)
    thisExp.addData('instr5_mouse.time', instr5_mouse.time)
    # check responses
    if instr5_key_resp.keys in ['', [], None]:  # No response was made
        instr5_key_resp.keys = None
    thisExp.addData('instr5_key_resp.keys',instr5_key_resp.keys)
    if instr5_key_resp.keys != None:  # we had a response
        thisExp.addData('instr5_key_resp.rt', instr5_key_resp.rt)
        thisExp.addData('instr5_key_resp.duration', instr5_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr6" ---
    # create an object to store info about Routine instr6
    instr6 = data.Routine(
        name='instr6',
        components=[instr6_top_text, instr6_center_text, instr6_bottom_text, instr6_left_text, instr6_right_text, instr6_image, instr6_slider, instr6_mouse, instr6_key_resp],
    )
    instr6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instr6_slider.reset()
    # setup some python lists for storing info about the instr6_mouse
    instr6_mouse.x = []
    instr6_mouse.y = []
    instr6_mouse.leftButton = []
    instr6_mouse.midButton = []
    instr6_mouse.rightButton = []
    instr6_mouse.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for instr6_key_resp
    instr6_key_resp.keys = []
    instr6_key_resp.rt = []
    _instr6_key_resp_allKeys = []
    # Run 'Begin Routine' code from instr6_code
    # 控制文字的左对齐
    instr6_center_text.alignText = 'left'
    instr6_top_text.alignText = 'left'
    
    # 显示鼠标可见
    win.mouseVisible = True
    
    # 初始化鼠标位置在slider中间
    instr6_mouse.setPos([0, -0.1])
    
    # 设置slider的marker外观
    instr6_slider.marker.size = (0.02, 0.08)  # 设置marker大小（宽度，高度）
    instr6_slider.marker.shape = 'square'     # 设置为方形
    instr6_slider.marker.fillColor = 'white'   # 设置颜色
    
    # 记录slider宽度，用于后续计算
    slider_width = 0.8
    
    # store start times for instr6
    instr6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr6.tStart = globalClock.getTime(format='float')
    instr6.status = STARTED
    thisExp.addData('instr6.started', instr6.tStart)
    instr6.maxDuration = None
    # keep track of which components have finished
    instr6Components = instr6.components
    for thisComponent in instr6.components:
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
    
    # --- Run Routine "instr6" ---
    instr6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr6_top_text* updates
        
        # if instr6_top_text is starting this frame...
        if instr6_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_top_text.frameNStart = frameN  # exact frame index
            instr6_top_text.tStart = t  # local t and not account for scr refresh
            instr6_top_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_top_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr6_top_text.started')
            # update status
            instr6_top_text.status = STARTED
            instr6_top_text.setAutoDraw(True)
        
        # if instr6_top_text is active this frame...
        if instr6_top_text.status == STARTED:
            # update params
            pass
        
        # *instr6_center_text* updates
        
        # if instr6_center_text is starting this frame...
        if instr6_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_center_text.frameNStart = frameN  # exact frame index
            instr6_center_text.tStart = t  # local t and not account for scr refresh
            instr6_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr6_center_text.started')
            # update status
            instr6_center_text.status = STARTED
            instr6_center_text.setAutoDraw(True)
        
        # if instr6_center_text is active this frame...
        if instr6_center_text.status == STARTED:
            # update params
            pass
        
        # *instr6_bottom_text* updates
        
        # if instr6_bottom_text is starting this frame...
        if instr6_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_bottom_text.frameNStart = frameN  # exact frame index
            instr6_bottom_text.tStart = t  # local t and not account for scr refresh
            instr6_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr6_bottom_text.started')
            # update status
            instr6_bottom_text.status = STARTED
            instr6_bottom_text.setAutoDraw(True)
        
        # if instr6_bottom_text is active this frame...
        if instr6_bottom_text.status == STARTED:
            # update params
            pass
        
        # *instr6_left_text* updates
        
        # if instr6_left_text is starting this frame...
        if instr6_left_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_left_text.frameNStart = frameN  # exact frame index
            instr6_left_text.tStart = t  # local t and not account for scr refresh
            instr6_left_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_left_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr6_left_text.started')
            # update status
            instr6_left_text.status = STARTED
            instr6_left_text.setAutoDraw(True)
        
        # if instr6_left_text is active this frame...
        if instr6_left_text.status == STARTED:
            # update params
            pass
        
        # *instr6_right_text* updates
        
        # if instr6_right_text is starting this frame...
        if instr6_right_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_right_text.frameNStart = frameN  # exact frame index
            instr6_right_text.tStart = t  # local t and not account for scr refresh
            instr6_right_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_right_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr6_right_text.started')
            # update status
            instr6_right_text.status = STARTED
            instr6_right_text.setAutoDraw(True)
        
        # if instr6_right_text is active this frame...
        if instr6_right_text.status == STARTED:
            # update params
            pass
        
        # *instr6_image* updates
        
        # if instr6_image is starting this frame...
        if instr6_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_image.frameNStart = frameN  # exact frame index
            instr6_image.tStart = t  # local t and not account for scr refresh
            instr6_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr6_image.started')
            # update status
            instr6_image.status = STARTED
            instr6_image.setAutoDraw(True)
        
        # if instr6_image is active this frame...
        if instr6_image.status == STARTED:
            # update params
            pass
        
        # *instr6_slider* updates
        
        # if instr6_slider is starting this frame...
        if instr6_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_slider.frameNStart = frameN  # exact frame index
            instr6_slider.tStart = t  # local t and not account for scr refresh
            instr6_slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr6_slider.started')
            # update status
            instr6_slider.status = STARTED
            instr6_slider.setAutoDraw(True)
        
        # if instr6_slider is active this frame...
        if instr6_slider.status == STARTED:
            # update params
            pass
        # *instr6_mouse* updates
        
        # if instr6_mouse is starting this frame...
        if instr6_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_mouse.frameNStart = frameN  # exact frame index
            instr6_mouse.tStart = t  # local t and not account for scr refresh
            instr6_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('instr6_mouse.started', t)
            # update status
            instr6_mouse.status = STARTED
            instr6_mouse.mouseClock.reset()
            prevButtonState = instr6_mouse.getPressed()  # if button is down already this ISN'T a new click
        if instr6_mouse.status == STARTED:  # only update if started and not finished!
            buttons = instr6_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
        
        # *instr6_key_resp* updates
        waitOnFlip = False
        
        # if instr6_key_resp is starting this frame...
        if instr6_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr6_key_resp.frameNStart = frameN  # exact frame index
            instr6_key_resp.tStart = t  # local t and not account for scr refresh
            instr6_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr6_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr6_key_resp.started')
            # update status
            instr6_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr6_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr6_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr6_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr6_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr6_key_resp_allKeys.extend(theseKeys)
            if len(_instr6_key_resp_allKeys):
                instr6_key_resp.keys = _instr6_key_resp_allKeys[-1].name  # just the last key pressed
                instr6_key_resp.rt = _instr6_key_resp_allKeys[-1].rt
                instr6_key_resp.duration = _instr6_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # Run 'Each Frame' code from instr6_code
        # 获取当前鼠标位置
        mouse_pos = instr6_mouse.getPos()
        mouse_x, mouse_y = mouse_pos
        
        # 基于slider的位置和大小，计算slider的边界
        slider_center_x = 0      # slider的中心x坐标
        slider_center_y = -0.1   # slider的中心y坐标
        slider_left = slider_center_x - slider_width/2    # 左边界
        slider_right = slider_center_x + slider_width/2   # 右边界
        slider_bottom = slider_center_y - 0.025          # 下边界
        slider_top = slider_center_y + 0.025             # 上边界
        
        # 检查鼠标是否在slider区域内
        mouse_in_slider = (slider_left <= mouse_x <= slider_right and
                           slider_bottom <= mouse_y <= slider_top)
        
        if mouse_in_slider:
            # 当鼠标在slider上时，让marker跟随鼠标移动
            slider_value = (mouse_x - slider_left) / slider_width
            slider_value = max(0, min(1, slider_value))
            instr6_slider.markerPos = slider_value
        
        
        
        
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
            instr6.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr6" ---
    for thisComponent in instr6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr6
    instr6.tStop = globalClock.getTime(format='float')
    instr6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr6.stopped', instr6.tStop)
    thisExp.addData('instr6_slider.response', instr6_slider.getRating())
    thisExp.addData('instr6_slider.rt', instr6_slider.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('instr6_mouse.x', instr6_mouse.x)
    thisExp.addData('instr6_mouse.y', instr6_mouse.y)
    thisExp.addData('instr6_mouse.leftButton', instr6_mouse.leftButton)
    thisExp.addData('instr6_mouse.midButton', instr6_mouse.midButton)
    thisExp.addData('instr6_mouse.rightButton', instr6_mouse.rightButton)
    thisExp.addData('instr6_mouse.time', instr6_mouse.time)
    # check responses
    if instr6_key_resp.keys in ['', [], None]:  # No response was made
        instr6_key_resp.keys = None
    thisExp.addData('instr6_key_resp.keys',instr6_key_resp.keys)
    if instr6_key_resp.keys != None:  # we had a response
        thisExp.addData('instr6_key_resp.rt', instr6_key_resp.rt)
        thisExp.addData('instr6_key_resp.duration', instr6_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr7" ---
    # create an object to store info about Routine instr7
    instr7 = data.Routine(
        name='instr7',
        components=[instr7_top_text, instr7_center_text, instr7_bottom_text, instr7_left_text, instr7_right_text, instr7_image, instr7_slider, instr7_mouse, instr7_key_resp],
    )
    instr7.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instr7_slider.reset()
    # setup some python lists for storing info about the instr7_mouse
    instr7_mouse.x = []
    instr7_mouse.y = []
    instr7_mouse.leftButton = []
    instr7_mouse.midButton = []
    instr7_mouse.rightButton = []
    instr7_mouse.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for instr7_key_resp
    instr7_key_resp.keys = []
    instr7_key_resp.rt = []
    _instr7_key_resp_allKeys = []
    # Run 'Begin Routine' code from instr7_code
    # 控制文字的左对齐
    instr7_center_text.alignText = 'left'
    instr7_top_text.alignText = 'left'
    
    # 显示鼠标可见
    win.mouseVisible = True
    
    # 初始化鼠标位置在slider中间
    instr7_mouse.setPos([0, -0.1])
    
    # 设置slider的marker外观
    instr7_slider.marker.size = (0.02, 0.08)  # 设置marker大小（宽度，高度）
    instr7_slider.marker.shape = 'square'     # 设置为方形
    instr7_slider.marker.fillColor = 'white'   # 设置颜色
    
    # 记录slider宽度，用于后续计算
    slider_width = 0.8
    
    # store start times for instr7
    instr7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr7.tStart = globalClock.getTime(format='float')
    instr7.status = STARTED
    thisExp.addData('instr7.started', instr7.tStart)
    instr7.maxDuration = None
    # keep track of which components have finished
    instr7Components = instr7.components
    for thisComponent in instr7.components:
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
    
    # --- Run Routine "instr7" ---
    instr7.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr7_top_text* updates
        
        # if instr7_top_text is starting this frame...
        if instr7_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_top_text.frameNStart = frameN  # exact frame index
            instr7_top_text.tStart = t  # local t and not account for scr refresh
            instr7_top_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_top_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr7_top_text.started')
            # update status
            instr7_top_text.status = STARTED
            instr7_top_text.setAutoDraw(True)
        
        # if instr7_top_text is active this frame...
        if instr7_top_text.status == STARTED:
            # update params
            pass
        
        # *instr7_center_text* updates
        
        # if instr7_center_text is starting this frame...
        if instr7_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_center_text.frameNStart = frameN  # exact frame index
            instr7_center_text.tStart = t  # local t and not account for scr refresh
            instr7_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr7_center_text.started')
            # update status
            instr7_center_text.status = STARTED
            instr7_center_text.setAutoDraw(True)
        
        # if instr7_center_text is active this frame...
        if instr7_center_text.status == STARTED:
            # update params
            pass
        
        # *instr7_bottom_text* updates
        
        # if instr7_bottom_text is starting this frame...
        if instr7_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_bottom_text.frameNStart = frameN  # exact frame index
            instr7_bottom_text.tStart = t  # local t and not account for scr refresh
            instr7_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr7_bottom_text.started')
            # update status
            instr7_bottom_text.status = STARTED
            instr7_bottom_text.setAutoDraw(True)
        
        # if instr7_bottom_text is active this frame...
        if instr7_bottom_text.status == STARTED:
            # update params
            pass
        
        # *instr7_left_text* updates
        
        # if instr7_left_text is starting this frame...
        if instr7_left_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_left_text.frameNStart = frameN  # exact frame index
            instr7_left_text.tStart = t  # local t and not account for scr refresh
            instr7_left_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_left_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr7_left_text.started')
            # update status
            instr7_left_text.status = STARTED
            instr7_left_text.setAutoDraw(True)
        
        # if instr7_left_text is active this frame...
        if instr7_left_text.status == STARTED:
            # update params
            pass
        
        # *instr7_right_text* updates
        
        # if instr7_right_text is starting this frame...
        if instr7_right_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_right_text.frameNStart = frameN  # exact frame index
            instr7_right_text.tStart = t  # local t and not account for scr refresh
            instr7_right_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_right_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr7_right_text.started')
            # update status
            instr7_right_text.status = STARTED
            instr7_right_text.setAutoDraw(True)
        
        # if instr7_right_text is active this frame...
        if instr7_right_text.status == STARTED:
            # update params
            pass
        
        # *instr7_image* updates
        
        # if instr7_image is starting this frame...
        if instr7_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_image.frameNStart = frameN  # exact frame index
            instr7_image.tStart = t  # local t and not account for scr refresh
            instr7_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr7_image.started')
            # update status
            instr7_image.status = STARTED
            instr7_image.setAutoDraw(True)
        
        # if instr7_image is active this frame...
        if instr7_image.status == STARTED:
            # update params
            pass
        
        # *instr7_slider* updates
        
        # if instr7_slider is starting this frame...
        if instr7_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_slider.frameNStart = frameN  # exact frame index
            instr7_slider.tStart = t  # local t and not account for scr refresh
            instr7_slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr7_slider.started')
            # update status
            instr7_slider.status = STARTED
            instr7_slider.setAutoDraw(True)
        
        # if instr7_slider is active this frame...
        if instr7_slider.status == STARTED:
            # update params
            pass
        # *instr7_mouse* updates
        
        # if instr7_mouse is starting this frame...
        if instr7_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_mouse.frameNStart = frameN  # exact frame index
            instr7_mouse.tStart = t  # local t and not account for scr refresh
            instr7_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('instr7_mouse.started', t)
            # update status
            instr7_mouse.status = STARTED
            instr7_mouse.mouseClock.reset()
            prevButtonState = instr7_mouse.getPressed()  # if button is down already this ISN'T a new click
        if instr7_mouse.status == STARTED:  # only update if started and not finished!
            buttons = instr7_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
        
        # *instr7_key_resp* updates
        waitOnFlip = False
        
        # if instr7_key_resp is starting this frame...
        if instr7_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr7_key_resp.frameNStart = frameN  # exact frame index
            instr7_key_resp.tStart = t  # local t and not account for scr refresh
            instr7_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr7_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr7_key_resp.started')
            # update status
            instr7_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr7_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr7_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr7_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr7_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr7_key_resp_allKeys.extend(theseKeys)
            if len(_instr7_key_resp_allKeys):
                instr7_key_resp.keys = _instr7_key_resp_allKeys[-1].name  # just the last key pressed
                instr7_key_resp.rt = _instr7_key_resp_allKeys[-1].rt
                instr7_key_resp.duration = _instr7_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # Run 'Each Frame' code from instr7_code
        # 获取当前鼠标位置
        mouse_pos = instr7_mouse.getPos()
        mouse_x, mouse_y = mouse_pos
        
        # 基于slider的位置和大小，计算slider的边界
        slider_center_x = 0      # slider的中心x坐标
        slider_center_y = -0.1   # slider的中心y坐标
        slider_left = slider_center_x - slider_width/2    # 左边界
        slider_right = slider_center_x + slider_width/2   # 右边界
        slider_bottom = slider_center_y - 0.025          # 下边界
        slider_top = slider_center_y + 0.025             # 上边界
        
        # 检查鼠标是否在slider区域内
        mouse_in_slider = (slider_left <= mouse_x <= slider_right and
                           slider_bottom <= mouse_y <= slider_top)
        
        if mouse_in_slider:
            # 当鼠标在slider上时，让marker跟随鼠标移动
            slider_value = (mouse_x - slider_left) / slider_width
            slider_value = max(0, min(1, slider_value))
            instr7_slider.markerPos = slider_value
        
        
        
        
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
            instr7.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr7.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr7" ---
    for thisComponent in instr7.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr7
    instr7.tStop = globalClock.getTime(format='float')
    instr7.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr7.stopped', instr7.tStop)
    thisExp.addData('instr7_slider.response', instr7_slider.getRating())
    thisExp.addData('instr7_slider.rt', instr7_slider.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('instr7_mouse.x', instr7_mouse.x)
    thisExp.addData('instr7_mouse.y', instr7_mouse.y)
    thisExp.addData('instr7_mouse.leftButton', instr7_mouse.leftButton)
    thisExp.addData('instr7_mouse.midButton', instr7_mouse.midButton)
    thisExp.addData('instr7_mouse.rightButton', instr7_mouse.rightButton)
    thisExp.addData('instr7_mouse.time', instr7_mouse.time)
    # check responses
    if instr7_key_resp.keys in ['', [], None]:  # No response was made
        instr7_key_resp.keys = None
    thisExp.addData('instr7_key_resp.keys',instr7_key_resp.keys)
    if instr7_key_resp.keys != None:  # we had a response
        thisExp.addData('instr7_key_resp.rt', instr7_key_resp.rt)
        thisExp.addData('instr7_key_resp.duration', instr7_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr8" ---
    # create an object to store info about Routine instr8
    instr8 = data.Routine(
        name='instr8',
        components=[instr8_top_text, instr8_center_text, instr8_bottom_text, instr8_left_text, instr8_right_text, instr8_image, instr8_slider, instr8_mouse, instr8_key_resp],
    )
    instr8.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instr8_slider.reset()
    # setup some python lists for storing info about the instr8_mouse
    instr8_mouse.x = []
    instr8_mouse.y = []
    instr8_mouse.leftButton = []
    instr8_mouse.midButton = []
    instr8_mouse.rightButton = []
    instr8_mouse.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for instr8_key_resp
    instr8_key_resp.keys = []
    instr8_key_resp.rt = []
    _instr8_key_resp_allKeys = []
    # Run 'Begin Routine' code from instr8_code
    # 控制文字的左对齐
    instr8_center_text.alignText = 'left'
    instr8_top_text.alignText = 'left'
    
    # 显示鼠标可见
    win.mouseVisible = True
    
    # 初始化鼠标位置在slider中间
    instr8_mouse.setPos([0, -0.1])
    
    # 设置slider的marker外观
    instr8_slider.marker.size = (0.02, 0.08)  # 设置marker大小（宽度，高度）
    instr8_slider.marker.shape = 'square'     # 设置为方形
    instr8_slider.marker.fillColor = 'white'   # 设置颜色
    
    # 记录slider宽度，用于后续计算
    slider_width = 0.8
    
    # store start times for instr8
    instr8.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr8.tStart = globalClock.getTime(format='float')
    instr8.status = STARTED
    thisExp.addData('instr8.started', instr8.tStart)
    instr8.maxDuration = None
    # keep track of which components have finished
    instr8Components = instr8.components
    for thisComponent in instr8.components:
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
    
    # --- Run Routine "instr8" ---
    instr8.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr8_top_text* updates
        
        # if instr8_top_text is starting this frame...
        if instr8_top_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_top_text.frameNStart = frameN  # exact frame index
            instr8_top_text.tStart = t  # local t and not account for scr refresh
            instr8_top_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_top_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr8_top_text.started')
            # update status
            instr8_top_text.status = STARTED
            instr8_top_text.setAutoDraw(True)
        
        # if instr8_top_text is active this frame...
        if instr8_top_text.status == STARTED:
            # update params
            pass
        
        # *instr8_center_text* updates
        
        # if instr8_center_text is starting this frame...
        if instr8_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_center_text.frameNStart = frameN  # exact frame index
            instr8_center_text.tStart = t  # local t and not account for scr refresh
            instr8_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr8_center_text.started')
            # update status
            instr8_center_text.status = STARTED
            instr8_center_text.setAutoDraw(True)
        
        # if instr8_center_text is active this frame...
        if instr8_center_text.status == STARTED:
            # update params
            pass
        
        # *instr8_bottom_text* updates
        
        # if instr8_bottom_text is starting this frame...
        if instr8_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_bottom_text.frameNStart = frameN  # exact frame index
            instr8_bottom_text.tStart = t  # local t and not account for scr refresh
            instr8_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr8_bottom_text.started')
            # update status
            instr8_bottom_text.status = STARTED
            instr8_bottom_text.setAutoDraw(True)
        
        # if instr8_bottom_text is active this frame...
        if instr8_bottom_text.status == STARTED:
            # update params
            pass
        
        # *instr8_left_text* updates
        
        # if instr8_left_text is starting this frame...
        if instr8_left_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_left_text.frameNStart = frameN  # exact frame index
            instr8_left_text.tStart = t  # local t and not account for scr refresh
            instr8_left_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_left_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr8_left_text.started')
            # update status
            instr8_left_text.status = STARTED
            instr8_left_text.setAutoDraw(True)
        
        # if instr8_left_text is active this frame...
        if instr8_left_text.status == STARTED:
            # update params
            pass
        
        # *instr8_right_text* updates
        
        # if instr8_right_text is starting this frame...
        if instr8_right_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_right_text.frameNStart = frameN  # exact frame index
            instr8_right_text.tStart = t  # local t and not account for scr refresh
            instr8_right_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_right_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr8_right_text.started')
            # update status
            instr8_right_text.status = STARTED
            instr8_right_text.setAutoDraw(True)
        
        # if instr8_right_text is active this frame...
        if instr8_right_text.status == STARTED:
            # update params
            pass
        
        # *instr8_image* updates
        
        # if instr8_image is starting this frame...
        if instr8_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_image.frameNStart = frameN  # exact frame index
            instr8_image.tStart = t  # local t and not account for scr refresh
            instr8_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr8_image.started')
            # update status
            instr8_image.status = STARTED
            instr8_image.setAutoDraw(True)
        
        # if instr8_image is active this frame...
        if instr8_image.status == STARTED:
            # update params
            pass
        
        # *instr8_slider* updates
        
        # if instr8_slider is starting this frame...
        if instr8_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_slider.frameNStart = frameN  # exact frame index
            instr8_slider.tStart = t  # local t and not account for scr refresh
            instr8_slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr8_slider.started')
            # update status
            instr8_slider.status = STARTED
            instr8_slider.setAutoDraw(True)
        
        # if instr8_slider is active this frame...
        if instr8_slider.status == STARTED:
            # update params
            pass
        # *instr8_mouse* updates
        
        # if instr8_mouse is starting this frame...
        if instr8_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_mouse.frameNStart = frameN  # exact frame index
            instr8_mouse.tStart = t  # local t and not account for scr refresh
            instr8_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('instr8_mouse.started', t)
            # update status
            instr8_mouse.status = STARTED
            instr8_mouse.mouseClock.reset()
            prevButtonState = instr8_mouse.getPressed()  # if button is down already this ISN'T a new click
        if instr8_mouse.status == STARTED:  # only update if started and not finished!
            buttons = instr8_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
        
        # *instr8_key_resp* updates
        waitOnFlip = False
        
        # if instr8_key_resp is starting this frame...
        if instr8_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr8_key_resp.frameNStart = frameN  # exact frame index
            instr8_key_resp.tStart = t  # local t and not account for scr refresh
            instr8_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr8_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr8_key_resp.started')
            # update status
            instr8_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr8_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr8_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr8_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr8_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr8_key_resp_allKeys.extend(theseKeys)
            if len(_instr8_key_resp_allKeys):
                instr8_key_resp.keys = _instr8_key_resp_allKeys[-1].name  # just the last key pressed
                instr8_key_resp.rt = _instr8_key_resp_allKeys[-1].rt
                instr8_key_resp.duration = _instr8_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # Run 'Each Frame' code from instr8_code
        # 获取当前鼠标位置
        mouse_pos = instr8_mouse.getPos()
        mouse_x, mouse_y = mouse_pos
        
        # 基于slider的位置和大小，计算slider的边界
        slider_center_x = 0      # slider的中心x坐标
        slider_center_y = -0.1   # slider的中心y坐标
        slider_left = slider_center_x - slider_width/2    # 左边界
        slider_right = slider_center_x + slider_width/2   # 右边界
        slider_bottom = slider_center_y - 0.025          # 下边界
        slider_top = slider_center_y + 0.025             # 上边界
        
        # 检查鼠标是否在slider区域内
        mouse_in_slider = (slider_left <= mouse_x <= slider_right and
                           slider_bottom <= mouse_y <= slider_top)
        
        if mouse_in_slider:
            # 当鼠标在slider上时，让marker跟随鼠标移动
            slider_value = (mouse_x - slider_left) / slider_width
            slider_value = max(0, min(1, slider_value))
            instr8_slider.markerPos = slider_value
        
        
        
        
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
            instr8.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr8.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr8" ---
    for thisComponent in instr8.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr8
    instr8.tStop = globalClock.getTime(format='float')
    instr8.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr8.stopped', instr8.tStop)
    thisExp.addData('instr8_slider.response', instr8_slider.getRating())
    thisExp.addData('instr8_slider.rt', instr8_slider.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('instr8_mouse.x', instr8_mouse.x)
    thisExp.addData('instr8_mouse.y', instr8_mouse.y)
    thisExp.addData('instr8_mouse.leftButton', instr8_mouse.leftButton)
    thisExp.addData('instr8_mouse.midButton', instr8_mouse.midButton)
    thisExp.addData('instr8_mouse.rightButton', instr8_mouse.rightButton)
    thisExp.addData('instr8_mouse.time', instr8_mouse.time)
    # check responses
    if instr8_key_resp.keys in ['', [], None]:  # No response was made
        instr8_key_resp.keys = None
    thisExp.addData('instr8_key_resp.keys',instr8_key_resp.keys)
    if instr8_key_resp.keys != None:  # we had a response
        thisExp.addData('instr8_key_resp.rt', instr8_key_resp.rt)
        thisExp.addData('instr8_key_resp.duration', instr8_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr8" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr9" ---
    # create an object to store info about Routine instr9
    instr9 = data.Routine(
        name='instr9',
        components=[instr9_center_text, instr9_bottom_text, instr9_key_resp],
    )
    instr9.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instr9_center_text.setText('【重要说明】:\n\n1.在实验中，你将会看到两个人的面孔。\n\n实验分为几个短小的部分，每个部分会专注于其中一人的情绪面孔\n\n2.每一部分中，会出现多张同一人的面孔，\n\n但每张面孔之间可能会有微妙的情绪差异，\n\n请您仔细感受并根据直觉印象进行评分，没有标准答案。\n\n3.在每个部分之间会有短暂的休息时间，\n\n你可以眨眨眼或者稍微放松一下，整个实验会自动引导你完成，\n\n你只需要专注于对每张面孔的情绪进行判断即可。\n\n接下来，我们将在练习中进一步熟悉这个任务。\n')
    # create starting attributes for instr9_key_resp
    instr9_key_resp.keys = []
    instr9_key_resp.rt = []
    _instr9_key_resp_allKeys = []
    # store start times for instr9
    instr9.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr9.tStart = globalClock.getTime(format='float')
    instr9.status = STARTED
    thisExp.addData('instr9.started', instr9.tStart)
    instr9.maxDuration = None
    # keep track of which components have finished
    instr9Components = instr9.components
    for thisComponent in instr9.components:
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
    
    # --- Run Routine "instr9" ---
    instr9.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr9_center_text* updates
        
        # if instr9_center_text is starting this frame...
        if instr9_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr9_center_text.frameNStart = frameN  # exact frame index
            instr9_center_text.tStart = t  # local t and not account for scr refresh
            instr9_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr9_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr9_center_text.started')
            # update status
            instr9_center_text.status = STARTED
            instr9_center_text.setAutoDraw(True)
        
        # if instr9_center_text is active this frame...
        if instr9_center_text.status == STARTED:
            # update params
            pass
        
        # *instr9_bottom_text* updates
        
        # if instr9_bottom_text is starting this frame...
        if instr9_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr9_bottom_text.frameNStart = frameN  # exact frame index
            instr9_bottom_text.tStart = t  # local t and not account for scr refresh
            instr9_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr9_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr9_bottom_text.started')
            # update status
            instr9_bottom_text.status = STARTED
            instr9_bottom_text.setAutoDraw(True)
        
        # if instr9_bottom_text is active this frame...
        if instr9_bottom_text.status == STARTED:
            # update params
            pass
        
        # *instr9_key_resp* updates
        waitOnFlip = False
        
        # if instr9_key_resp is starting this frame...
        if instr9_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr9_key_resp.frameNStart = frameN  # exact frame index
            instr9_key_resp.tStart = t  # local t and not account for scr refresh
            instr9_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr9_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr9_key_resp.started')
            # update status
            instr9_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr9_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr9_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr9_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr9_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr9_key_resp_allKeys.extend(theseKeys)
            if len(_instr9_key_resp_allKeys):
                instr9_key_resp.keys = _instr9_key_resp_allKeys[-1].name  # just the last key pressed
                instr9_key_resp.rt = _instr9_key_resp_allKeys[-1].rt
                instr9_key_resp.duration = _instr9_key_resp_allKeys[-1].duration
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
            instr9.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr9.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr9" ---
    for thisComponent in instr9.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr9
    instr9.tStop = globalClock.getTime(format='float')
    instr9.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr9.stopped', instr9.tStop)
    # check responses
    if instr9_key_resp.keys in ['', [], None]:  # No response was made
        instr9_key_resp.keys = None
    thisExp.addData('instr9_key_resp.keys',instr9_key_resp.keys)
    if instr9_key_resp.keys != None:  # we had a response
        thisExp.addData('instr9_key_resp.rt', instr9_key_resp.rt)
        thisExp.addData('instr9_key_resp.duration', instr9_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr9" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr10" ---
    # create an object to store info about Routine instr10
    instr10 = data.Routine(
        name='instr10',
        components=[instr10_center_text, instr10_bottom_text, instr10_key_resp],
    )
    instr10.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr10_key_resp
    instr10_key_resp.keys = []
    instr10_key_resp.rt = []
    _instr10_key_resp_allKeys = []
    # store start times for instr10
    instr10.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr10.tStart = globalClock.getTime(format='float')
    instr10.status = STARTED
    thisExp.addData('instr10.started', instr10.tStart)
    instr10.maxDuration = None
    # keep track of which components have finished
    instr10Components = instr10.components
    for thisComponent in instr10.components:
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
    
    # --- Run Routine "instr10" ---
    instr10.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr10_center_text* updates
        
        # if instr10_center_text is starting this frame...
        if instr10_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr10_center_text.frameNStart = frameN  # exact frame index
            instr10_center_text.tStart = t  # local t and not account for scr refresh
            instr10_center_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr10_center_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr10_center_text.started')
            # update status
            instr10_center_text.status = STARTED
            instr10_center_text.setAutoDraw(True)
        
        # if instr10_center_text is active this frame...
        if instr10_center_text.status == STARTED:
            # update params
            pass
        
        # *instr10_bottom_text* updates
        
        # if instr10_bottom_text is starting this frame...
        if instr10_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr10_bottom_text.frameNStart = frameN  # exact frame index
            instr10_bottom_text.tStart = t  # local t and not account for scr refresh
            instr10_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr10_bottom_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr10_bottom_text.started')
            # update status
            instr10_bottom_text.status = STARTED
            instr10_bottom_text.setAutoDraw(True)
        
        # if instr10_bottom_text is active this frame...
        if instr10_bottom_text.status == STARTED:
            # update params
            pass
        
        # *instr10_key_resp* updates
        waitOnFlip = False
        
        # if instr10_key_resp is starting this frame...
        if instr10_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr10_key_resp.frameNStart = frameN  # exact frame index
            instr10_key_resp.tStart = t  # local t and not account for scr refresh
            instr10_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr10_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr10_key_resp.started')
            # update status
            instr10_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr10_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr10_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr10_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr10_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr10_key_resp_allKeys.extend(theseKeys)
            if len(_instr10_key_resp_allKeys):
                instr10_key_resp.keys = _instr10_key_resp_allKeys[-1].name  # just the last key pressed
                instr10_key_resp.rt = _instr10_key_resp_allKeys[-1].rt
                instr10_key_resp.duration = _instr10_key_resp_allKeys[-1].duration
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
            instr10.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr10.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr10" ---
    for thisComponent in instr10.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr10
    instr10.tStop = globalClock.getTime(format='float')
    instr10.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr10.stopped', instr10.tStop)
    # check responses
    if instr10_key_resp.keys in ['', [], None]:  # No response was made
        instr10_key_resp.keys = None
    thisExp.addData('instr10_key_resp.keys',instr10_key_resp.keys)
    if instr10_key_resp.keys != None:  # we had a response
        thisExp.addData('instr10_key_resp.rt', instr10_key_resp.rt)
        thisExp.addData('instr10_key_resp.duration', instr10_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr10" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    continue_prac = data.TrialHandler2(
        name='continue_prac',
        nReps=999.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(continue_prac)  # add the loop to the experiment
    thisContinue_prac = continue_prac.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisContinue_prac.rgb)
    if thisContinue_prac != None:
        for paramName in thisContinue_prac:
            globals()[paramName] = thisContinue_prac[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisContinue_prac in continue_prac:
        currentLoop = continue_prac
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisContinue_prac.rgb)
        if thisContinue_prac != None:
            for paramName in thisContinue_prac:
                globals()[paramName] = thisContinue_prac[paramName]
        
        # --- Prepare to start Routine "instr11" ---
        # create an object to store info about Routine instr11
        instr11 = data.Routine(
            name='instr11',
            components=[instr11_center_text, instr11_bottom_text, instr11_key_resp],
        )
        instr11.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for instr11_key_resp
        instr11_key_resp.keys = []
        instr11_key_resp.rt = []
        _instr11_key_resp_allKeys = []
        # store start times for instr11
        instr11.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instr11.tStart = globalClock.getTime(format='float')
        instr11.status = STARTED
        thisExp.addData('instr11.started', instr11.tStart)
        instr11.maxDuration = None
        # keep track of which components have finished
        instr11Components = instr11.components
        for thisComponent in instr11.components:
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
        
        # --- Run Routine "instr11" ---
        # if trial has changed, end Routine now
        if isinstance(continue_prac, data.TrialHandler2) and thisContinue_prac.thisN != continue_prac.thisTrial.thisN:
            continueRoutine = False
        instr11.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *instr11_center_text* updates
            
            # if instr11_center_text is starting this frame...
            if instr11_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr11_center_text.frameNStart = frameN  # exact frame index
                instr11_center_text.tStart = t  # local t and not account for scr refresh
                instr11_center_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr11_center_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr11_center_text.started')
                # update status
                instr11_center_text.status = STARTED
                instr11_center_text.setAutoDraw(True)
            
            # if instr11_center_text is active this frame...
            if instr11_center_text.status == STARTED:
                # update params
                pass
            
            # *instr11_bottom_text* updates
            
            # if instr11_bottom_text is starting this frame...
            if instr11_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr11_bottom_text.frameNStart = frameN  # exact frame index
                instr11_bottom_text.tStart = t  # local t and not account for scr refresh
                instr11_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr11_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr11_bottom_text.started')
                # update status
                instr11_bottom_text.status = STARTED
                instr11_bottom_text.setAutoDraw(True)
            
            # if instr11_bottom_text is active this frame...
            if instr11_bottom_text.status == STARTED:
                # update params
                pass
            
            # *instr11_key_resp* updates
            waitOnFlip = False
            
            # if instr11_key_resp is starting this frame...
            if instr11_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instr11_key_resp.frameNStart = frameN  # exact frame index
                instr11_key_resp.tStart = t  # local t and not account for scr refresh
                instr11_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instr11_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr11_key_resp.started')
                # update status
                instr11_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(instr11_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(instr11_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if instr11_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = instr11_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _instr11_key_resp_allKeys.extend(theseKeys)
                if len(_instr11_key_resp_allKeys):
                    instr11_key_resp.keys = _instr11_key_resp_allKeys[-1].name  # just the last key pressed
                    instr11_key_resp.rt = _instr11_key_resp_allKeys[-1].rt
                    instr11_key_resp.duration = _instr11_key_resp_allKeys[-1].duration
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
                instr11.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instr11.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instr11" ---
        for thisComponent in instr11.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instr11
        instr11.tStop = globalClock.getTime(format='float')
        instr11.tStopRefresh = tThisFlipGlobal
        thisExp.addData('instr11.stopped', instr11.tStop)
        # check responses
        if instr11_key_resp.keys in ['', [], None]:  # No response was made
            instr11_key_resp.keys = None
        continue_prac.addData('instr11_key_resp.keys',instr11_key_resp.keys)
        if instr11_key_resp.keys != None:  # we had a response
            continue_prac.addData('instr11_key_resp.rt', instr11_key_resp.rt)
            continue_prac.addData('instr11_key_resp.duration', instr11_key_resp.duration)
        # the Routine "instr11" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        prac_m = data.TrialHandler2(
            name='prac_m',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('prac_training_m_faces.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(prac_m)  # add the loop to the experiment
        thisPrac_m = prac_m.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_m.rgb)
        if thisPrac_m != None:
            for paramName in thisPrac_m:
                globals()[paramName] = thisPrac_m[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPrac_m in prac_m:
            currentLoop = prac_m
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPrac_m.rgb)
            if thisPrac_m != None:
                for paramName in thisPrac_m:
                    globals()[paramName] = thisPrac_m[paramName]
            
            # --- Prepare to start Routine "prac_begin_m" ---
            # create an object to store info about Routine prac_begin_m
            prac_begin_m = data.Routine(
                name='prac_begin_m',
                components=[prac_begin_text, prac_begin_key_resp],
            )
            prac_begin_m.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for prac_begin_key_resp
            prac_begin_key_resp.keys = []
            prac_begin_key_resp.rt = []
            _prac_begin_key_resp_allKeys = []
            # store start times for prac_begin_m
            prac_begin_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_begin_m.tStart = globalClock.getTime(format='float')
            prac_begin_m.status = STARTED
            thisExp.addData('prac_begin_m.started', prac_begin_m.tStart)
            prac_begin_m.maxDuration = None
            # keep track of which components have finished
            prac_begin_mComponents = prac_begin_m.components
            for thisComponent in prac_begin_m.components:
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
            
            # --- Run Routine "prac_begin_m" ---
            # if trial has changed, end Routine now
            if isinstance(prac_m, data.TrialHandler2) and thisPrac_m.thisN != prac_m.thisTrial.thisN:
                continueRoutine = False
            prac_begin_m.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_begin_text* updates
                
                # if prac_begin_text is starting this frame...
                if prac_begin_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_begin_text.frameNStart = frameN  # exact frame index
                    prac_begin_text.tStart = t  # local t and not account for scr refresh
                    prac_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_begin_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_begin_text.started')
                    # update status
                    prac_begin_text.status = STARTED
                    prac_begin_text.setAutoDraw(True)
                
                # if prac_begin_text is active this frame...
                if prac_begin_text.status == STARTED:
                    # update params
                    pass
                
                # *prac_begin_key_resp* updates
                waitOnFlip = False
                
                # if prac_begin_key_resp is starting this frame...
                if prac_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_begin_key_resp.frameNStart = frameN  # exact frame index
                    prac_begin_key_resp.tStart = t  # local t and not account for scr refresh
                    prac_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_begin_key_resp.started')
                    # update status
                    prac_begin_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_begin_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac_begin_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = prac_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_begin_key_resp_allKeys.extend(theseKeys)
                    if len(_prac_begin_key_resp_allKeys):
                        prac_begin_key_resp.keys = _prac_begin_key_resp_allKeys[-1].name  # just the last key pressed
                        prac_begin_key_resp.rt = _prac_begin_key_resp_allKeys[-1].rt
                        prac_begin_key_resp.duration = _prac_begin_key_resp_allKeys[-1].duration
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
                    prac_begin_m.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_begin_m.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_begin_m" ---
            for thisComponent in prac_begin_m.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_begin_m
            prac_begin_m.tStop = globalClock.getTime(format='float')
            prac_begin_m.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_begin_m.stopped', prac_begin_m.tStop)
            # check responses
            if prac_begin_key_resp.keys in ['', [], None]:  # No response was made
                prac_begin_key_resp.keys = None
            prac_m.addData('prac_begin_key_resp.keys',prac_begin_key_resp.keys)
            if prac_begin_key_resp.keys != None:  # we had a response
                prac_m.addData('prac_begin_key_resp.rt', prac_begin_key_resp.rt)
                prac_m.addData('prac_begin_key_resp.duration', prac_begin_key_resp.duration)
            # the Routine "prac_begin_m" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac_cross_m" ---
            # create an object to store info about Routine prac_cross_m
            prac_cross_m = data.Routine(
                name='prac_cross_m',
                components=[prac_cross_m_text],
            )
            prac_cross_m.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for prac_cross_m
            prac_cross_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_cross_m.tStart = globalClock.getTime(format='float')
            prac_cross_m.status = STARTED
            thisExp.addData('prac_cross_m.started', prac_cross_m.tStart)
            prac_cross_m.maxDuration = None
            # keep track of which components have finished
            prac_cross_mComponents = prac_cross_m.components
            for thisComponent in prac_cross_m.components:
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
            
            # --- Run Routine "prac_cross_m" ---
            # if trial has changed, end Routine now
            if isinstance(prac_m, data.TrialHandler2) and thisPrac_m.thisN != prac_m.thisTrial.thisN:
                continueRoutine = False
            prac_cross_m.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_cross_m_text* updates
                
                # if prac_cross_m_text is starting this frame...
                if prac_cross_m_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_cross_m_text.frameNStart = frameN  # exact frame index
                    prac_cross_m_text.tStart = t  # local t and not account for scr refresh
                    prac_cross_m_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_cross_m_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cross_m_text.started')
                    # update status
                    prac_cross_m_text.status = STARTED
                    prac_cross_m_text.setAutoDraw(True)
                
                # if prac_cross_m_text is active this frame...
                if prac_cross_m_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_cross_m_text is stopping this frame...
                if prac_cross_m_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_cross_m_text.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_cross_m_text.tStop = t  # not accounting for scr refresh
                        prac_cross_m_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_cross_m_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_cross_m_text.stopped')
                        # update status
                        prac_cross_m_text.status = FINISHED
                        prac_cross_m_text.setAutoDraw(False)
                
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
                    prac_cross_m.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_cross_m.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_cross_m" ---
            for thisComponent in prac_cross_m.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_cross_m
            prac_cross_m.tStop = globalClock.getTime(format='float')
            prac_cross_m.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_cross_m.stopped', prac_cross_m.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_cross_m.maxDurationReached:
                routineTimer.addTime(-prac_cross_m.maxDuration)
            elif prac_cross_m.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "prac_resp_m" ---
            # create an object to store info about Routine prac_resp_m
            prac_resp_m = data.Routine(
                name='prac_resp_m',
                components=[prac_resp_m_face_image, prac_resp_m_left_text, prac_resp_m_right_text, prac_resp_m_slider, prac_resp_m_mouse],
            )
            prac_resp_m.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac_resp_m_face_image.setImage(prac_training_face)
            prac_resp_m_slider.reset()
            # setup some python lists for storing info about the prac_resp_m_mouse
            prac_resp_m_mouse.x = []
            prac_resp_m_mouse.y = []
            prac_resp_m_mouse.leftButton = []
            prac_resp_m_mouse.midButton = []
            prac_resp_m_mouse.rightButton = []
            prac_resp_m_mouse.time = []
            gotValidClick = False  # until a click is received
            # Run 'Begin Routine' code from prac_resp_m_code
            ##########
            # 以下是组件基本设置的代码：
            # 鼠标设置
            win.mouseVisible = True
            
            # 初始化鼠标位置在slider中间
            prac_resp_m_mouse.setPos([0, -0.25])
            
            # 设置男性模特slider的marker外观
            prac_resp_m_slider.marker.size = (0.02, 0.08)  # 设置marker大小
            prac_resp_m_slider.marker.shape = 'square'     # 设置为方形
            prac_resp_m_slider.marker.fillColor = 'white'   # 设置颜色
            
            # 记录slider宽度，用于后续计算
            slider_width = 0.8
            
            ############
            # 以下代码用于保存和记录数据
            
            # 试次计数和性别标记
            prac_cont_trial += 1
            prac_current_gender = 'male'
            
            # 记录这个试次开始的时间，用于后续计算反应时
            start_time = core.getTime()
            prac_cont_resp = None  # 初始化反应值
            prac_cont_rt = None    # 初始化反应时间
            
            # 创建实时反馈文本对象
            feedback_text = visual.TextStim(
                win=win,
                text='50%开心（50%悲伤）',  # 初始显示文本
                font='Arial',
                pos=[0, -0.35],  # 位置在slider下方
                height=0.03,
                color='white'
            )
            
            # 更新previous_gender变量，
            # 为下一个试次的mini-block检测做准备
            prac_previous_gender = prac_current_gender
            # store start times for prac_resp_m
            prac_resp_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_resp_m.tStart = globalClock.getTime(format='float')
            prac_resp_m.status = STARTED
            thisExp.addData('prac_resp_m.started', prac_resp_m.tStart)
            prac_resp_m.maxDuration = None
            # keep track of which components have finished
            prac_resp_mComponents = prac_resp_m.components
            for thisComponent in prac_resp_m.components:
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
            
            # --- Run Routine "prac_resp_m" ---
            # if trial has changed, end Routine now
            if isinstance(prac_m, data.TrialHandler2) and thisPrac_m.thisN != prac_m.thisTrial.thisN:
                continueRoutine = False
            prac_resp_m.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_resp_m_face_image* updates
                
                # if prac_resp_m_face_image is starting this frame...
                if prac_resp_m_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_m_face_image.frameNStart = frameN  # exact frame index
                    prac_resp_m_face_image.tStart = t  # local t and not account for scr refresh
                    prac_resp_m_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_m_face_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_resp_m_face_image.started')
                    # update status
                    prac_resp_m_face_image.status = STARTED
                    prac_resp_m_face_image.setAutoDraw(True)
                
                # if prac_resp_m_face_image is active this frame...
                if prac_resp_m_face_image.status == STARTED:
                    # update params
                    pass
                
                # *prac_resp_m_left_text* updates
                
                # if prac_resp_m_left_text is starting this frame...
                if prac_resp_m_left_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_m_left_text.frameNStart = frameN  # exact frame index
                    prac_resp_m_left_text.tStart = t  # local t and not account for scr refresh
                    prac_resp_m_left_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_m_left_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_resp_m_left_text.started')
                    # update status
                    prac_resp_m_left_text.status = STARTED
                    prac_resp_m_left_text.setAutoDraw(True)
                
                # if prac_resp_m_left_text is active this frame...
                if prac_resp_m_left_text.status == STARTED:
                    # update params
                    pass
                
                # *prac_resp_m_right_text* updates
                
                # if prac_resp_m_right_text is starting this frame...
                if prac_resp_m_right_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_m_right_text.frameNStart = frameN  # exact frame index
                    prac_resp_m_right_text.tStart = t  # local t and not account for scr refresh
                    prac_resp_m_right_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_m_right_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_resp_m_right_text.started')
                    # update status
                    prac_resp_m_right_text.status = STARTED
                    prac_resp_m_right_text.setAutoDraw(True)
                
                # if prac_resp_m_right_text is active this frame...
                if prac_resp_m_right_text.status == STARTED:
                    # update params
                    pass
                
                # *prac_resp_m_slider* updates
                
                # if prac_resp_m_slider is starting this frame...
                if prac_resp_m_slider.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_m_slider.frameNStart = frameN  # exact frame index
                    prac_resp_m_slider.tStart = t  # local t and not account for scr refresh
                    prac_resp_m_slider.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_m_slider, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_resp_m_slider.started')
                    # update status
                    prac_resp_m_slider.status = STARTED
                    prac_resp_m_slider.setAutoDraw(True)
                
                # if prac_resp_m_slider is active this frame...
                if prac_resp_m_slider.status == STARTED:
                    # update params
                    pass
                # *prac_resp_m_mouse* updates
                
                # if prac_resp_m_mouse is starting this frame...
                if prac_resp_m_mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_m_mouse.frameNStart = frameN  # exact frame index
                    prac_resp_m_mouse.tStart = t  # local t and not account for scr refresh
                    prac_resp_m_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_m_mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('prac_resp_m_mouse.started', t)
                    # update status
                    prac_resp_m_mouse.status = STARTED
                    prac_resp_m_mouse.mouseClock.reset()
                    prevButtonState = prac_resp_m_mouse.getPressed()  # if button is down already this ISN'T a new click
                if prac_resp_m_mouse.status == STARTED:  # only update if started and not finished!
                    buttons = prac_resp_m_mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                # Run 'Each Frame' code from prac_resp_m_code
                ##########
                 
                # 获取当前鼠标位置
                mouse_pos = prac_resp_m_mouse.getPos()  
                mouse_x, mouse_y = mouse_pos
                
                # 基于slider的位置和大小，计算slider的边界
                slider_center_x = 0
                slider_center_y = -0.25
                slider_left = slider_center_x - slider_width/2
                slider_right = slider_center_x + slider_width/2
                slider_bottom = slider_center_y - 0.025
                slider_top = slider_center_y + 0.025
                
                # 检查鼠标是否在slider区域内
                mouse_in_slider = (slider_left <= mouse_x <= slider_right and
                                   slider_bottom <= mouse_y <= slider_top)
                
                # 处理slider交互逻辑
                if mouse_in_slider:
                    # 计算slider值，修正边界问题
                    slider_value = (mouse_x - slider_left) / slider_width
                    
                    # 关键修改：扩大边界范围，允许0和1
                    if slider_value < 0.01:
                        slider_value = 0.0
                    elif slider_value > 0.99:
                        slider_value = 1.0
                    else:
                        slider_value = max(0, min(1, slider_value))
                   
                    # 更新slider marker位置，使用男性模特的slider组件
                    prac_resp_m_slider.markerPos = slider_value
                   
                    # 计算并更新实时反馈文本
                    sad_percent = int(slider_value * 100)
                    happy_percent = 100 - sad_percent
                    feedback_text.text = f'{happy_percent}%开心（{sad_percent}%悲伤）'
                   
                    # 检测鼠标点击
                    #使用男性模特的鼠标组件
                    if prac_resp_m_mouse.getPressed()[0]:
                        # 保存反应数据
                        prac_cont_resp = round(slider_value, 2)
                        prac_cont_rt = round(core.getTime() - start_time, 3)
                        continueRoutine = False
                else:
                    # 鼠标不在slider上时，显示当前marker位置对应的文本
                    current_value = prac_resp_m_slider.markerPos if prac_resp_m_slider.markerPos is not None else 0.5
                    sad_percent = int(current_value * 100)
                    happy_percent = 100 - sad_percent
                    feedback_text.text = f'{happy_percent}%开心（{sad_percent}%悲伤）'
                
                feedback_text.draw()
                
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
                    prac_resp_m.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_resp_m.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_resp_m" ---
            for thisComponent in prac_resp_m.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_resp_m
            prac_resp_m.tStop = globalClock.getTime(format='float')
            prac_resp_m.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_resp_m.stopped', prac_resp_m.tStop)
            prac_m.addData('prac_resp_m_slider.response', prac_resp_m_slider.getRating())
            prac_m.addData('prac_resp_m_slider.rt', prac_resp_m_slider.getRT())
            # store data for prac_m (TrialHandler)
            prac_m.addData('prac_resp_m_mouse.x', prac_resp_m_mouse.x)
            prac_m.addData('prac_resp_m_mouse.y', prac_resp_m_mouse.y)
            prac_m.addData('prac_resp_m_mouse.leftButton', prac_resp_m_mouse.leftButton)
            prac_m.addData('prac_resp_m_mouse.midButton', prac_resp_m_mouse.midButton)
            prac_m.addData('prac_resp_m_mouse.rightButton', prac_resp_m_mouse.rightButton)
            prac_m.addData('prac_resp_m_mouse.time', prac_resp_m_mouse.time)
            # Run 'End Routine' code from prac_resp_m_code
            #########
            # 保存详细的试次数据
            
            # 重新练习次数（block数）
            thisExp.addData('prac_cont_block', prac_cont_block)
            # 总试次编号（trial数）
            thisExp.addData('prac_cont_trial', prac_cont_trial)
            # 当前模特性别
            thisExp.addData('prac_current_gender', prac_current_gender)
            # 被试连续评分值
            thisExp.addData('prac_cont_resp', prac_cont_resp)
            # 反应时间
            thisExp.addData('prac_cont_rt', prac_cont_rt)
            # 进入下一条数据记录
            thisExp.nextEntry()
            
            # 隐藏鼠标
            win.mouseVisible = False  
            # the Routine "prac_resp_m" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac_blank_m" ---
            # create an object to store info about Routine prac_blank_m
            prac_blank_m = data.Routine(
                name='prac_blank_m',
                components=[prac_blank_text],
            )
            prac_blank_m.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for prac_blank_m
            prac_blank_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_blank_m.tStart = globalClock.getTime(format='float')
            prac_blank_m.status = STARTED
            thisExp.addData('prac_blank_m.started', prac_blank_m.tStart)
            prac_blank_m.maxDuration = None
            # keep track of which components have finished
            prac_blank_mComponents = prac_blank_m.components
            for thisComponent in prac_blank_m.components:
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
            
            # --- Run Routine "prac_blank_m" ---
            # if trial has changed, end Routine now
            if isinstance(prac_m, data.TrialHandler2) and thisPrac_m.thisN != prac_m.thisTrial.thisN:
                continueRoutine = False
            prac_blank_m.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_blank_text* updates
                
                # if prac_blank_text is starting this frame...
                if prac_blank_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_blank_text.frameNStart = frameN  # exact frame index
                    prac_blank_text.tStart = t  # local t and not account for scr refresh
                    prac_blank_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_blank_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_blank_text.started')
                    # update status
                    prac_blank_text.status = STARTED
                    prac_blank_text.setAutoDraw(True)
                
                # if prac_blank_text is active this frame...
                if prac_blank_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_blank_text is stopping this frame...
                if prac_blank_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_blank_text.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_blank_text.tStop = t  # not accounting for scr refresh
                        prac_blank_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_blank_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_blank_text.stopped')
                        # update status
                        prac_blank_text.status = FINISHED
                        prac_blank_text.setAutoDraw(False)
                
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
                    prac_blank_m.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_blank_m.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_blank_m" ---
            for thisComponent in prac_blank_m.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_blank_m
            prac_blank_m.tStop = globalClock.getTime(format='float')
            prac_blank_m.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_blank_m.stopped', prac_blank_m.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_blank_m.maxDurationReached:
                routineTimer.addTime(-prac_blank_m.maxDuration)
            elif prac_blank_m.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'prac_m'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "prac_break_m" ---
        # create an object to store info about Routine prac_break_m
        prac_break_m = data.Routine(
            name='prac_break_m',
            components=[prac_m_break_text],
        )
        prac_break_m.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for prac_break_m
        prac_break_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_break_m.tStart = globalClock.getTime(format='float')
        prac_break_m.status = STARTED
        thisExp.addData('prac_break_m.started', prac_break_m.tStart)
        prac_break_m.maxDuration = None
        # keep track of which components have finished
        prac_break_mComponents = prac_break_m.components
        for thisComponent in prac_break_m.components:
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
        
        # --- Run Routine "prac_break_m" ---
        # if trial has changed, end Routine now
        if isinstance(continue_prac, data.TrialHandler2) and thisContinue_prac.thisN != continue_prac.thisTrial.thisN:
            continueRoutine = False
        prac_break_m.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_m_break_text* updates
            
            # if prac_m_break_text is starting this frame...
            if prac_m_break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_m_break_text.frameNStart = frameN  # exact frame index
                prac_m_break_text.tStart = t  # local t and not account for scr refresh
                prac_m_break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_m_break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_m_break_text.started')
                # update status
                prac_m_break_text.status = STARTED
                prac_m_break_text.setAutoDraw(True)
            
            # if prac_m_break_text is active this frame...
            if prac_m_break_text.status == STARTED:
                # update params
                pass
            
            # if prac_m_break_text is stopping this frame...
            if prac_m_break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_m_break_text.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_m_break_text.tStop = t  # not accounting for scr refresh
                    prac_m_break_text.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_m_break_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_m_break_text.stopped')
                    # update status
                    prac_m_break_text.status = FINISHED
                    prac_m_break_text.setAutoDraw(False)
            
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
                prac_break_m.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_break_m.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_break_m" ---
        for thisComponent in prac_break_m.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_break_m
        prac_break_m.tStop = globalClock.getTime(format='float')
        prac_break_m.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_break_m.stopped', prac_break_m.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_break_m.maxDurationReached:
            routineTimer.addTime(-prac_break_m.maxDuration)
        elif prac_break_m.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # set up handler to look after randomisation of conditions etc
        prac_f = data.TrialHandler2(
            name='prac_f',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('prac_training_f_faces.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(prac_f)  # add the loop to the experiment
        thisPrac_f = prac_f.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_f.rgb)
        if thisPrac_f != None:
            for paramName in thisPrac_f:
                globals()[paramName] = thisPrac_f[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPrac_f in prac_f:
            currentLoop = prac_f
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPrac_f.rgb)
            if thisPrac_f != None:
                for paramName in thisPrac_f:
                    globals()[paramName] = thisPrac_f[paramName]
            
            # --- Prepare to start Routine "prac_begin_f" ---
            # create an object to store info about Routine prac_begin_f
            prac_begin_f = data.Routine(
                name='prac_begin_f',
                components=[prac_begin_f_text, prac_begin_f_key_resp],
            )
            prac_begin_f.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for prac_begin_f_key_resp
            prac_begin_f_key_resp.keys = []
            prac_begin_f_key_resp.rt = []
            _prac_begin_f_key_resp_allKeys = []
            # store start times for prac_begin_f
            prac_begin_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_begin_f.tStart = globalClock.getTime(format='float')
            prac_begin_f.status = STARTED
            thisExp.addData('prac_begin_f.started', prac_begin_f.tStart)
            prac_begin_f.maxDuration = None
            # keep track of which components have finished
            prac_begin_fComponents = prac_begin_f.components
            for thisComponent in prac_begin_f.components:
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
            
            # --- Run Routine "prac_begin_f" ---
            # if trial has changed, end Routine now
            if isinstance(prac_f, data.TrialHandler2) and thisPrac_f.thisN != prac_f.thisTrial.thisN:
                continueRoutine = False
            prac_begin_f.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_begin_f_text* updates
                
                # if prac_begin_f_text is starting this frame...
                if prac_begin_f_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_begin_f_text.frameNStart = frameN  # exact frame index
                    prac_begin_f_text.tStart = t  # local t and not account for scr refresh
                    prac_begin_f_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_begin_f_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_begin_f_text.started')
                    # update status
                    prac_begin_f_text.status = STARTED
                    prac_begin_f_text.setAutoDraw(True)
                
                # if prac_begin_f_text is active this frame...
                if prac_begin_f_text.status == STARTED:
                    # update params
                    pass
                
                # *prac_begin_f_key_resp* updates
                waitOnFlip = False
                
                # if prac_begin_f_key_resp is starting this frame...
                if prac_begin_f_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_begin_f_key_resp.frameNStart = frameN  # exact frame index
                    prac_begin_f_key_resp.tStart = t  # local t and not account for scr refresh
                    prac_begin_f_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_begin_f_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_begin_f_key_resp.started')
                    # update status
                    prac_begin_f_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_begin_f_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_begin_f_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if prac_begin_f_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = prac_begin_f_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_begin_f_key_resp_allKeys.extend(theseKeys)
                    if len(_prac_begin_f_key_resp_allKeys):
                        prac_begin_f_key_resp.keys = _prac_begin_f_key_resp_allKeys[-1].name  # just the last key pressed
                        prac_begin_f_key_resp.rt = _prac_begin_f_key_resp_allKeys[-1].rt
                        prac_begin_f_key_resp.duration = _prac_begin_f_key_resp_allKeys[-1].duration
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
                    prac_begin_f.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_begin_f.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_begin_f" ---
            for thisComponent in prac_begin_f.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_begin_f
            prac_begin_f.tStop = globalClock.getTime(format='float')
            prac_begin_f.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_begin_f.stopped', prac_begin_f.tStop)
            # check responses
            if prac_begin_f_key_resp.keys in ['', [], None]:  # No response was made
                prac_begin_f_key_resp.keys = None
            prac_f.addData('prac_begin_f_key_resp.keys',prac_begin_f_key_resp.keys)
            if prac_begin_f_key_resp.keys != None:  # we had a response
                prac_f.addData('prac_begin_f_key_resp.rt', prac_begin_f_key_resp.rt)
                prac_f.addData('prac_begin_f_key_resp.duration', prac_begin_f_key_resp.duration)
            # the Routine "prac_begin_f" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac_cross_f" ---
            # create an object to store info about Routine prac_cross_f
            prac_cross_f = data.Routine(
                name='prac_cross_f',
                components=[prac_cross_f_text],
            )
            prac_cross_f.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for prac_cross_f
            prac_cross_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_cross_f.tStart = globalClock.getTime(format='float')
            prac_cross_f.status = STARTED
            thisExp.addData('prac_cross_f.started', prac_cross_f.tStart)
            prac_cross_f.maxDuration = None
            # keep track of which components have finished
            prac_cross_fComponents = prac_cross_f.components
            for thisComponent in prac_cross_f.components:
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
            
            # --- Run Routine "prac_cross_f" ---
            # if trial has changed, end Routine now
            if isinstance(prac_f, data.TrialHandler2) and thisPrac_f.thisN != prac_f.thisTrial.thisN:
                continueRoutine = False
            prac_cross_f.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_cross_f_text* updates
                
                # if prac_cross_f_text is starting this frame...
                if prac_cross_f_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_cross_f_text.frameNStart = frameN  # exact frame index
                    prac_cross_f_text.tStart = t  # local t and not account for scr refresh
                    prac_cross_f_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_cross_f_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_cross_f_text.started')
                    # update status
                    prac_cross_f_text.status = STARTED
                    prac_cross_f_text.setAutoDraw(True)
                
                # if prac_cross_f_text is active this frame...
                if prac_cross_f_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_cross_f_text is stopping this frame...
                if prac_cross_f_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_cross_f_text.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_cross_f_text.tStop = t  # not accounting for scr refresh
                        prac_cross_f_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_cross_f_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_cross_f_text.stopped')
                        # update status
                        prac_cross_f_text.status = FINISHED
                        prac_cross_f_text.setAutoDraw(False)
                
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
                    prac_cross_f.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_cross_f.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_cross_f" ---
            for thisComponent in prac_cross_f.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_cross_f
            prac_cross_f.tStop = globalClock.getTime(format='float')
            prac_cross_f.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_cross_f.stopped', prac_cross_f.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_cross_f.maxDurationReached:
                routineTimer.addTime(-prac_cross_f.maxDuration)
            elif prac_cross_f.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "prac_resp_f" ---
            # create an object to store info about Routine prac_resp_f
            prac_resp_f = data.Routine(
                name='prac_resp_f',
                components=[prac_resp_f_face_image, prac_resp_f_left_text, prac_resp_f_right_text, prac_resp_f_slider, prac_resp_f_mouse],
            )
            prac_resp_f.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac_resp_f_face_image.setImage(prac_training_face)
            prac_resp_f_slider.reset()
            # setup some python lists for storing info about the prac_resp_f_mouse
            prac_resp_f_mouse.x = []
            prac_resp_f_mouse.y = []
            prac_resp_f_mouse.leftButton = []
            prac_resp_f_mouse.midButton = []
            prac_resp_f_mouse.rightButton = []
            prac_resp_f_mouse.time = []
            gotValidClick = False  # until a click is received
            # Run 'Begin Routine' code from prac_resp_f_code
            ##########
            # 以下是组件基本设置的代码：
            
            # 鼠标设置
            win.mouseVisible = True
            # 初始化鼠标位置在slider中间
            prac_resp_f_mouse.setPos([0, -0.25])
            
            # 设置slider的marker外观
            prac_resp_f_slider.marker.size = (0.02, 0.08)
            prac_resp_f_slider.marker.shape = 'square'
            prac_resp_f_slider.marker.fillColor = 'white'
            
            # 记录slider宽度，用于后续计算
            slider_width = 0.8
            
            ############
            # 以下代码用于保存和记录数据
            
            # 试次计数和性别标记
            prac_cont_trial += 1
            prac_current_gender = 'female'
            
            # 记录这个试次开始的时间，用于后续计算反应时
            start_time = core.getTime()
            prac_cont_resp = None  # 初始化反应值
            prac_cont_rt = None    # 初始化反应时间
            
            # 创建实时反馈文本对象
            feedback_text = visual.TextStim(
                win=win,
                text='50%开心（50%悲伤）',  # 初始显示文本
                font='Arial',
                pos=[0, -0.35],  # 位置在slider下方
                height=0.03,
                color='white'
            )
            
            # 更新previous_gender变量，
            # 为下一个试次的mini-block检测做准备
            prac_previous_gender = prac_current_gender
            
            # store start times for prac_resp_f
            prac_resp_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_resp_f.tStart = globalClock.getTime(format='float')
            prac_resp_f.status = STARTED
            thisExp.addData('prac_resp_f.started', prac_resp_f.tStart)
            prac_resp_f.maxDuration = None
            # keep track of which components have finished
            prac_resp_fComponents = prac_resp_f.components
            for thisComponent in prac_resp_f.components:
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
            
            # --- Run Routine "prac_resp_f" ---
            # if trial has changed, end Routine now
            if isinstance(prac_f, data.TrialHandler2) and thisPrac_f.thisN != prac_f.thisTrial.thisN:
                continueRoutine = False
            prac_resp_f.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_resp_f_face_image* updates
                
                # if prac_resp_f_face_image is starting this frame...
                if prac_resp_f_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_f_face_image.frameNStart = frameN  # exact frame index
                    prac_resp_f_face_image.tStart = t  # local t and not account for scr refresh
                    prac_resp_f_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_f_face_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_resp_f_face_image.started')
                    # update status
                    prac_resp_f_face_image.status = STARTED
                    prac_resp_f_face_image.setAutoDraw(True)
                
                # if prac_resp_f_face_image is active this frame...
                if prac_resp_f_face_image.status == STARTED:
                    # update params
                    pass
                
                # *prac_resp_f_left_text* updates
                
                # if prac_resp_f_left_text is starting this frame...
                if prac_resp_f_left_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_f_left_text.frameNStart = frameN  # exact frame index
                    prac_resp_f_left_text.tStart = t  # local t and not account for scr refresh
                    prac_resp_f_left_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_f_left_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_resp_f_left_text.started')
                    # update status
                    prac_resp_f_left_text.status = STARTED
                    prac_resp_f_left_text.setAutoDraw(True)
                
                # if prac_resp_f_left_text is active this frame...
                if prac_resp_f_left_text.status == STARTED:
                    # update params
                    pass
                
                # *prac_resp_f_right_text* updates
                
                # if prac_resp_f_right_text is starting this frame...
                if prac_resp_f_right_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_f_right_text.frameNStart = frameN  # exact frame index
                    prac_resp_f_right_text.tStart = t  # local t and not account for scr refresh
                    prac_resp_f_right_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_f_right_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_resp_f_right_text.started')
                    # update status
                    prac_resp_f_right_text.status = STARTED
                    prac_resp_f_right_text.setAutoDraw(True)
                
                # if prac_resp_f_right_text is active this frame...
                if prac_resp_f_right_text.status == STARTED:
                    # update params
                    pass
                
                # *prac_resp_f_slider* updates
                
                # if prac_resp_f_slider is starting this frame...
                if prac_resp_f_slider.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_f_slider.frameNStart = frameN  # exact frame index
                    prac_resp_f_slider.tStart = t  # local t and not account for scr refresh
                    prac_resp_f_slider.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_f_slider, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_resp_f_slider.started')
                    # update status
                    prac_resp_f_slider.status = STARTED
                    prac_resp_f_slider.setAutoDraw(True)
                
                # if prac_resp_f_slider is active this frame...
                if prac_resp_f_slider.status == STARTED:
                    # update params
                    pass
                # *prac_resp_f_mouse* updates
                
                # if prac_resp_f_mouse is starting this frame...
                if prac_resp_f_mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_resp_f_mouse.frameNStart = frameN  # exact frame index
                    prac_resp_f_mouse.tStart = t  # local t and not account for scr refresh
                    prac_resp_f_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_resp_f_mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('prac_resp_f_mouse.started', t)
                    # update status
                    prac_resp_f_mouse.status = STARTED
                    prac_resp_f_mouse.mouseClock.reset()
                    prevButtonState = prac_resp_f_mouse.getPressed()  # if button is down already this ISN'T a new click
                if prac_resp_f_mouse.status == STARTED:  # only update if started and not finished!
                    buttons = prac_resp_f_mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                # Run 'Each Frame' code from prac_resp_f_code
                ##########
                
                # 获取当前鼠标位置
                mouse_pos = prac_resp_f_mouse.getPos()
                mouse_x, mouse_y = mouse_pos
                
                # 基于slider的位置和大小，计算slider的边界
                slider_center_x = 0
                slider_center_y = -0.25
                slider_left = slider_center_x - slider_width/2
                slider_right = slider_center_x + slider_width/2
                slider_bottom = slider_center_y - 0.025
                slider_top = slider_center_y + 0.025
                
                # 检查鼠标是否在slider区域内
                mouse_in_slider = (slider_left <= mouse_x <= slider_right and
                                   slider_bottom <= mouse_y <= slider_top)
                
                # 处理slider交互逻辑
                if mouse_in_slider:
                    # 计算slider值，修正边界问题
                    slider_value = (mouse_x - slider_left) / slider_width
                
                    # 关键修改：扩大边界范围，允许0和1
                    if slider_value < 0.01:
                        slider_value = 0.0
                    elif slider_value > 0.99:
                        slider_value = 1.0
                    else:
                        slider_value = max(0, min(1, slider_value))
                
                    # 更新slider marker位置 - 使用女性模特的slider组件
                    prac_resp_f_slider.markerPos = slider_value
                
                    # 计算并更新实时反馈文本
                    sad_percent = int(slider_value * 100)
                    happy_percent = 100 - sad_percent
                    feedback_text.text = f'{happy_percent}%开心（{sad_percent}%悲伤）'
                
                    # 检测鼠标点击 - 使用女性模特的鼠标组件
                    if prac_resp_f_mouse.getPressed()[0]:
                        # 保存反应数据
                        prac_cont_resp = round(slider_value, 2)
                        prac_cont_rt = round(core.getTime() - start_time, 3)
                        continueRoutine = False
                else:
                    # 鼠标不在slider上时，显示当前marker位置对应的文本
                    current_value = prac_resp_f_slider.markerPos if prac_resp_f_slider.markerPos is not None else 0.5
                    sad_percent = int(current_value * 100)
                    happy_percent = 100 - sad_percent
                    feedback_text.text = f'{happy_percent}%开心（{sad_percent}%悲伤）'
                
                feedback_text.draw()
                
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
                    prac_resp_f.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_resp_f.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_resp_f" ---
            for thisComponent in prac_resp_f.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_resp_f
            prac_resp_f.tStop = globalClock.getTime(format='float')
            prac_resp_f.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_resp_f.stopped', prac_resp_f.tStop)
            prac_f.addData('prac_resp_f_slider.response', prac_resp_f_slider.getRating())
            prac_f.addData('prac_resp_f_slider.rt', prac_resp_f_slider.getRT())
            # store data for prac_f (TrialHandler)
            prac_f.addData('prac_resp_f_mouse.x', prac_resp_f_mouse.x)
            prac_f.addData('prac_resp_f_mouse.y', prac_resp_f_mouse.y)
            prac_f.addData('prac_resp_f_mouse.leftButton', prac_resp_f_mouse.leftButton)
            prac_f.addData('prac_resp_f_mouse.midButton', prac_resp_f_mouse.midButton)
            prac_f.addData('prac_resp_f_mouse.rightButton', prac_resp_f_mouse.rightButton)
            prac_f.addData('prac_resp_f_mouse.time', prac_resp_f_mouse.time)
            # Run 'End Routine' code from prac_resp_f_code
            #########
            # 以下代码是为了保存数据
            
            # block数
            thisExp.addData('prac_cont_block', prac_cont_block)
            # trial数
            thisExp.addData('prac_cont_trial', prac_cont_trial)
            # 当前模特性别
            thisExp.addData('prac_current_gender', prac_current_gender)
            # 被试连续评分值
            thisExp.addData('prac_cont_resp', prac_cont_resp)
            # 反应时间
            thisExp.addData('prac_cont_rt', prac_cont_rt)
            # 进入下一条数据记录
            thisExp.nextEntry()
            
            
            # 隐藏鼠标
            win.mouseVisible = False  
            # the Routine "prac_resp_f" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac_blank_f" ---
            # create an object to store info about Routine prac_blank_f
            prac_blank_f = data.Routine(
                name='prac_blank_f',
                components=[prac_blank_f_text],
            )
            prac_blank_f.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for prac_blank_f
            prac_blank_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_blank_f.tStart = globalClock.getTime(format='float')
            prac_blank_f.status = STARTED
            thisExp.addData('prac_blank_f.started', prac_blank_f.tStart)
            prac_blank_f.maxDuration = None
            # keep track of which components have finished
            prac_blank_fComponents = prac_blank_f.components
            for thisComponent in prac_blank_f.components:
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
            
            # --- Run Routine "prac_blank_f" ---
            # if trial has changed, end Routine now
            if isinstance(prac_f, data.TrialHandler2) and thisPrac_f.thisN != prac_f.thisTrial.thisN:
                continueRoutine = False
            prac_blank_f.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_blank_f_text* updates
                
                # if prac_blank_f_text is starting this frame...
                if prac_blank_f_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_blank_f_text.frameNStart = frameN  # exact frame index
                    prac_blank_f_text.tStart = t  # local t and not account for scr refresh
                    prac_blank_f_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_blank_f_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_blank_f_text.started')
                    # update status
                    prac_blank_f_text.status = STARTED
                    prac_blank_f_text.setAutoDraw(True)
                
                # if prac_blank_f_text is active this frame...
                if prac_blank_f_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_blank_f_text is stopping this frame...
                if prac_blank_f_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_blank_f_text.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_blank_f_text.tStop = t  # not accounting for scr refresh
                        prac_blank_f_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_blank_f_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_blank_f_text.stopped')
                        # update status
                        prac_blank_f_text.status = FINISHED
                        prac_blank_f_text.setAutoDraw(False)
                
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
                    prac_blank_f.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_blank_f.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_blank_f" ---
            for thisComponent in prac_blank_f.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_blank_f
            prac_blank_f.tStop = globalClock.getTime(format='float')
            prac_blank_f.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_blank_f.stopped', prac_blank_f.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_blank_f.maxDurationReached:
                routineTimer.addTime(-prac_blank_f.maxDuration)
            elif prac_blank_f.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'prac_f'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "prac_end" ---
        # create an object to store info about Routine prac_end
        prac_end = data.Routine(
            name='prac_end',
            components=[prac_end_center_text, prac_end_bottom_text, prac_end_key_resp],
        )
        prac_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for prac_end_key_resp
        prac_end_key_resp.keys = []
        prac_end_key_resp.rt = []
        _prac_end_key_resp_allKeys = []
        # Run 'Begin Routine' code from prac_end_code
        ###########
        #以下代码是为了按“p”键能够返回练习
        # 清空之前的按键缓存，准备接收新按键
        event.clearEvents()
        
        ##########
        # 如果按了P键重新开始练习，block计数器+1
        prac_cont_block += 1
        
        
        #########
        # 控制文字的左对齐
        prac_end_center_text.alignText = 'left'
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
        # if trial has changed, end Routine now
        if isinstance(continue_prac, data.TrialHandler2) and thisContinue_prac.thisN != continue_prac.thisTrial.thisN:
            continueRoutine = False
        prac_end.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_end_center_text* updates
            
            # if prac_end_center_text is starting this frame...
            if prac_end_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_end_center_text.frameNStart = frameN  # exact frame index
                prac_end_center_text.tStart = t  # local t and not account for scr refresh
                prac_end_center_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_end_center_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_end_center_text.started')
                # update status
                prac_end_center_text.status = STARTED
                prac_end_center_text.setAutoDraw(True)
            
            # if prac_end_center_text is active this frame...
            if prac_end_center_text.status == STARTED:
                # update params
                pass
            
            # *prac_end_bottom_text* updates
            
            # if prac_end_bottom_text is starting this frame...
            if prac_end_bottom_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_end_bottom_text.frameNStart = frameN  # exact frame index
                prac_end_bottom_text.tStart = t  # local t and not account for scr refresh
                prac_end_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_end_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_end_bottom_text.started')
                # update status
                prac_end_bottom_text.status = STARTED
                prac_end_bottom_text.setAutoDraw(True)
            
            # if prac_end_bottom_text is active this frame...
            if prac_end_bottom_text.status == STARTED:
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
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_end_key_resp.started')
                # update status
                prac_end_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_end_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_end_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_end_key_resp.getKeys(keyList=['space','p'], ignoreKeys=["escape"], waitRelease=False)
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
        continue_prac.addData('prac_end_key_resp.keys',prac_end_key_resp.keys)
        if prac_end_key_resp.keys != None:  # we had a response
            continue_prac.addData('prac_end_key_resp.rt', prac_end_key_resp.rt)
            continue_prac.addData('prac_end_key_resp.duration', prac_end_key_resp.duration)
        # Run 'End Routine' code from prac_end_code
        # 等待任意按键
        keys = event.waitKeys()  
        
        if 'p' in keys:
            # 继续练习，什么都不做
            pass
        else:
            # 结束练习
            continue_prac.finished = True
        
        # the Routine "prac_end" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 999.0 repeats of 'continue_prac'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    form_training_outer_block = data.TrialHandler2(
        name='form_training_outer_block',
        nReps=3.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(form_training_outer_block)  # add the loop to the experiment
    thisForm_training_outer_block = form_training_outer_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisForm_training_outer_block.rgb)
    if thisForm_training_outer_block != None:
        for paramName in thisForm_training_outer_block:
            globals()[paramName] = thisForm_training_outer_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisForm_training_outer_block in form_training_outer_block:
        currentLoop = form_training_outer_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisForm_training_outer_block.rgb)
        if thisForm_training_outer_block != None:
            for paramName in thisForm_training_outer_block:
                globals()[paramName] = thisForm_training_outer_block[paramName]
        
        # set up handler to look after randomisation of conditions etc
        form_training_inner_block = data.TrialHandler2(
            name='form_training_inner_block',
            nReps=4.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(form_training_inner_block)  # add the loop to the experiment
        thisForm_training_inner_block = form_training_inner_block.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisForm_training_inner_block.rgb)
        if thisForm_training_inner_block != None:
            for paramName in thisForm_training_inner_block:
                globals()[paramName] = thisForm_training_inner_block[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisForm_training_inner_block in form_training_inner_block:
            currentLoop = form_training_inner_block
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisForm_training_inner_block.rgb)
            if thisForm_training_inner_block != None:
                for paramName in thisForm_training_inner_block:
                    globals()[paramName] = thisForm_training_inner_block[paramName]
            
            # set up handler to look after randomisation of conditions etc
            form_m = data.TrialHandler2(
                name='form_m',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('formal_training_m_faces.xlsx'), 
                seed=None, 
            )
            thisExp.addLoop(form_m)  # add the loop to the experiment
            thisForm_m = form_m.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisForm_m.rgb)
            if thisForm_m != None:
                for paramName in thisForm_m:
                    globals()[paramName] = thisForm_m[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisForm_m in form_m:
                currentLoop = form_m
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisForm_m.rgb)
                if thisForm_m != None:
                    for paramName in thisForm_m:
                        globals()[paramName] = thisForm_m[paramName]
                
                # --- Prepare to start Routine "form_begin_m" ---
                # create an object to store info about Routine form_begin_m
                form_begin_m = data.Routine(
                    name='form_begin_m',
                    components=[form_begin_m_text, form_begin_m_key_resp],
                )
                form_begin_m.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # create starting attributes for form_begin_m_key_resp
                form_begin_m_key_resp.keys = []
                form_begin_m_key_resp.rt = []
                _form_begin_m_key_resp_allKeys = []
                # store start times for form_begin_m
                form_begin_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                form_begin_m.tStart = globalClock.getTime(format='float')
                form_begin_m.status = STARTED
                thisExp.addData('form_begin_m.started', form_begin_m.tStart)
                form_begin_m.maxDuration = None
                # keep track of which components have finished
                form_begin_mComponents = form_begin_m.components
                for thisComponent in form_begin_m.components:
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
                
                # --- Run Routine "form_begin_m" ---
                # if trial has changed, end Routine now
                if isinstance(form_m, data.TrialHandler2) and thisForm_m.thisN != form_m.thisTrial.thisN:
                    continueRoutine = False
                form_begin_m.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *form_begin_m_text* updates
                    
                    # if form_begin_m_text is starting this frame...
                    if form_begin_m_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        form_begin_m_text.frameNStart = frameN  # exact frame index
                        form_begin_m_text.tStart = t  # local t and not account for scr refresh
                        form_begin_m_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_begin_m_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_begin_m_text.started')
                        # update status
                        form_begin_m_text.status = STARTED
                        form_begin_m_text.setAutoDraw(True)
                    
                    # if form_begin_m_text is active this frame...
                    if form_begin_m_text.status == STARTED:
                        # update params
                        pass
                    
                    # *form_begin_m_key_resp* updates
                    waitOnFlip = False
                    
                    # if form_begin_m_key_resp is starting this frame...
                    if form_begin_m_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        form_begin_m_key_resp.frameNStart = frameN  # exact frame index
                        form_begin_m_key_resp.tStart = t  # local t and not account for scr refresh
                        form_begin_m_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_begin_m_key_resp, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_begin_m_key_resp.started')
                        # update status
                        form_begin_m_key_resp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(form_begin_m_key_resp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(form_begin_m_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if form_begin_m_key_resp.status == STARTED and not waitOnFlip:
                        theseKeys = form_begin_m_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                        _form_begin_m_key_resp_allKeys.extend(theseKeys)
                        if len(_form_begin_m_key_resp_allKeys):
                            form_begin_m_key_resp.keys = _form_begin_m_key_resp_allKeys[-1].name  # just the last key pressed
                            form_begin_m_key_resp.rt = _form_begin_m_key_resp_allKeys[-1].rt
                            form_begin_m_key_resp.duration = _form_begin_m_key_resp_allKeys[-1].duration
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
                        form_begin_m.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in form_begin_m.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "form_begin_m" ---
                for thisComponent in form_begin_m.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for form_begin_m
                form_begin_m.tStop = globalClock.getTime(format='float')
                form_begin_m.tStopRefresh = tThisFlipGlobal
                thisExp.addData('form_begin_m.stopped', form_begin_m.tStop)
                # check responses
                if form_begin_m_key_resp.keys in ['', [], None]:  # No response was made
                    form_begin_m_key_resp.keys = None
                form_m.addData('form_begin_m_key_resp.keys',form_begin_m_key_resp.keys)
                if form_begin_m_key_resp.keys != None:  # we had a response
                    form_m.addData('form_begin_m_key_resp.rt', form_begin_m_key_resp.rt)
                    form_m.addData('form_begin_m_key_resp.duration', form_begin_m_key_resp.duration)
                # the Routine "form_begin_m" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "form_cross_m" ---
                # create an object to store info about Routine form_cross_m
                form_cross_m = data.Routine(
                    name='form_cross_m',
                    components=[form_cross_m_text],
                )
                form_cross_m.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # store start times for form_cross_m
                form_cross_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                form_cross_m.tStart = globalClock.getTime(format='float')
                form_cross_m.status = STARTED
                thisExp.addData('form_cross_m.started', form_cross_m.tStart)
                form_cross_m.maxDuration = None
                # keep track of which components have finished
                form_cross_mComponents = form_cross_m.components
                for thisComponent in form_cross_m.components:
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
                
                # --- Run Routine "form_cross_m" ---
                # if trial has changed, end Routine now
                if isinstance(form_m, data.TrialHandler2) and thisForm_m.thisN != form_m.thisTrial.thisN:
                    continueRoutine = False
                form_cross_m.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.5:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *form_cross_m_text* updates
                    
                    # if form_cross_m_text is starting this frame...
                    if form_cross_m_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        form_cross_m_text.frameNStart = frameN  # exact frame index
                        form_cross_m_text.tStart = t  # local t and not account for scr refresh
                        form_cross_m_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_cross_m_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_cross_m_text.started')
                        # update status
                        form_cross_m_text.status = STARTED
                        form_cross_m_text.setAutoDraw(True)
                    
                    # if form_cross_m_text is active this frame...
                    if form_cross_m_text.status == STARTED:
                        # update params
                        pass
                    
                    # if form_cross_m_text is stopping this frame...
                    if form_cross_m_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > form_cross_m_text.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            form_cross_m_text.tStop = t  # not accounting for scr refresh
                            form_cross_m_text.tStopRefresh = tThisFlipGlobal  # on global time
                            form_cross_m_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'form_cross_m_text.stopped')
                            # update status
                            form_cross_m_text.status = FINISHED
                            form_cross_m_text.setAutoDraw(False)
                    
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
                        form_cross_m.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in form_cross_m.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "form_cross_m" ---
                for thisComponent in form_cross_m.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for form_cross_m
                form_cross_m.tStop = globalClock.getTime(format='float')
                form_cross_m.tStopRefresh = tThisFlipGlobal
                thisExp.addData('form_cross_m.stopped', form_cross_m.tStop)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if form_cross_m.maxDurationReached:
                    routineTimer.addTime(-form_cross_m.maxDuration)
                elif form_cross_m.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.500000)
                
                # --- Prepare to start Routine "form_resp_m" ---
                # create an object to store info about Routine form_resp_m
                form_resp_m = data.Routine(
                    name='form_resp_m',
                    components=[form_resp_m_face_image, form_resp_m_left_text, form_resp_m_right_text, form_resp_m_slider, form_resp_m_mouse],
                )
                form_resp_m.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                form_resp_m_face_image.setImage(formal_training_face)
                form_resp_m_slider.reset()
                # setup some python lists for storing info about the form_resp_m_mouse
                form_resp_m_mouse.x = []
                form_resp_m_mouse.y = []
                form_resp_m_mouse.leftButton = []
                form_resp_m_mouse.midButton = []
                form_resp_m_mouse.rightButton = []
                form_resp_m_mouse.time = []
                gotValidClick = False  # until a click is received
                # Run 'Begin Routine' code from form_resp_m_code
                ##########
                # 以下是组件基本设置的代码：
                # 鼠标设置
                win.mouseVisible = True
                # 初始化鼠标位置在slider中间
                form_resp_m_mouse.setPos([0, -0.25])
                
                # 设置slider的marker外观
                form_resp_m_slider.marker.size = (0.02, 0.08)
                form_resp_m_slider.marker.shape = 'square'
                form_resp_m_slider.marker.fillColor = 'white'
                # 记录slider宽度，用于后续计算
                slider_width = 0.8
                
                ############
                # 以下代码用于保存和记录数据
                
                # 试次计数和性别标记
                form_cont_trial += 1
                form_current_gender = 'male'
                
                # 记录这个试次开始的时间，用于后续计算反应时
                start_time = core.getTime()
                # 初始化反应值
                form_cont_resp = None
                # 初始化反应时间
                form_cont_rt = None
                
                # 更新previous_gender变量，
                # 为下一个试次的mini-block检测做准备
                form_previous_gender = form_current_gender
                
                # store start times for form_resp_m
                form_resp_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                form_resp_m.tStart = globalClock.getTime(format='float')
                form_resp_m.status = STARTED
                thisExp.addData('form_resp_m.started', form_resp_m.tStart)
                form_resp_m.maxDuration = None
                # keep track of which components have finished
                form_resp_mComponents = form_resp_m.components
                for thisComponent in form_resp_m.components:
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
                
                # --- Run Routine "form_resp_m" ---
                # if trial has changed, end Routine now
                if isinstance(form_m, data.TrialHandler2) and thisForm_m.thisN != form_m.thisTrial.thisN:
                    continueRoutine = False
                form_resp_m.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *form_resp_m_face_image* updates
                    
                    # if form_resp_m_face_image is starting this frame...
                    if form_resp_m_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_m_face_image.frameNStart = frameN  # exact frame index
                        form_resp_m_face_image.tStart = t  # local t and not account for scr refresh
                        form_resp_m_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_m_face_image, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_resp_m_face_image.started')
                        # update status
                        form_resp_m_face_image.status = STARTED
                        form_resp_m_face_image.setAutoDraw(True)
                    
                    # if form_resp_m_face_image is active this frame...
                    if form_resp_m_face_image.status == STARTED:
                        # update params
                        pass
                    
                    # *form_resp_m_left_text* updates
                    
                    # if form_resp_m_left_text is starting this frame...
                    if form_resp_m_left_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_m_left_text.frameNStart = frameN  # exact frame index
                        form_resp_m_left_text.tStart = t  # local t and not account for scr refresh
                        form_resp_m_left_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_m_left_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_resp_m_left_text.started')
                        # update status
                        form_resp_m_left_text.status = STARTED
                        form_resp_m_left_text.setAutoDraw(True)
                    
                    # if form_resp_m_left_text is active this frame...
                    if form_resp_m_left_text.status == STARTED:
                        # update params
                        pass
                    
                    # *form_resp_m_right_text* updates
                    
                    # if form_resp_m_right_text is starting this frame...
                    if form_resp_m_right_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_m_right_text.frameNStart = frameN  # exact frame index
                        form_resp_m_right_text.tStart = t  # local t and not account for scr refresh
                        form_resp_m_right_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_m_right_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_resp_m_right_text.started')
                        # update status
                        form_resp_m_right_text.status = STARTED
                        form_resp_m_right_text.setAutoDraw(True)
                    
                    # if form_resp_m_right_text is active this frame...
                    if form_resp_m_right_text.status == STARTED:
                        # update params
                        pass
                    
                    # *form_resp_m_slider* updates
                    
                    # if form_resp_m_slider is starting this frame...
                    if form_resp_m_slider.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_m_slider.frameNStart = frameN  # exact frame index
                        form_resp_m_slider.tStart = t  # local t and not account for scr refresh
                        form_resp_m_slider.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_m_slider, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_resp_m_slider.started')
                        # update status
                        form_resp_m_slider.status = STARTED
                        form_resp_m_slider.setAutoDraw(True)
                    
                    # if form_resp_m_slider is active this frame...
                    if form_resp_m_slider.status == STARTED:
                        # update params
                        pass
                    # *form_resp_m_mouse* updates
                    
                    # if form_resp_m_mouse is starting this frame...
                    if form_resp_m_mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_m_mouse.frameNStart = frameN  # exact frame index
                        form_resp_m_mouse.tStart = t  # local t and not account for scr refresh
                        form_resp_m_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_m_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('form_resp_m_mouse.started', t)
                        # update status
                        form_resp_m_mouse.status = STARTED
                        form_resp_m_mouse.mouseClock.reset()
                        prevButtonState = form_resp_m_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if form_resp_m_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = form_resp_m_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                pass
                    # Run 'Each Frame' code from form_resp_m_code
                    ##########
                    # 获取当前鼠标位置
                    mouse_pos = form_resp_m_mouse.getPos()
                    mouse_x, mouse_y = mouse_pos
                    
                    # 基于slider的位置和大小，计算slider的边界
                    slider_center_x = 0
                    slider_center_y = -0.25
                    slider_left = slider_center_x - slider_width/2
                    slider_right = slider_center_x + slider_width/2
                    slider_bottom = slider_center_y - 0.025
                    slider_top = slider_center_y + 0.025
                    
                    # 检查鼠标是否在slider区域内
                    mouse_in_slider = (slider_left <= mouse_x <= slider_right and
                                       slider_bottom <= mouse_y <= slider_top)
                    
                    # 处理slider交互逻辑
                    if mouse_in_slider:
                        # 计算slider值，修正边界问题
                        slider_value = (mouse_x - slider_left) / slider_width
                    
                        # 关键修改：扩大边界范围，允许0和1
                        if slider_value < 0.01:
                            slider_value = 0.0
                        elif slider_value > 0.99:
                            slider_value = 1.0
                        else:
                            slider_value = max(0, min(1, slider_value))
                    
                        # 更新slider marker位置
                        form_resp_m_slider.markerPos = slider_value
                    
                        # 检测鼠标点击
                        if form_resp_m_mouse.getPressed()[0]:
                            # 保存反应数据
                            form_cont_resp = round(slider_value, 2)
                            form_cont_rt = round(core.getTime() - start_time, 3)
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
                        form_resp_m.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in form_resp_m.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "form_resp_m" ---
                for thisComponent in form_resp_m.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for form_resp_m
                form_resp_m.tStop = globalClock.getTime(format='float')
                form_resp_m.tStopRefresh = tThisFlipGlobal
                thisExp.addData('form_resp_m.stopped', form_resp_m.tStop)
                form_m.addData('form_resp_m_slider.response', form_resp_m_slider.getRating())
                form_m.addData('form_resp_m_slider.rt', form_resp_m_slider.getRT())
                # store data for form_m (TrialHandler)
                form_m.addData('form_resp_m_mouse.x', form_resp_m_mouse.x)
                form_m.addData('form_resp_m_mouse.y', form_resp_m_mouse.y)
                form_m.addData('form_resp_m_mouse.leftButton', form_resp_m_mouse.leftButton)
                form_m.addData('form_resp_m_mouse.midButton', form_resp_m_mouse.midButton)
                form_m.addData('form_resp_m_mouse.rightButton', form_resp_m_mouse.rightButton)
                form_m.addData('form_resp_m_mouse.time', form_resp_m_mouse.time)
                # Run 'End Routine' code from form_resp_m_code
                #########
                # 以下代码是为了保存数据
                
                # block数
                thisExp.addData('form_cont_block', form_cont_block)
                # trial数
                thisExp.addData('form_cont_trial', form_cont_trial)
                # 当前模特性别
                thisExp.addData('form_current_gender', form_current_gender)
                # 被试连续评分值
                thisExp.addData('form_cont_resp', form_cont_resp)
                # 反应时间
                thisExp.addData('form_cont_rt', form_cont_rt)
                # 进入下一条数据记录
                thisExp.nextEntry()
                
                
                # 隐藏鼠标
                win.mouseVisible = False
                
                
                # the Routine "form_resp_m" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "form_blank_m" ---
                # create an object to store info about Routine form_blank_m
                form_blank_m = data.Routine(
                    name='form_blank_m',
                    components=[form_blank_m_text],
                )
                form_blank_m.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # store start times for form_blank_m
                form_blank_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                form_blank_m.tStart = globalClock.getTime(format='float')
                form_blank_m.status = STARTED
                thisExp.addData('form_blank_m.started', form_blank_m.tStart)
                form_blank_m.maxDuration = None
                # keep track of which components have finished
                form_blank_mComponents = form_blank_m.components
                for thisComponent in form_blank_m.components:
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
                
                # --- Run Routine "form_blank_m" ---
                # if trial has changed, end Routine now
                if isinstance(form_m, data.TrialHandler2) and thisForm_m.thisN != form_m.thisTrial.thisN:
                    continueRoutine = False
                form_blank_m.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.5:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *form_blank_m_text* updates
                    
                    # if form_blank_m_text is starting this frame...
                    if form_blank_m_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        form_blank_m_text.frameNStart = frameN  # exact frame index
                        form_blank_m_text.tStart = t  # local t and not account for scr refresh
                        form_blank_m_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_blank_m_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_blank_m_text.started')
                        # update status
                        form_blank_m_text.status = STARTED
                        form_blank_m_text.setAutoDraw(True)
                    
                    # if form_blank_m_text is active this frame...
                    if form_blank_m_text.status == STARTED:
                        # update params
                        pass
                    
                    # if form_blank_m_text is stopping this frame...
                    if form_blank_m_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > form_blank_m_text.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            form_blank_m_text.tStop = t  # not accounting for scr refresh
                            form_blank_m_text.tStopRefresh = tThisFlipGlobal  # on global time
                            form_blank_m_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'form_blank_m_text.stopped')
                            # update status
                            form_blank_m_text.status = FINISHED
                            form_blank_m_text.setAutoDraw(False)
                    
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
                        form_blank_m.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in form_blank_m.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "form_blank_m" ---
                for thisComponent in form_blank_m.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for form_blank_m
                form_blank_m.tStop = globalClock.getTime(format='float')
                form_blank_m.tStopRefresh = tThisFlipGlobal
                thisExp.addData('form_blank_m.stopped', form_blank_m.tStop)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if form_blank_m.maxDurationReached:
                    routineTimer.addTime(-form_blank_m.maxDuration)
                elif form_blank_m.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.500000)
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'form_m'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "form_break_m" ---
            # create an object to store info about Routine form_break_m
            form_break_m = data.Routine(
                name='form_break_m',
                components=[form_break_m_text],
            )
            form_break_m.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for form_break_m
            form_break_m.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            form_break_m.tStart = globalClock.getTime(format='float')
            form_break_m.status = STARTED
            thisExp.addData('form_break_m.started', form_break_m.tStart)
            form_break_m.maxDuration = None
            # keep track of which components have finished
            form_break_mComponents = form_break_m.components
            for thisComponent in form_break_m.components:
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
            
            # --- Run Routine "form_break_m" ---
            # if trial has changed, end Routine now
            if isinstance(form_training_inner_block, data.TrialHandler2) and thisForm_training_inner_block.thisN != form_training_inner_block.thisTrial.thisN:
                continueRoutine = False
            form_break_m.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 10.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *form_break_m_text* updates
                
                # if form_break_m_text is starting this frame...
                if form_break_m_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    form_break_m_text.frameNStart = frameN  # exact frame index
                    form_break_m_text.tStart = t  # local t and not account for scr refresh
                    form_break_m_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(form_break_m_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_break_m_text.started')
                    # update status
                    form_break_m_text.status = STARTED
                    form_break_m_text.setAutoDraw(True)
                
                # if form_break_m_text is active this frame...
                if form_break_m_text.status == STARTED:
                    # update params
                    pass
                
                # if form_break_m_text is stopping this frame...
                if form_break_m_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > form_break_m_text.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        form_break_m_text.tStop = t  # not accounting for scr refresh
                        form_break_m_text.tStopRefresh = tThisFlipGlobal  # on global time
                        form_break_m_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_break_m_text.stopped')
                        # update status
                        form_break_m_text.status = FINISHED
                        form_break_m_text.setAutoDraw(False)
                
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
                    form_break_m.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in form_break_m.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "form_break_m" ---
            for thisComponent in form_break_m.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for form_break_m
            form_break_m.tStop = globalClock.getTime(format='float')
            form_break_m.tStopRefresh = tThisFlipGlobal
            thisExp.addData('form_break_m.stopped', form_break_m.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if form_break_m.maxDurationReached:
                routineTimer.addTime(-form_break_m.maxDuration)
            elif form_break_m.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-10.000000)
            
            # set up handler to look after randomisation of conditions etc
            form_f = data.TrialHandler2(
                name='form_f',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('formal_training_f_faces.xlsx'), 
                seed=None, 
            )
            thisExp.addLoop(form_f)  # add the loop to the experiment
            thisForm_f = form_f.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisForm_f.rgb)
            if thisForm_f != None:
                for paramName in thisForm_f:
                    globals()[paramName] = thisForm_f[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisForm_f in form_f:
                currentLoop = form_f
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisForm_f.rgb)
                if thisForm_f != None:
                    for paramName in thisForm_f:
                        globals()[paramName] = thisForm_f[paramName]
                
                # --- Prepare to start Routine "form_begin_f" ---
                # create an object to store info about Routine form_begin_f
                form_begin_f = data.Routine(
                    name='form_begin_f',
                    components=[form_begin_f_text, form_begin_f_key_resp],
                )
                form_begin_f.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # create starting attributes for form_begin_f_key_resp
                form_begin_f_key_resp.keys = []
                form_begin_f_key_resp.rt = []
                _form_begin_f_key_resp_allKeys = []
                # store start times for form_begin_f
                form_begin_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                form_begin_f.tStart = globalClock.getTime(format='float')
                form_begin_f.status = STARTED
                thisExp.addData('form_begin_f.started', form_begin_f.tStart)
                form_begin_f.maxDuration = None
                # keep track of which components have finished
                form_begin_fComponents = form_begin_f.components
                for thisComponent in form_begin_f.components:
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
                
                # --- Run Routine "form_begin_f" ---
                # if trial has changed, end Routine now
                if isinstance(form_f, data.TrialHandler2) and thisForm_f.thisN != form_f.thisTrial.thisN:
                    continueRoutine = False
                form_begin_f.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *form_begin_f_text* updates
                    
                    # if form_begin_f_text is starting this frame...
                    if form_begin_f_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        form_begin_f_text.frameNStart = frameN  # exact frame index
                        form_begin_f_text.tStart = t  # local t and not account for scr refresh
                        form_begin_f_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_begin_f_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_begin_f_text.started')
                        # update status
                        form_begin_f_text.status = STARTED
                        form_begin_f_text.setAutoDraw(True)
                    
                    # if form_begin_f_text is active this frame...
                    if form_begin_f_text.status == STARTED:
                        # update params
                        pass
                    
                    # *form_begin_f_key_resp* updates
                    waitOnFlip = False
                    
                    # if form_begin_f_key_resp is starting this frame...
                    if form_begin_f_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        form_begin_f_key_resp.frameNStart = frameN  # exact frame index
                        form_begin_f_key_resp.tStart = t  # local t and not account for scr refresh
                        form_begin_f_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_begin_f_key_resp, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_begin_f_key_resp.started')
                        # update status
                        form_begin_f_key_resp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(form_begin_f_key_resp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(form_begin_f_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if form_begin_f_key_resp.status == STARTED and not waitOnFlip:
                        theseKeys = form_begin_f_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                        _form_begin_f_key_resp_allKeys.extend(theseKeys)
                        if len(_form_begin_f_key_resp_allKeys):
                            form_begin_f_key_resp.keys = _form_begin_f_key_resp_allKeys[-1].name  # just the last key pressed
                            form_begin_f_key_resp.rt = _form_begin_f_key_resp_allKeys[-1].rt
                            form_begin_f_key_resp.duration = _form_begin_f_key_resp_allKeys[-1].duration
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
                        form_begin_f.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in form_begin_f.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "form_begin_f" ---
                for thisComponent in form_begin_f.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for form_begin_f
                form_begin_f.tStop = globalClock.getTime(format='float')
                form_begin_f.tStopRefresh = tThisFlipGlobal
                thisExp.addData('form_begin_f.stopped', form_begin_f.tStop)
                # check responses
                if form_begin_f_key_resp.keys in ['', [], None]:  # No response was made
                    form_begin_f_key_resp.keys = None
                form_f.addData('form_begin_f_key_resp.keys',form_begin_f_key_resp.keys)
                if form_begin_f_key_resp.keys != None:  # we had a response
                    form_f.addData('form_begin_f_key_resp.rt', form_begin_f_key_resp.rt)
                    form_f.addData('form_begin_f_key_resp.duration', form_begin_f_key_resp.duration)
                # the Routine "form_begin_f" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "form_cross_f" ---
                # create an object to store info about Routine form_cross_f
                form_cross_f = data.Routine(
                    name='form_cross_f',
                    components=[form_cross_f_text],
                )
                form_cross_f.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # store start times for form_cross_f
                form_cross_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                form_cross_f.tStart = globalClock.getTime(format='float')
                form_cross_f.status = STARTED
                thisExp.addData('form_cross_f.started', form_cross_f.tStart)
                form_cross_f.maxDuration = None
                # keep track of which components have finished
                form_cross_fComponents = form_cross_f.components
                for thisComponent in form_cross_f.components:
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
                
                # --- Run Routine "form_cross_f" ---
                # if trial has changed, end Routine now
                if isinstance(form_f, data.TrialHandler2) and thisForm_f.thisN != form_f.thisTrial.thisN:
                    continueRoutine = False
                form_cross_f.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.5:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *form_cross_f_text* updates
                    
                    # if form_cross_f_text is starting this frame...
                    if form_cross_f_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        form_cross_f_text.frameNStart = frameN  # exact frame index
                        form_cross_f_text.tStart = t  # local t and not account for scr refresh
                        form_cross_f_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_cross_f_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_cross_f_text.started')
                        # update status
                        form_cross_f_text.status = STARTED
                        form_cross_f_text.setAutoDraw(True)
                    
                    # if form_cross_f_text is active this frame...
                    if form_cross_f_text.status == STARTED:
                        # update params
                        pass
                    
                    # if form_cross_f_text is stopping this frame...
                    if form_cross_f_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > form_cross_f_text.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            form_cross_f_text.tStop = t  # not accounting for scr refresh
                            form_cross_f_text.tStopRefresh = tThisFlipGlobal  # on global time
                            form_cross_f_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'form_cross_f_text.stopped')
                            # update status
                            form_cross_f_text.status = FINISHED
                            form_cross_f_text.setAutoDraw(False)
                    
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
                        form_cross_f.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in form_cross_f.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "form_cross_f" ---
                for thisComponent in form_cross_f.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for form_cross_f
                form_cross_f.tStop = globalClock.getTime(format='float')
                form_cross_f.tStopRefresh = tThisFlipGlobal
                thisExp.addData('form_cross_f.stopped', form_cross_f.tStop)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if form_cross_f.maxDurationReached:
                    routineTimer.addTime(-form_cross_f.maxDuration)
                elif form_cross_f.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.500000)
                
                # --- Prepare to start Routine "form_resp_f" ---
                # create an object to store info about Routine form_resp_f
                form_resp_f = data.Routine(
                    name='form_resp_f',
                    components=[form_resp_f_face_image, form_resp_f_left_text, form_resp_f_right_text, form_resp_f_slider, form_resp_f_mouse],
                )
                form_resp_f.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                form_resp_f_face_image.setImage(formal_training_face)
                form_resp_f_slider.reset()
                # setup some python lists for storing info about the form_resp_f_mouse
                form_resp_f_mouse.x = []
                form_resp_f_mouse.y = []
                form_resp_f_mouse.leftButton = []
                form_resp_f_mouse.midButton = []
                form_resp_f_mouse.rightButton = []
                form_resp_f_mouse.time = []
                gotValidClick = False  # until a click is received
                # Run 'Begin Routine' code from form_resp_f_code
                ##########
                # 以下是组件基本设置的代码：
                
                # 鼠标设置
                win.mouseVisible = True
                # 初始化鼠标位置在slider中间
                form_resp_f_mouse.setPos([0, -0.25])
                
                # 设置slider的marker外观
                form_resp_f_slider.marker.size = (0.02, 0.08)
                form_resp_f_slider.marker.shape = 'square'
                form_resp_f_slider.marker.fillColor = 'white'
                # 记录slider宽度，用于后续计算
                slider_width = 0.8
                
                ############
                # 以下代码用于保存和记录数据
                
                # 试次计数和性别标记
                form_cont_trial += 1
                form_current_gender = 'female'
                
                # 记录这个试次开始的时间，用于后续计算反应时
                start_time = core.getTime()
                form_cont_resp = None  # 初始化反应值
                form_cont_rt = None    # 初始化反应时间
                
                # 更新previous_gender变量，
                # 为下一个试次的mini-block检测做准备
                form_previous_gender = form_current_gender
                
                # store start times for form_resp_f
                form_resp_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                form_resp_f.tStart = globalClock.getTime(format='float')
                form_resp_f.status = STARTED
                thisExp.addData('form_resp_f.started', form_resp_f.tStart)
                form_resp_f.maxDuration = None
                # keep track of which components have finished
                form_resp_fComponents = form_resp_f.components
                for thisComponent in form_resp_f.components:
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
                
                # --- Run Routine "form_resp_f" ---
                # if trial has changed, end Routine now
                if isinstance(form_f, data.TrialHandler2) and thisForm_f.thisN != form_f.thisTrial.thisN:
                    continueRoutine = False
                form_resp_f.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *form_resp_f_face_image* updates
                    
                    # if form_resp_f_face_image is starting this frame...
                    if form_resp_f_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_f_face_image.frameNStart = frameN  # exact frame index
                        form_resp_f_face_image.tStart = t  # local t and not account for scr refresh
                        form_resp_f_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_f_face_image, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_resp_f_face_image.started')
                        # update status
                        form_resp_f_face_image.status = STARTED
                        form_resp_f_face_image.setAutoDraw(True)
                    
                    # if form_resp_f_face_image is active this frame...
                    if form_resp_f_face_image.status == STARTED:
                        # update params
                        pass
                    
                    # *form_resp_f_left_text* updates
                    
                    # if form_resp_f_left_text is starting this frame...
                    if form_resp_f_left_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_f_left_text.frameNStart = frameN  # exact frame index
                        form_resp_f_left_text.tStart = t  # local t and not account for scr refresh
                        form_resp_f_left_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_f_left_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_resp_f_left_text.started')
                        # update status
                        form_resp_f_left_text.status = STARTED
                        form_resp_f_left_text.setAutoDraw(True)
                    
                    # if form_resp_f_left_text is active this frame...
                    if form_resp_f_left_text.status == STARTED:
                        # update params
                        pass
                    
                    # *form_resp_f_right_text* updates
                    
                    # if form_resp_f_right_text is starting this frame...
                    if form_resp_f_right_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_f_right_text.frameNStart = frameN  # exact frame index
                        form_resp_f_right_text.tStart = t  # local t and not account for scr refresh
                        form_resp_f_right_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_f_right_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_resp_f_right_text.started')
                        # update status
                        form_resp_f_right_text.status = STARTED
                        form_resp_f_right_text.setAutoDraw(True)
                    
                    # if form_resp_f_right_text is active this frame...
                    if form_resp_f_right_text.status == STARTED:
                        # update params
                        pass
                    
                    # *form_resp_f_slider* updates
                    
                    # if form_resp_f_slider is starting this frame...
                    if form_resp_f_slider.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_f_slider.frameNStart = frameN  # exact frame index
                        form_resp_f_slider.tStart = t  # local t and not account for scr refresh
                        form_resp_f_slider.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_f_slider, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_resp_f_slider.started')
                        # update status
                        form_resp_f_slider.status = STARTED
                        form_resp_f_slider.setAutoDraw(True)
                    
                    # if form_resp_f_slider is active this frame...
                    if form_resp_f_slider.status == STARTED:
                        # update params
                        pass
                    # *form_resp_f_mouse* updates
                    
                    # if form_resp_f_mouse is starting this frame...
                    if form_resp_f_mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        form_resp_f_mouse.frameNStart = frameN  # exact frame index
                        form_resp_f_mouse.tStart = t  # local t and not account for scr refresh
                        form_resp_f_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_resp_f_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('form_resp_f_mouse.started', t)
                        # update status
                        form_resp_f_mouse.status = STARTED
                        form_resp_f_mouse.mouseClock.reset()
                        prevButtonState = form_resp_f_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if form_resp_f_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = form_resp_f_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                pass
                    # Run 'Each Frame' code from form_resp_f_code
                    ##########
                    # 获取当前鼠标位置
                    mouse_pos = form_resp_f_mouse.getPos()
                    mouse_x, mouse_y = mouse_pos
                    
                    # 基于slider的位置和大小，计算slider的边界
                    slider_center_x = 0
                    slider_center_y = -0.25
                    slider_left = slider_center_x - slider_width/2
                    slider_right = slider_center_x + slider_width/2
                    slider_bottom = slider_center_y - 0.025
                    slider_top = slider_center_y + 0.025
                    
                    # 检查鼠标是否在slider区域内
                    mouse_in_slider = (slider_left <= mouse_x <= slider_right and
                                       slider_bottom <= mouse_y <= slider_top)
                    
                    # 处理slider交互逻辑
                    if mouse_in_slider:
                        # 计算slider值，修正边界问题
                        slider_value = (mouse_x - slider_left) / slider_width
                    
                        # 关键修改：扩大边界范围，允许0和1
                        if slider_value < 0.01:
                            slider_value = 0.0
                        elif slider_value > 0.99:
                            slider_value = 1.0
                        else:
                            slider_value = max(0, min(1, slider_value))
                    
                        # 更新slider marker位置
                        form_resp_f_slider.markerPos = slider_value
                    
                        # 检测鼠标点击
                        if form_resp_f_mouse.getPressed()[0]:
                            # 保存反应数据
                            form_cont_resp = round(slider_value, 2)
                            form_cont_rt = round(core.getTime() - start_time, 3)
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
                        form_resp_f.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in form_resp_f.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "form_resp_f" ---
                for thisComponent in form_resp_f.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for form_resp_f
                form_resp_f.tStop = globalClock.getTime(format='float')
                form_resp_f.tStopRefresh = tThisFlipGlobal
                thisExp.addData('form_resp_f.stopped', form_resp_f.tStop)
                form_f.addData('form_resp_f_slider.response', form_resp_f_slider.getRating())
                form_f.addData('form_resp_f_slider.rt', form_resp_f_slider.getRT())
                # store data for form_f (TrialHandler)
                form_f.addData('form_resp_f_mouse.x', form_resp_f_mouse.x)
                form_f.addData('form_resp_f_mouse.y', form_resp_f_mouse.y)
                form_f.addData('form_resp_f_mouse.leftButton', form_resp_f_mouse.leftButton)
                form_f.addData('form_resp_f_mouse.midButton', form_resp_f_mouse.midButton)
                form_f.addData('form_resp_f_mouse.rightButton', form_resp_f_mouse.rightButton)
                form_f.addData('form_resp_f_mouse.time', form_resp_f_mouse.time)
                # Run 'End Routine' code from form_resp_f_code
                #########
                # 以下代码是为了保存数据
                
                # block数
                thisExp.addData('form_cont_block', form_cont_block)
                # trial数
                thisExp.addData('form_cont_trial', form_cont_trial)
                # 当前模特性别
                thisExp.addData('form_current_gender', form_current_gender)
                # 被试连续评分值
                thisExp.addData('form_cont_resp', form_cont_resp)
                # 反应时间
                thisExp.addData('form_cont_rt', form_cont_rt)
                # 进入下一条数据记录
                thisExp.nextEntry()
                
                # 隐藏鼠标
                win.mouseVisible = False
                
                # the Routine "form_resp_f" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "form_blank_f" ---
                # create an object to store info about Routine form_blank_f
                form_blank_f = data.Routine(
                    name='form_blank_f',
                    components=[form_blank_f_text],
                )
                form_blank_f.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # store start times for form_blank_f
                form_blank_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                form_blank_f.tStart = globalClock.getTime(format='float')
                form_blank_f.status = STARTED
                thisExp.addData('form_blank_f.started', form_blank_f.tStart)
                form_blank_f.maxDuration = None
                # keep track of which components have finished
                form_blank_fComponents = form_blank_f.components
                for thisComponent in form_blank_f.components:
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
                
                # --- Run Routine "form_blank_f" ---
                # if trial has changed, end Routine now
                if isinstance(form_f, data.TrialHandler2) and thisForm_f.thisN != form_f.thisTrial.thisN:
                    continueRoutine = False
                form_blank_f.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.5:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *form_blank_f_text* updates
                    
                    # if form_blank_f_text is starting this frame...
                    if form_blank_f_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        form_blank_f_text.frameNStart = frameN  # exact frame index
                        form_blank_f_text.tStart = t  # local t and not account for scr refresh
                        form_blank_f_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(form_blank_f_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_blank_f_text.started')
                        # update status
                        form_blank_f_text.status = STARTED
                        form_blank_f_text.setAutoDraw(True)
                    
                    # if form_blank_f_text is active this frame...
                    if form_blank_f_text.status == STARTED:
                        # update params
                        pass
                    
                    # if form_blank_f_text is stopping this frame...
                    if form_blank_f_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > form_blank_f_text.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            form_blank_f_text.tStop = t  # not accounting for scr refresh
                            form_blank_f_text.tStopRefresh = tThisFlipGlobal  # on global time
                            form_blank_f_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'form_blank_f_text.stopped')
                            # update status
                            form_blank_f_text.status = FINISHED
                            form_blank_f_text.setAutoDraw(False)
                    
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
                        form_blank_f.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in form_blank_f.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "form_blank_f" ---
                for thisComponent in form_blank_f.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for form_blank_f
                form_blank_f.tStop = globalClock.getTime(format='float')
                form_blank_f.tStopRefresh = tThisFlipGlobal
                thisExp.addData('form_blank_f.stopped', form_blank_f.tStop)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if form_blank_f.maxDurationReached:
                    routineTimer.addTime(-form_blank_f.maxDuration)
                elif form_blank_f.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.500000)
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'form_f'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "form_break_f" ---
            # create an object to store info about Routine form_break_f
            form_break_f = data.Routine(
                name='form_break_f',
                components=[form_break_f_text],
            )
            form_break_f.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for form_break_f
            form_break_f.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            form_break_f.tStart = globalClock.getTime(format='float')
            form_break_f.status = STARTED
            thisExp.addData('form_break_f.started', form_break_f.tStart)
            form_break_f.maxDuration = None
            # keep track of which components have finished
            form_break_fComponents = form_break_f.components
            for thisComponent in form_break_f.components:
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
            
            # --- Run Routine "form_break_f" ---
            # if trial has changed, end Routine now
            if isinstance(form_training_inner_block, data.TrialHandler2) and thisForm_training_inner_block.thisN != form_training_inner_block.thisTrial.thisN:
                continueRoutine = False
            form_break_f.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 10.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *form_break_f_text* updates
                
                # if form_break_f_text is starting this frame...
                if form_break_f_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    form_break_f_text.frameNStart = frameN  # exact frame index
                    form_break_f_text.tStart = t  # local t and not account for scr refresh
                    form_break_f_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(form_break_f_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_break_f_text.started')
                    # update status
                    form_break_f_text.status = STARTED
                    form_break_f_text.setAutoDraw(True)
                
                # if form_break_f_text is active this frame...
                if form_break_f_text.status == STARTED:
                    # update params
                    pass
                
                # if form_break_f_text is stopping this frame...
                if form_break_f_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > form_break_f_text.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        form_break_f_text.tStop = t  # not accounting for scr refresh
                        form_break_f_text.tStopRefresh = tThisFlipGlobal  # on global time
                        form_break_f_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'form_break_f_text.stopped')
                        # update status
                        form_break_f_text.status = FINISHED
                        form_break_f_text.setAutoDraw(False)
                
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
                    form_break_f.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in form_break_f.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "form_break_f" ---
            for thisComponent in form_break_f.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for form_break_f
            form_break_f.tStop = globalClock.getTime(format='float')
            form_break_f.tStopRefresh = tThisFlipGlobal
            thisExp.addData('form_break_f.stopped', form_break_f.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if form_break_f.maxDurationReached:
                routineTimer.addTime(-form_break_f.maxDuration)
            elif form_break_f.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-10.000000)
            thisExp.nextEntry()
            
        # completed 4.0 repeats of 'form_training_inner_block'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "form_break" ---
        # create an object to store info about Routine form_break
        form_break = data.Routine(
            name='form_break',
            components=[form_break_center_text, form_break_bottom_text, form_break_key_resp],
        )
        form_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for form_break_key_resp
        form_break_key_resp.keys = []
        form_break_key_resp.rt = []
        _form_break_key_resp_allKeys = []
        # Run 'Begin Routine' code from form_break_code
        ##########
        # 每次休息，block计数器＋1
        form_cont_block += 1
        
        #########
        # 显示实验进度
        progress_text = visual.TextStim(
            win=win,
            text=f'已完成{form_cont_block - 1}/3部分',  # 直接在这里计算
            font='Arial',
            pos=[0, 0.1],  # 纵坐标0.1
            height=0.03,
            color='black'
        )
        
        # 记录文本开始显示的时间
        text_start_time = core.getTime()  
        # store start times for form_break
        form_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        form_break.tStart = globalClock.getTime(format='float')
        form_break.status = STARTED
        thisExp.addData('form_break.started', form_break.tStart)
        form_break.maxDuration = None
        # keep track of which components have finished
        form_breakComponents = form_break.components
        for thisComponent in form_break.components:
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
        
        # --- Run Routine "form_break" ---
        # if trial has changed, end Routine now
        if isinstance(form_training_outer_block, data.TrialHandler2) and thisForm_training_outer_block.thisN != form_training_outer_block.thisTrial.thisN:
            continueRoutine = False
        form_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *form_break_center_text* updates
            
            # if form_break_center_text is starting this frame...
            if form_break_center_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                form_break_center_text.frameNStart = frameN  # exact frame index
                form_break_center_text.tStart = t  # local t and not account for scr refresh
                form_break_center_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(form_break_center_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'form_break_center_text.started')
                # update status
                form_break_center_text.status = STARTED
                form_break_center_text.setAutoDraw(True)
            
            # if form_break_center_text is active this frame...
            if form_break_center_text.status == STARTED:
                # update params
                pass
            
            # if form_break_center_text is stopping this frame...
            if form_break_center_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > form_break_center_text.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    form_break_center_text.tStop = t  # not accounting for scr refresh
                    form_break_center_text.tStopRefresh = tThisFlipGlobal  # on global time
                    form_break_center_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'form_break_center_text.stopped')
                    # update status
                    form_break_center_text.status = FINISHED
                    form_break_center_text.setAutoDraw(False)
            
            # *form_break_bottom_text* updates
            
            # if form_break_bottom_text is starting this frame...
            if form_break_bottom_text.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                # keep track of start time/frame for later
                form_break_bottom_text.frameNStart = frameN  # exact frame index
                form_break_bottom_text.tStart = t  # local t and not account for scr refresh
                form_break_bottom_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(form_break_bottom_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'form_break_bottom_text.started')
                # update status
                form_break_bottom_text.status = STARTED
                form_break_bottom_text.setAutoDraw(True)
            
            # if form_break_bottom_text is active this frame...
            if form_break_bottom_text.status == STARTED:
                # update params
                pass
            
            # *form_break_key_resp* updates
            waitOnFlip = False
            
            # if form_break_key_resp is starting this frame...
            if form_break_key_resp.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                # keep track of start time/frame for later
                form_break_key_resp.frameNStart = frameN  # exact frame index
                form_break_key_resp.tStart = t  # local t and not account for scr refresh
                form_break_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(form_break_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'form_break_key_resp.started')
                # update status
                form_break_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(form_break_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(form_break_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if form_break_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = form_break_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _form_break_key_resp_allKeys.extend(theseKeys)
                if len(_form_break_key_resp_allKeys):
                    form_break_key_resp.keys = _form_break_key_resp_allKeys[-1].name  # just the last key pressed
                    form_break_key_resp.rt = _form_break_key_resp_allKeys[-1].rt
                    form_break_key_resp.duration = _form_break_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from form_break_code
            # 控制文本只显示60秒
            if core.getTime() - text_start_time <= 6.0:  
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
                form_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in form_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "form_break" ---
        for thisComponent in form_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for form_break
        form_break.tStop = globalClock.getTime(format='float')
        form_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('form_break.stopped', form_break.tStop)
        # check responses
        if form_break_key_resp.keys in ['', [], None]:  # No response was made
            form_break_key_resp.keys = None
        form_training_outer_block.addData('form_break_key_resp.keys',form_break_key_resp.keys)
        if form_break_key_resp.keys != None:  # we had a response
            form_training_outer_block.addData('form_break_key_resp.rt', form_break_key_resp.rt)
            form_training_outer_block.addData('form_break_key_resp.duration', form_break_key_resp.duration)
        # the Routine "form_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 3.0 repeats of 'form_training_outer_block'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
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
                components=[pracl_xab_blank1_text],
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
                
                # *pracl_xab_blank1_text* updates
                
                # if pracl_xab_blank1_text is starting this frame...
                if pracl_xab_blank1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pracl_xab_blank1_text.frameNStart = frameN  # exact frame index
                    pracl_xab_blank1_text.tStart = t  # local t and not account for scr refresh
                    pracl_xab_blank1_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracl_xab_blank1_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracl_xab_blank1_text.started')
                    # update status
                    pracl_xab_blank1_text.status = STARTED
                    pracl_xab_blank1_text.setAutoDraw(True)
                
                # if pracl_xab_blank1_text is active this frame...
                if pracl_xab_blank1_text.status == STARTED:
                    # update params
                    pass
                
                # if pracl_xab_blank1_text is stopping this frame...
                if pracl_xab_blank1_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracl_xab_blank1_text.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        pracl_xab_blank1_text.tStop = t  # not accounting for scr refresh
                        pracl_xab_blank1_text.tStopRefresh = tThisFlipGlobal  # on global time
                        pracl_xab_blank1_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracl_xab_blank1_text.stopped')
                        # update status
                        pracl_xab_blank1_text.status = FINISHED
                        pracl_xab_blank1_text.setAutoDraw(False)
                
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
            prac_user_key = None       # 被试按下的键
            
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
