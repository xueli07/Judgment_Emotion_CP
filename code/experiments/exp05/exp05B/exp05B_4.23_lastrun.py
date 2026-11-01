#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on 四月 24, 2024, at 15:52
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
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
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'exp05B'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'sex': 'female/male',
    'age': '18~30',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
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
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
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
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1280, 720], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
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
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
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
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
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
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
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
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from intro_code
    win.mouseVisible = False
    
    # --- Initialize components for Routine "with_intro_2" ---
    with_intro_2_text = visual.TextStim(win=win, name='with_intro_2_text',
        text='1.分类判断：\n\n如果您看到‘恐惧’和‘平静’两个文字（如下所示），\n这表示您需要对接下来呈现的面孔图片进行分类情绪判断，\n即从"恐惧"或"平静"两个选项中任选其一。',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_2_key_resp = keyboard.Keyboard()
    with_intro_2_left_calm = visual.TextStim(win=win, name='with_intro_2_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    with_intro_2_right_fear = visual.TextStim(win=win, name='with_intro_2_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "with_intro_3" ---
    with_intro_3_text = visual.TextStim(win=win, name='with_intro_3_text',
        text='接着，您会看到一个短暂呈现的面孔图片，例如下图。\n请注意，您可能会看到同一个人出现多次，但不一定是同样的表情，\n您只需要对当前呈现的面孔情绪做出判断。\n',
        font='Open Sans',
        pos=(0, 0.15), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_3_key_resp = keyboard.Keyboard()
    with_intro_3_image = visual.ImageStim(
        win=win,
        name='with_intro_3_image', 
        image='stimuli/m_y50f_10.bmp', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "with_intro_4" ---
    with_intro_4_text = visual.TextStim(win=win, name='with_intro_4_text',
        text='面孔消失后，请尽快使用鼠标左键选择一个与面孔相符的情绪类别，\n被选中的情绪词周围会出现白色方框(如下所示)。\n\n',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_4_key_resp = keyboard.Keyboard()
    with_intro_4_rect_left = visual.Rect(
        win=win, name='with_intro_4_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, -0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    with_intro_4_left_calm = visual.TextStim(win=win, name='with_intro_4_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    with_intro_4_right_fear = visual.TextStim(win=win, name='with_intro_4_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "with_intro_5" ---
    with_intro_5_text = visual.TextStim(win=win, name='with_intro_5_text',
        text='2.连续判断：\n\n如果您看到‘恐惧’和‘平静’两个文字且中间有一条横线（如下所示），\n这表示您需要对接下来呈现的面孔图片进行连续情绪判断。\n横线的最左端表示0%的恐惧（即100%的平静），最右端表示100%的恐惧。 \n\n',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_5_key_resp = keyboard.Keyboard()
    with_intro_5_left_calm = visual.TextStim(win=win, name='with_intro_5_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    with_intro_5_right_fear = visual.TextStim(win=win, name='with_intro_5_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    with_intro_5_cont_fix_line = visual.Rect(
        win=win, name='with_intro_5_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, -0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "with_intro_6" ---
    with_intro_6_text = visual.TextStim(win=win, name='with_intro_6_text',
        text='接着，您会看到一个短暂呈现的面孔图片，例如下图。\n请注意，您可能会看到同一个人出现多次，但不一定是同样的表情，\n您只需要对当前呈现的面孔情绪做出判断。\n\n',
        font='Open Sans',
        pos=(0, 0.15), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_6_key_resp = keyboard.Keyboard()
    with_intro_6_image = visual.ImageStim(
        win=win,
        name='with_intro_6_image', 
        image='stimuli/m_y28m_70.bmp', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "with_intro_7" ---
    with_intro_7_text = visual.TextStim(win=win, name='with_intro_7_text',
        text='面孔消失后，您需要尽快对与面孔相符的情绪程度做出标记，\n在横线区域内移动鼠标（无需单击）即可沿线移动滑块，\n当移动到目标程度的位置时，点击鼠标左键即可完成判断。\n\n',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_7_key_resp = keyboard.Keyboard()
    with_intro_7_left_calm = visual.TextStim(win=win, name='with_intro_7_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    with_intro_7_right_fear = visual.TextStim(win=win, name='with_intro_7_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    with_intro_7_marker = visual.Rect(
        win=win, name='with_intro_7_marker',
        width=(0.01, 0.05)[0], height=(0.01, 0.05)[1],
        ori=0.0, pos=(0.1, -0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    with_intro_7_cont_fix_line = visual.Rect(
        win=win, name='with_intro_7_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, -0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    
    # --- Initialize components for Routine "with_intro_8" ---
    with_intro_8_text = visual.TextStim(win=win, name='with_intro_8_text',
        text='祝贺你！你已经了解了这一部分的实验程序。\n现在我们来练习一下。您需要对一系列图片进行情绪判断。\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_intro_8_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "prac_with_begin" ---
    prac_with_begin_text = visual.TextStim(win=win, name='prac_with_begin_text',
        text='请对接下来呈现的图片进行分类或连续判断，\n\n如果您看到‘恐惧’和‘平静’两个文字，\n这表示您需要对接下来呈现的面孔图片进行分类情绪判断，\n如果您看到‘恐惧’和‘平静’两个文字且中间有一条横线，\n这表示您需要对接下来呈现的面孔图片进行连续情绪判断。\n\n准备好按空格键开始。',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_with_begin_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "prac_with_mix_begin" ---
    prac_with_mix_begin_text = visual.TextStim(win=win, name='prac_with_mix_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_with_mix_begin_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "prac_with_mix_cate_fix" ---
    prac_with_mix_cate_fix_text = visual.TextStim(win=win, name='prac_with_mix_cate_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_with_mix_cate_fix_left_calm = visual.TextStim(win=win, name='prac_with_mix_cate_fix_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_with_mix_cate_fix_right_fear = visual.TextStim(win=win, name='prac_with_mix_cate_fix_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "prac_with_mix_cont_fix" ---
    prac_with_mix_cont_fix_text = visual.TextStim(win=win, name='prac_with_mix_cont_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_with_mix_cont_fix_left_calm = visual.TextStim(win=win, name='prac_with_mix_cont_fix_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_with_mix_cont_fix_right_fear = visual.TextStim(win=win, name='prac_with_mix_cont_fix_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_with_mix_cont_fix_line = visual.Rect(
        win=win, name='prac_with_mix_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "prac_with_mix_face" ---
    prac_with_mix_face_image = visual.ImageStim(
        win=win,
        name='prac_with_mix_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "prac_with_mix_cate_resp" ---
    prac_with_mix_cate_resp_prompt = visual.TextStim(win=win, name='prac_with_mix_cate_resp_prompt',
        text='请选择情绪类别',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_with_mix_cate_resp_rect_left = visual.Rect(
        win=win, name='prac_with_mix_cate_resp_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    prac_with_mix_cate_resp_rect_right = visual.Rect(
        win=win, name='prac_with_mix_cate_resp_rect_right',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(0.3, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    prac_with_mix_cate_resp_left_calm = visual.TextStim(win=win, name='prac_with_mix_cate_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_with_mix_cate_resp_right_fear = visual.TextStim(win=win, name='prac_with_mix_cate_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    prac_with_mix_cate_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_with_mix_cate_resp_mouse.mouseClock = core.Clock()
    # Run 'Begin Experiment' code from prac_cate_resp_code
    # 在 'Begin Experiment' 代码框中
    win.mouseVisible = False
    
    
    
    
    
    # --- Initialize components for Routine "prac_with_mix_cont_resp" ---
    prac_with_mix_cont_resp_prompt = visual.TextStim(win=win, name='prac_with_mix_cont_resp_prompt',
        text='请选择情绪程度',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_with_mix_cont_resp_slider_shape = visual.Rect(
        win=win, name='prac_with_mix_cont_resp_slider_shape',
        width=(0.6, 0.07)[0], height=(0.6, 0.07)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    prac_with_mix_cont_resp_slider = visual.Slider(win=win, name='prac_with_mix_cont_resp_slider',
        startValue=None, size=(0.5, 0.05), pos=(0, 0), units=win.units,
        labels=None, ticks=(0, 1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor=[1.0000, 1.0000, 1.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    prac_with_mix_cont_resp_left_calm = visual.TextStim(win=win, name='prac_with_mix_cont_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_with_mix_cont_resp_right_fear = visual.TextStim(win=win, name='prac_with_mix_cont_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    prac_with_mix_cont_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_with_mix_cont_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prac_end" ---
    prac_end_text = visual.TextStim(win=win, name='prac_end_text',
        text='练习部分到此为止。\n如果您有任何疑问，请咨询实验员。\n\n请对接下来呈现的图片进行分类或连续判断，\n\n如果您看到‘恐惧’和‘平静’两个文字，\n这表示您需要对接下来呈现的面孔图片进行分类情绪判断，\n如果您看到‘恐惧’和‘平静’两个文字且中间有一条横线，\n这表示您需要对接下来呈现的面孔图片进行连续情绪判断。\n\n如果您已经准备好，请按“enter”键开始正式实验。\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_end_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "with_mix_begin" ---
    with_mix_begin_text = visual.TextStim(win=win, name='with_mix_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_mix_begin_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "with_mix_cate_fix" ---
    with_mix_cate_fix_text = visual.TextStim(win=win, name='with_mix_cate_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_mix_cate_fix_left_calm = visual.TextStim(win=win, name='with_mix_cate_fix_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    with_mix_cate_fix_right_fear = visual.TextStim(win=win, name='with_mix_cate_fix_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    # Run 'Begin Experiment' code from with_mix_cate_fix_Init_trial
    trialID = 0
    
    # --- Initialize components for Routine "with_mix_cont_fix" ---
    with_mix_cont_fix_text = visual.TextStim(win=win, name='with_mix_cont_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_mix_cont_fix_left_calm = visual.TextStim(win=win, name='with_mix_cont_fix_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    with_mix_cont_fix_right_fear = visual.TextStim(win=win, name='with_mix_cont_fix_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    with_mix_cont_fix_line = visual.Rect(
        win=win, name='with_mix_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "with_mix_face" ---
    with_mix_face_image = visual.ImageStim(
        win=win,
        name='with_mix_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "with_mix_cate_resp" ---
    with_mix_cate_resp_prompt = visual.TextStim(win=win, name='with_mix_cate_resp_prompt',
        text='请选择情绪类别',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_mix_cate_resp_rect_left = visual.Rect(
        win=win, name='with_mix_cate_resp_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    with_mix_cate_resp_rect_right = visual.Rect(
        win=win, name='with_mix_cate_resp_rect_right',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(0.3, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    with_mix_cate_resp_left_calm = visual.TextStim(win=win, name='with_mix_cate_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    with_mix_cate_resp_right_fear = visual.TextStim(win=win, name='with_mix_cate_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    with_mix_cate_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    with_mix_cate_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "with_mix_cont_resp" ---
    with_mix_cont_resp_prompt = visual.TextStim(win=win, name='with_mix_cont_resp_prompt',
        text='请选择情绪程度',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_mix_cont_resp_slider_shape = visual.Rect(
        win=win, name='with_mix_cont_resp_slider_shape',
        width=(0.6, 0.07)[0], height=(0.6, 0.07)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    with_mix_cont_resp_slider = visual.Slider(win=win, name='with_mix_cont_resp_slider',
        startValue=None, size=(0.5, 0.05), pos=(0, 0), units=win.units,
        labels=None, ticks=(0, 1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor=[1.0000, 1.0000, 1.0000], lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    with_mix_cont_resp_left_calm = visual.TextStim(win=win, name='with_mix_cont_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    with_mix_cont_resp_right_fear = visual.TextStim(win=win, name='with_mix_cont_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    with_mix_cont_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    with_mix_cont_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "with_mix_brea" ---
    with_brea_text = visual.TextStim(win=win, name='with_brea_text',
        text='现在，请您稍事休息，\n当您感觉准备好继续实验时，\n请按下“enter”键继续进行实验任务。\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    with_brea_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "formal_wo_end" ---
    formal_wo_end_text = visual.TextStim(win=win, name='formal_wo_end_text',
        text='第一部分的实验到此为止。恭喜你已经完成了整个实验的50%。\n请您稍事休息。\n如果您已经准备好，请按键盘“enter”键继续第二部分的实验。\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    formal_wo_end_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "intro_2" ---
    intro_2_text = visual.TextStim(win=win, name='intro_2_text',
        text='首先，您会看到一个短暂呈现的面孔图片，例如下图。\n请注意，您可能会看到同一个人出现多次，但不一定是同样的表情，\n您只需要对当前呈现的面孔情绪做出判断。\n',
        font='Open Sans',
        pos=(0, 0.15), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_2_key_resp = keyboard.Keyboard()
    intro_2_image = visual.ImageStim(
        win=win,
        name='intro_2_image', 
        image='stimuli/m_y50f_10.bmp', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "intro_3" ---
    intro_3_text = visual.TextStim(win=win, name='intro_3_text',
        text='接着，您可能会进行2种判断任务中的一种。\n1.分类判断：\n面孔消失后，如果您看到‘恐惧’和‘平静’两个文字（如下所示），\n这表示您需要对刚才呈现的面孔图片进行分类情绪判断，\n即从"恐惧"或"平静"两个选项中任选其一。',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_3_key_resp = keyboard.Keyboard()
    intro_3_left_calm = visual.TextStim(win=win, name='intro_3_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    intro_3_right_fear = visual.TextStim(win=win, name='intro_3_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "intro_4" ---
    intro_4_text = visual.TextStim(win=win, name='intro_4_text',
        text='您需要尽快使用鼠标左键选择一个与面孔相符的情绪类别，\n被选中的情绪词周围会出现白色方框(如下所示)。\n',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_4_key_resp = keyboard.Keyboard()
    intro_4_rect_left = visual.Rect(
        win=win, name='intro_4_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, -0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    intro_4_left_calm = visual.TextStim(win=win, name='intro_4_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_4_right_fear = visual.TextStim(win=win, name='intro_4_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "intro_5" ---
    intro_5_text = visual.TextStim(win=win, name='intro_5_text',
        text='2.连续判断：\n面孔消失后，如果您看到‘恐惧’和‘平静’两个文字且中间有一条横线（如下所示），\n这表示您需要对刚才呈现的面孔图片进行连续情绪判断。\n横线的最左端表示0%的恐惧（即100%的平静），最右端表示100%的恐惧。 \n\n',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_5_key_resp = keyboard.Keyboard()
    intro_5_left_calm = visual.TextStim(win=win, name='intro_5_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    intro_5_right_fear = visual.TextStim(win=win, name='intro_5_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_5_cont_fix_line = visual.Rect(
        win=win, name='intro_5_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, -0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "intro_6" ---
    intro_6_text = visual.TextStim(win=win, name='intro_6_text',
        text='您需要尽快对与面孔相符的情绪程度做出标记，\n在横线区域内移动鼠标（无需单击）即可沿线移动滑块，\n当移动到目标程度的位置时，点击鼠标左键即可完成判断。\n',
        font='Open Sans',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_6_key_resp = keyboard.Keyboard()
    intro_6_left_calm = visual.TextStim(win=win, name='intro_6_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    intro_6_right_fear = visual.TextStim(win=win, name='intro_6_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_6_marker = visual.Rect(
        win=win, name='intro_6_marker',
        width=(0.01, 0.05)[0], height=(0.01, 0.05)[1],
        ori=0.0, pos=(0.1, -0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    intro_6_cont_fix_line = visual.Rect(
        win=win, name='intro_6_cont_fix_line',
        width=(0.5, 0.005)[0], height=(0.5, 0.005)[1],
        ori=0.0, pos=(0, -0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    
    # --- Initialize components for Routine "intro_7" ---
    intro_7_text = visual.TextStim(win=win, name='intro_7_text',
        text='祝贺你！你已经了解了实验程序。\n现在我们来练习一下。您需要对一系列图片进行情绪判断。\n\n准备好按空格键开始。\n\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_7_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "prac_wo_mix_begin" ---
    prac_wo_mix_begin_text = visual.TextStim(win=win, name='prac_wo_mix_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_wo_mix_begin_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "prac_wo_mix_fix" ---
    prac_wo_mix_fix_text = visual.TextStim(win=win, name='prac_wo_mix_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_wo_mix_face" ---
    prac_wo_mix_face_image = visual.ImageStim(
        win=win,
        name='prac_wo_mix_face_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "prac_wo_mix_cate_resp" ---
    prac_wo_mix_cate_resp_prompt = visual.TextStim(win=win, name='prac_wo_mix_cate_resp_prompt',
        text='请选择情绪类别',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_wo_mix_cate_resp_rect_left = visual.Rect(
        win=win, name='prac_wo_mix_cate_resp_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    prac_wo_mix_cate_resp_rect_right = visual.Rect(
        win=win, name='prac_wo_mix_cate_resp_rect_right',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(0.3, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    prac_wo_mix_cate_resp_left_calm = visual.TextStim(win=win, name='prac_wo_mix_cate_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_wo_mix_cate_resp_right_fear = visual.TextStim(win=win, name='prac_wo_mix_cate_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
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
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_wo_mix_cont_resp_slider_shape = visual.Rect(
        win=win, name='prac_wo_mix_cont_resp_slider_shape',
        width=(0.6, 0.07)[0], height=(0.6, 0.07)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
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
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_wo_mix_cont_resp_right_fear = visual.TextStim(win=win, name='prac_wo_mix_cont_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    prac_wo_mix_cont_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    prac_wo_mix_cont_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prac_wo_end" ---
    prac_wo_end_text = visual.TextStim(win=win, name='prac_wo_end_text',
        text='练习部分到此为止。\n如果您有任何疑问，请咨询实验员。\n如果您已经准备好，请按键盘“enter”键开始正式实验。\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_wo_end_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "wo_mix_begin" ---
    wo_mix_begin_text = visual.TextStim(win=win, name='wo_mix_begin_text',
        text='按空格键开始',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wo_mix_begin_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "wo_mix_fix" ---
    wo_mix_fix_text = visual.TextStim(win=win, name='wo_mix_fix_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
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
        ori=0.0, pos=(0, 0), size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "wo_mix_cate_resp" ---
    wo_mix_cate_resp_prompt = visual.TextStim(win=win, name='wo_mix_cate_resp_prompt',
        text='请选择情绪类别',
        font='Open Sans',
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wo_mix_cate_resp_rect_left = visual.Rect(
        win=win, name='wo_mix_cate_resp_rect_left',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(-0.3, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    wo_mix_cate_resp_rect_right = visual.Rect(
        win=win, name='wo_mix_cate_resp_rect_right',
        width=(0.15, 0.1)[0], height=(0.15, 0.1)[1],
        ori=0.0, pos=(0.3, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    wo_mix_cate_resp_left_calm = visual.TextStim(win=win, name='wo_mix_cate_resp_left_calm',
        text='平静',
        font='Open Sans',
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    wo_mix_cate_resp_right_fear = visual.TextStim(win=win, name='wo_mix_cate_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
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
        pos=(0, 0.2), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wo_mix_cont_resp_slider_shape = visual.Rect(
        win=win, name='wo_mix_cont_resp_slider_shape',
        width=(0.6, 0.07)[0], height=(0.6, 0.07)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
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
        pos=(-0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    wo_mix_cont_resp_right_fear = visual.TextStim(win=win, name='wo_mix_cont_resp_right_fear',
        text='恐惧',
        font='Open Sans',
        pos=(0.3, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    wo_mix_cont_resp_mouse = event.Mouse(win=win)
    x, y = [None, None]
    wo_mix_cont_resp_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "wo_mix_brea" ---
    wo_mix_brea_text = visual.TextStim(win=win, name='wo_mix_brea_text',
        text='现在，请您稍事休息，\n当您感觉准备好继续实验时，\n请按下“enter”键继续进行实验任务。\n',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    wo_mix_brea_key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "end_exp" ---
    end_exp_text = visual.TextStim(win=win, name='end_exp_text',
        text='实验任务已全部完成，感谢您的参与！',
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "intro" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro.started', globalClock.getTime())
    intro_key_resp.keys = []
    intro_key_resp.rt = []
    _intro_key_resp_allKeys = []
    # keep track of which components have finished
    introComponents = [intro_text, intro_key_resp]
    for thisComponent in introComponents:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in introComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro.stopped', globalClock.getTime())
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
    
    # --- Prepare to start Routine "with_intro_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('with_intro_2.started', globalClock.getTime())
    with_intro_2_key_resp.keys = []
    with_intro_2_key_resp.rt = []
    _with_intro_2_key_resp_allKeys = []
    # keep track of which components have finished
    with_intro_2Components = [with_intro_2_text, with_intro_2_key_resp, with_intro_2_left_calm, with_intro_2_right_fear]
    for thisComponent in with_intro_2Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_2" ---
    for thisComponent in with_intro_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('with_intro_2.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('with_intro_3.started', globalClock.getTime())
    with_intro_3_key_resp.keys = []
    with_intro_3_key_resp.rt = []
    _with_intro_3_key_resp_allKeys = []
    # keep track of which components have finished
    with_intro_3Components = [with_intro_3_text, with_intro_3_key_resp, with_intro_3_image]
    for thisComponent in with_intro_3Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_3" ---
    for thisComponent in with_intro_3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('with_intro_3.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('with_intro_4.started', globalClock.getTime())
    with_intro_4_key_resp.keys = []
    with_intro_4_key_resp.rt = []
    _with_intro_4_key_resp_allKeys = []
    # keep track of which components have finished
    with_intro_4Components = [with_intro_4_text, with_intro_4_key_resp, with_intro_4_rect_left, with_intro_4_left_calm, with_intro_4_right_fear]
    for thisComponent in with_intro_4Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_4Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_4" ---
    for thisComponent in with_intro_4Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('with_intro_4.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('with_intro_5.started', globalClock.getTime())
    with_intro_5_key_resp.keys = []
    with_intro_5_key_resp.rt = []
    _with_intro_5_key_resp_allKeys = []
    # keep track of which components have finished
    with_intro_5Components = [with_intro_5_text, with_intro_5_key_resp, with_intro_5_left_calm, with_intro_5_right_fear, with_intro_5_cont_fix_line]
    for thisComponent in with_intro_5Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_5Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_5" ---
    for thisComponent in with_intro_5Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('with_intro_5.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('with_intro_6.started', globalClock.getTime())
    with_intro_6_key_resp.keys = []
    with_intro_6_key_resp.rt = []
    _with_intro_6_key_resp_allKeys = []
    # keep track of which components have finished
    with_intro_6Components = [with_intro_6_text, with_intro_6_key_resp, with_intro_6_image]
    for thisComponent in with_intro_6Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_6Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_6" ---
    for thisComponent in with_intro_6Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('with_intro_6.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('with_intro_7.started', globalClock.getTime())
    with_intro_7_key_resp.keys = []
    with_intro_7_key_resp.rt = []
    _with_intro_7_key_resp_allKeys = []
    # keep track of which components have finished
    with_intro_7Components = [with_intro_7_text, with_intro_7_key_resp, with_intro_7_left_calm, with_intro_7_right_fear, with_intro_7_marker, with_intro_7_cont_fix_line]
    for thisComponent in with_intro_7Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_7Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_7" ---
    for thisComponent in with_intro_7Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('with_intro_7.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('with_intro_8.started', globalClock.getTime())
    with_intro_8_key_resp.keys = []
    with_intro_8_key_resp.rt = []
    _with_intro_8_key_resp_allKeys = []
    # keep track of which components have finished
    with_intro_8Components = [with_intro_8_text, with_intro_8_key_resp]
    for thisComponent in with_intro_8Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in with_intro_8Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "with_intro_8" ---
    for thisComponent in with_intro_8Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('with_intro_8.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('prac_with_begin.started', globalClock.getTime())
    prac_with_begin_key_resp.keys = []
    prac_with_begin_key_resp.rt = []
    _prac_with_begin_key_resp_allKeys = []
    # keep track of which components have finished
    prac_with_beginComponents = [prac_with_begin_text, prac_with_begin_key_resp]
    for thisComponent in prac_with_beginComponents:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_with_beginComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_with_begin" ---
    for thisComponent in prac_with_beginComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('prac_with_begin.stopped', globalClock.getTime())
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
    prac_with_mix = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('pics - 4.5.xlsx', selection='0:6'),
        seed=None, name='prac_with_mix')
    thisExp.addLoop(prac_with_mix)  # add the loop to the experiment
    thisPrac_with_mix = prac_with_mix.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPrac_with_mix.rgb)
    if thisPrac_with_mix != None:
        for paramName in thisPrac_with_mix:
            globals()[paramName] = thisPrac_with_mix[paramName]
    
    for thisPrac_with_mix in prac_with_mix:
        currentLoop = prac_with_mix
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_with_mix.rgb)
        if thisPrac_with_mix != None:
            for paramName in thisPrac_with_mix:
                globals()[paramName] = thisPrac_with_mix[paramName]
        
        # --- Prepare to start Routine "prac_with_mix_begin" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_with_mix_begin.started', globalClock.getTime())
        prac_with_mix_begin_key_resp.keys = []
        prac_with_mix_begin_key_resp.rt = []
        _prac_with_mix_begin_key_resp_allKeys = []
        # keep track of which components have finished
        prac_with_mix_beginComponents = [prac_with_mix_begin_text, prac_with_mix_begin_key_resp]
        for thisComponent in prac_with_mix_beginComponents:
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
        
        # --- Run Routine "prac_with_mix_begin" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_with_mix_begin_text* updates
            
            # if prac_with_mix_begin_text is starting this frame...
            if prac_with_mix_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_begin_text.frameNStart = frameN  # exact frame index
                prac_with_mix_begin_text.tStart = t  # local t and not account for scr refresh
                prac_with_mix_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_begin_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_begin_text.started')
                # update status
                prac_with_mix_begin_text.status = STARTED
                prac_with_mix_begin_text.setAutoDraw(True)
            
            # if prac_with_mix_begin_text is active this frame...
            if prac_with_mix_begin_text.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_begin_key_resp* updates
            waitOnFlip = False
            
            # if prac_with_mix_begin_key_resp is starting this frame...
            if prac_with_mix_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_begin_key_resp.frameNStart = frameN  # exact frame index
                prac_with_mix_begin_key_resp.tStart = t  # local t and not account for scr refresh
                prac_with_mix_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_begin_key_resp.started')
                # update status
                prac_with_mix_begin_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_with_mix_begin_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_with_mix_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_with_mix_begin_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_with_mix_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _prac_with_mix_begin_key_resp_allKeys.extend(theseKeys)
                if len(_prac_with_mix_begin_key_resp_allKeys):
                    prac_with_mix_begin_key_resp.keys = _prac_with_mix_begin_key_resp_allKeys[-1].name  # just the last key pressed
                    prac_with_mix_begin_key_resp.rt = _prac_with_mix_begin_key_resp_allKeys[-1].rt
                    prac_with_mix_begin_key_resp.duration = _prac_with_mix_begin_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_with_mix_beginComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_with_mix_begin" ---
        for thisComponent in prac_with_mix_beginComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_with_mix_begin.stopped', globalClock.getTime())
        # check responses
        if prac_with_mix_begin_key_resp.keys in ['', [], None]:  # No response was made
            prac_with_mix_begin_key_resp.keys = None
        prac_with_mix.addData('prac_with_mix_begin_key_resp.keys',prac_with_mix_begin_key_resp.keys)
        if prac_with_mix_begin_key_resp.keys != None:  # we had a response
            prac_with_mix.addData('prac_with_mix_begin_key_resp.rt', prac_with_mix_begin_key_resp.rt)
            prac_with_mix.addData('prac_with_mix_begin_key_resp.duration', prac_with_mix_begin_key_resp.duration)
        # the Routine "prac_with_mix_begin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_with_mix_cate_fix" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_with_mix_cate_fix.started', globalClock.getTime())
        # Run 'Begin Routine' code from prac_with_mix_cate_fix_code
        # 检查当前试次应该运行的例程
        if prac_with_mix.thisTrial['with_prac_mix_type'] == 'prac_with_mix_cate_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        # keep track of which components have finished
        prac_with_mix_cate_fixComponents = [prac_with_mix_cate_fix_text, prac_with_mix_cate_fix_left_calm, prac_with_mix_cate_fix_right_fear]
        for thisComponent in prac_with_mix_cate_fixComponents:
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
        
        # --- Run Routine "prac_with_mix_cate_fix" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_with_mix_cate_fix_text* updates
            
            # if prac_with_mix_cate_fix_text is starting this frame...
            if prac_with_mix_cate_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_fix_text.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_fix_text.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cate_fix_text.started')
                # update status
                prac_with_mix_cate_fix_text.status = STARTED
                prac_with_mix_cate_fix_text.setAutoDraw(True)
            
            # if prac_with_mix_cate_fix_text is active this frame...
            if prac_with_mix_cate_fix_text.status == STARTED:
                # update params
                pass
            
            # if prac_with_mix_cate_fix_text is stopping this frame...
            if prac_with_mix_cate_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_with_mix_cate_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_with_mix_cate_fix_text.tStop = t  # not accounting for scr refresh
                    prac_with_mix_cate_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_with_mix_cate_fix_text.stopped')
                    # update status
                    prac_with_mix_cate_fix_text.status = FINISHED
                    prac_with_mix_cate_fix_text.setAutoDraw(False)
            
            # *prac_with_mix_cate_fix_left_calm* updates
            
            # if prac_with_mix_cate_fix_left_calm is starting this frame...
            if prac_with_mix_cate_fix_left_calm.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_fix_left_calm.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_fix_left_calm.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_fix_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_fix_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cate_fix_left_calm.started')
                # update status
                prac_with_mix_cate_fix_left_calm.status = STARTED
                prac_with_mix_cate_fix_left_calm.setAutoDraw(True)
            
            # if prac_with_mix_cate_fix_left_calm is active this frame...
            if prac_with_mix_cate_fix_left_calm.status == STARTED:
                # update params
                pass
            
            # if prac_with_mix_cate_fix_left_calm is stopping this frame...
            if prac_with_mix_cate_fix_left_calm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_with_mix_cate_fix_left_calm.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_with_mix_cate_fix_left_calm.tStop = t  # not accounting for scr refresh
                    prac_with_mix_cate_fix_left_calm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_with_mix_cate_fix_left_calm.stopped')
                    # update status
                    prac_with_mix_cate_fix_left_calm.status = FINISHED
                    prac_with_mix_cate_fix_left_calm.setAutoDraw(False)
            
            # *prac_with_mix_cate_fix_right_fear* updates
            
            # if prac_with_mix_cate_fix_right_fear is starting this frame...
            if prac_with_mix_cate_fix_right_fear.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_fix_right_fear.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_fix_right_fear.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_fix_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_fix_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cate_fix_right_fear.started')
                # update status
                prac_with_mix_cate_fix_right_fear.status = STARTED
                prac_with_mix_cate_fix_right_fear.setAutoDraw(True)
            
            # if prac_with_mix_cate_fix_right_fear is active this frame...
            if prac_with_mix_cate_fix_right_fear.status == STARTED:
                # update params
                pass
            
            # if prac_with_mix_cate_fix_right_fear is stopping this frame...
            if prac_with_mix_cate_fix_right_fear.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_with_mix_cate_fix_right_fear.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_with_mix_cate_fix_right_fear.tStop = t  # not accounting for scr refresh
                    prac_with_mix_cate_fix_right_fear.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_with_mix_cate_fix_right_fear.stopped')
                    # update status
                    prac_with_mix_cate_fix_right_fear.status = FINISHED
                    prac_with_mix_cate_fix_right_fear.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_with_mix_cate_fixComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_with_mix_cate_fix" ---
        for thisComponent in prac_with_mix_cate_fixComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_with_mix_cate_fix.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "prac_with_mix_cont_fix" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_with_mix_cont_fix.started', globalClock.getTime())
        # Run 'Begin Routine' code from prac_with_mix_cont_fix_code
        # 检查当前试次应该运行的例程
        if prac_with_mix.thisTrial['with_prac_mix_type'] == 'prac_with_mix_cont_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        # keep track of which components have finished
        prac_with_mix_cont_fixComponents = [prac_with_mix_cont_fix_text, prac_with_mix_cont_fix_left_calm, prac_with_mix_cont_fix_right_fear, prac_with_mix_cont_fix_line]
        for thisComponent in prac_with_mix_cont_fixComponents:
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
        
        # --- Run Routine "prac_with_mix_cont_fix" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_with_mix_cont_fix_text* updates
            
            # if prac_with_mix_cont_fix_text is starting this frame...
            if prac_with_mix_cont_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_fix_text.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_fix_text.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_fix_text.started')
                # update status
                prac_with_mix_cont_fix_text.status = STARTED
                prac_with_mix_cont_fix_text.setAutoDraw(True)
            
            # if prac_with_mix_cont_fix_text is active this frame...
            if prac_with_mix_cont_fix_text.status == STARTED:
                # update params
                pass
            
            # if prac_with_mix_cont_fix_text is stopping this frame...
            if prac_with_mix_cont_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_with_mix_cont_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_with_mix_cont_fix_text.tStop = t  # not accounting for scr refresh
                    prac_with_mix_cont_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_with_mix_cont_fix_text.stopped')
                    # update status
                    prac_with_mix_cont_fix_text.status = FINISHED
                    prac_with_mix_cont_fix_text.setAutoDraw(False)
            
            # *prac_with_mix_cont_fix_left_calm* updates
            
            # if prac_with_mix_cont_fix_left_calm is starting this frame...
            if prac_with_mix_cont_fix_left_calm.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_fix_left_calm.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_fix_left_calm.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_fix_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_fix_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_fix_left_calm.started')
                # update status
                prac_with_mix_cont_fix_left_calm.status = STARTED
                prac_with_mix_cont_fix_left_calm.setAutoDraw(True)
            
            # if prac_with_mix_cont_fix_left_calm is active this frame...
            if prac_with_mix_cont_fix_left_calm.status == STARTED:
                # update params
                pass
            
            # if prac_with_mix_cont_fix_left_calm is stopping this frame...
            if prac_with_mix_cont_fix_left_calm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_with_mix_cont_fix_left_calm.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_with_mix_cont_fix_left_calm.tStop = t  # not accounting for scr refresh
                    prac_with_mix_cont_fix_left_calm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_with_mix_cont_fix_left_calm.stopped')
                    # update status
                    prac_with_mix_cont_fix_left_calm.status = FINISHED
                    prac_with_mix_cont_fix_left_calm.setAutoDraw(False)
            
            # *prac_with_mix_cont_fix_right_fear* updates
            
            # if prac_with_mix_cont_fix_right_fear is starting this frame...
            if prac_with_mix_cont_fix_right_fear.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_fix_right_fear.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_fix_right_fear.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_fix_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_fix_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_fix_right_fear.started')
                # update status
                prac_with_mix_cont_fix_right_fear.status = STARTED
                prac_with_mix_cont_fix_right_fear.setAutoDraw(True)
            
            # if prac_with_mix_cont_fix_right_fear is active this frame...
            if prac_with_mix_cont_fix_right_fear.status == STARTED:
                # update params
                pass
            
            # if prac_with_mix_cont_fix_right_fear is stopping this frame...
            if prac_with_mix_cont_fix_right_fear.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_with_mix_cont_fix_right_fear.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_with_mix_cont_fix_right_fear.tStop = t  # not accounting for scr refresh
                    prac_with_mix_cont_fix_right_fear.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_with_mix_cont_fix_right_fear.stopped')
                    # update status
                    prac_with_mix_cont_fix_right_fear.status = FINISHED
                    prac_with_mix_cont_fix_right_fear.setAutoDraw(False)
            
            # *prac_with_mix_cont_fix_line* updates
            
            # if prac_with_mix_cont_fix_line is starting this frame...
            if prac_with_mix_cont_fix_line.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_fix_line.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_fix_line.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_fix_line.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_fix_line, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_fix_line.started')
                # update status
                prac_with_mix_cont_fix_line.status = STARTED
                prac_with_mix_cont_fix_line.setAutoDraw(True)
            
            # if prac_with_mix_cont_fix_line is active this frame...
            if prac_with_mix_cont_fix_line.status == STARTED:
                # update params
                pass
            
            # if prac_with_mix_cont_fix_line is stopping this frame...
            if prac_with_mix_cont_fix_line.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_with_mix_cont_fix_line.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_with_mix_cont_fix_line.tStop = t  # not accounting for scr refresh
                    prac_with_mix_cont_fix_line.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_with_mix_cont_fix_line.stopped')
                    # update status
                    prac_with_mix_cont_fix_line.status = FINISHED
                    prac_with_mix_cont_fix_line.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_with_mix_cont_fixComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_with_mix_cont_fix" ---
        for thisComponent in prac_with_mix_cont_fixComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_with_mix_cont_fix.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "prac_with_mix_face" ---
        continueRoutine = True
        # update component parameters for each repeat
        prac_with_mix_face_image.setImage(with_prac_mix)
        # keep track of which components have finished
        prac_with_mix_faceComponents = [prac_with_mix_face_image]
        for thisComponent in prac_with_mix_faceComponents:
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
        
        # --- Run Routine "prac_with_mix_face" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_with_mix_face_image* updates
            
            # if prac_with_mix_face_image is starting this frame...
            if prac_with_mix_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_face_image.frameNStart = frameN  # exact frame index
                prac_with_mix_face_image.tStart = t  # local t and not account for scr refresh
                prac_with_mix_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_face_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_face_image.started')
                # update status
                prac_with_mix_face_image.status = STARTED
                prac_with_mix_face_image.setAutoDraw(True)
            
            # if prac_with_mix_face_image is active this frame...
            if prac_with_mix_face_image.status == STARTED:
                # update params
                pass
            
            # if prac_with_mix_face_image is stopping this frame...
            if prac_with_mix_face_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_with_mix_face_image.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_with_mix_face_image.tStop = t  # not accounting for scr refresh
                    prac_with_mix_face_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_with_mix_face_image.stopped')
                    # update status
                    prac_with_mix_face_image.status = FINISHED
                    prac_with_mix_face_image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_with_mix_faceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_with_mix_face" ---
        for thisComponent in prac_with_mix_faceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "prac_with_mix_cate_resp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_with_mix_cate_resp.started', globalClock.getTime())
        # setup some python lists for storing info about the prac_with_mix_cate_resp_mouse
        prac_with_mix_cate_resp_mouse.x = []
        prac_with_mix_cate_resp_mouse.y = []
        prac_with_mix_cate_resp_mouse.leftButton = []
        prac_with_mix_cate_resp_mouse.midButton = []
        prac_with_mix_cate_resp_mouse.rightButton = []
        prac_with_mix_cate_resp_mouse.time = []
        prac_with_mix_cate_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from prac_cate_resp_code
        # 设置Rectangle的初始状态为不可见
        prac_with_mix_cate_resp_rect_left.opacity = 0
        prac_with_mix_cate_resp_rect_right.opacity = 0
        
        # 在您的循环之前，设置鼠标的初始位置（在屏幕中心）
        prac_with_mix_cate_resp_mouse.setPos((0, 0))
        
        # 检查当前试次应该运行的例程
        if prac_with_mix.thisTrial['with_prac_mix_type'] == 'prac_with_mix_cate_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        
        # keep track of which components have finished
        prac_with_mix_cate_respComponents = [prac_with_mix_cate_resp_prompt, prac_with_mix_cate_resp_rect_left, prac_with_mix_cate_resp_rect_right, prac_with_mix_cate_resp_left_calm, prac_with_mix_cate_resp_right_fear, prac_with_mix_cate_resp_mouse]
        for thisComponent in prac_with_mix_cate_respComponents:
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
        
        # --- Run Routine "prac_with_mix_cate_resp" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_with_mix_cate_resp_prompt* updates
            
            # if prac_with_mix_cate_resp_prompt is starting this frame...
            if prac_with_mix_cate_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_resp_prompt.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_resp_prompt.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cate_resp_prompt.started')
                # update status
                prac_with_mix_cate_resp_prompt.status = STARTED
                prac_with_mix_cate_resp_prompt.setAutoDraw(True)
            
            # if prac_with_mix_cate_resp_prompt is active this frame...
            if prac_with_mix_cate_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_cate_resp_rect_left* updates
            
            # if prac_with_mix_cate_resp_rect_left is starting this frame...
            if prac_with_mix_cate_resp_rect_left.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_resp_rect_left.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_resp_rect_left.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_resp_rect_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_resp_rect_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cate_resp_rect_left.started')
                # update status
                prac_with_mix_cate_resp_rect_left.status = STARTED
                prac_with_mix_cate_resp_rect_left.setAutoDraw(True)
            
            # if prac_with_mix_cate_resp_rect_left is active this frame...
            if prac_with_mix_cate_resp_rect_left.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_cate_resp_rect_right* updates
            
            # if prac_with_mix_cate_resp_rect_right is starting this frame...
            if prac_with_mix_cate_resp_rect_right.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_resp_rect_right.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_resp_rect_right.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_resp_rect_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_resp_rect_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cate_resp_rect_right.started')
                # update status
                prac_with_mix_cate_resp_rect_right.status = STARTED
                prac_with_mix_cate_resp_rect_right.setAutoDraw(True)
            
            # if prac_with_mix_cate_resp_rect_right is active this frame...
            if prac_with_mix_cate_resp_rect_right.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_cate_resp_left_calm* updates
            
            # if prac_with_mix_cate_resp_left_calm is starting this frame...
            if prac_with_mix_cate_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_resp_left_calm.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_resp_left_calm.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cate_resp_left_calm.started')
                # update status
                prac_with_mix_cate_resp_left_calm.status = STARTED
                prac_with_mix_cate_resp_left_calm.setAutoDraw(True)
            
            # if prac_with_mix_cate_resp_left_calm is active this frame...
            if prac_with_mix_cate_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_cate_resp_right_fear* updates
            
            # if prac_with_mix_cate_resp_right_fear is starting this frame...
            if prac_with_mix_cate_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_resp_right_fear.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_resp_right_fear.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cate_resp_right_fear.started')
                # update status
                prac_with_mix_cate_resp_right_fear.status = STARTED
                prac_with_mix_cate_resp_right_fear.setAutoDraw(True)
            
            # if prac_with_mix_cate_resp_right_fear is active this frame...
            if prac_with_mix_cate_resp_right_fear.status == STARTED:
                # update params
                pass
            # *prac_with_mix_cate_resp_mouse* updates
            
            # if prac_with_mix_cate_resp_mouse is starting this frame...
            if prac_with_mix_cate_resp_mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cate_resp_mouse.frameNStart = frameN  # exact frame index
                prac_with_mix_cate_resp_mouse.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cate_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cate_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('prac_with_mix_cate_resp_mouse.started', t)
                # update status
                prac_with_mix_cate_resp_mouse.status = STARTED
                prac_with_mix_cate_resp_mouse.mouseClock.reset()
                prevButtonState = prac_with_mix_cate_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if prac_with_mix_cate_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = prac_with_mix_cate_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([prac_with_mix_cate_resp_left_calm, prac_with_mix_cate_resp_right_fear], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(prac_with_mix_cate_resp_mouse):
                                gotValidClick = True
                                prac_with_mix_cate_resp_mouse.clicked_name.append(obj.name)
                        x, y = prac_with_mix_cate_resp_mouse.getPos()
                        prac_with_mix_cate_resp_mouse.x.append(x)
                        prac_with_mix_cate_resp_mouse.y.append(y)
                        buttons = prac_with_mix_cate_resp_mouse.getPressed()
                        prac_with_mix_cate_resp_mouse.leftButton.append(buttons[0])
                        prac_with_mix_cate_resp_mouse.midButton.append(buttons[1])
                        prac_with_mix_cate_resp_mouse.rightButton.append(buttons[2])
                        prac_with_mix_cate_resp_mouse.time.append(prac_with_mix_cate_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from prac_cate_resp_code
            # 检测鼠标是否悬停在左边单词上方
            if prac_with_mix_cate_resp_left_calm.contains(prac_with_mix_cate_resp_mouse):
                prac_with_mix_cate_resp_rect_left.opacity = 1  # 设置左边Rectangle可见
            else:
                prac_with_mix_cate_resp_rect_left.opacity = 0  # 设置左边Rectangle不可见
            
            # 检测鼠标是否悬停在右边单词上方
            if prac_with_mix_cate_resp_right_fear.contains(prac_with_mix_cate_resp_mouse):
                prac_with_mix_cate_resp_rect_right.opacity = 1  # 设置右边Rectangle可见
            else:
                prac_with_mix_cate_resp_rect_right.opacity = 0  # 设置右边Rectangle不可见
            
            # 检测鼠标点击事件
            if prac_with_mix_cate_resp_mouse.getPressed()[0]:
                # 检测鼠标点击是否在左边单词上方
                if prac_with_mix_cate_resp_left_calm.contains(prac_with_mix_cate_resp_mouse):
                    response = '0'  # 记录被试选择左边单词
                # 检测鼠标点击是否在右边单词上方
                elif prac_with_mix_cate_resp_right_fear.contains(prac_with_mix_cate_resp_mouse):
                    response = '1'  # 记录被试选择右边单词
            
            
            #mouse.setVisible(True)  # 显示鼠标
            win.mouseVisible = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_with_mix_cate_respComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_with_mix_cate_resp" ---
        for thisComponent in prac_with_mix_cate_respComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_with_mix_cate_resp.stopped', globalClock.getTime())
        # store data for prac_with_mix (TrialHandler)
        prac_with_mix.addData('prac_with_mix_cate_resp_mouse.x', prac_with_mix_cate_resp_mouse.x)
        prac_with_mix.addData('prac_with_mix_cate_resp_mouse.y', prac_with_mix_cate_resp_mouse.y)
        prac_with_mix.addData('prac_with_mix_cate_resp_mouse.leftButton', prac_with_mix_cate_resp_mouse.leftButton)
        prac_with_mix.addData('prac_with_mix_cate_resp_mouse.midButton', prac_with_mix_cate_resp_mouse.midButton)
        prac_with_mix.addData('prac_with_mix_cate_resp_mouse.rightButton', prac_with_mix_cate_resp_mouse.rightButton)
        prac_with_mix.addData('prac_with_mix_cate_resp_mouse.time', prac_with_mix_cate_resp_mouse.time)
        prac_with_mix.addData('prac_with_mix_cate_resp_mouse.clicked_name', prac_with_mix_cate_resp_mouse.clicked_name)
        # Run 'End Routine' code from prac_cate_resp_code
        win.mouseVisible = False
        # the Routine "prac_with_mix_cate_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_with_mix_cont_resp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_with_mix_cont_resp.started', globalClock.getTime())
        prac_with_mix_cont_resp_slider.reset()
        # setup some python lists for storing info about the prac_with_mix_cont_resp_mouse
        prac_with_mix_cont_resp_mouse.x = []
        prac_with_mix_cont_resp_mouse.y = []
        prac_with_mix_cont_resp_mouse.leftButton = []
        prac_with_mix_cont_resp_mouse.midButton = []
        prac_with_mix_cont_resp_mouse.rightButton = []
        prac_with_mix_cont_resp_mouse.time = []
        prac_with_mix_cont_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from prac_with_mix_cont_resp_code
        # Change the following parameters according to your own exp
        slider_width = 0.5
        slider_height = .05
        slider_orientation = 0
        #Failing to set ticks or labels in code
        slider_ticks=[0, 1]
        
        slider_granularity = 0
        slider_decimals = 2
        
        prac_with_mix_cont_resp_slider.decimals = 2
        
        # 获取Slider组件的标记（marker）
        marker = prac_with_mix_cont_resp_slider.marker
        # 将标记的形状更改为一个特定尺寸的方块
        marker.shape = 'square'
        marker.size = (0.01, 0.05)  # 设置方块的尺寸
        
        # create an empty list to store data
        slider_data=[]
        
        # record the current position of the mouse
        mouseRec=prac_with_mix_cont_resp_mouse.getPos()
        
        prac_with_mix_cont_resp_mouse.setPos((0, 0))
        
        # 检查当前试次应该运行的例程
        if prac_with_mix.thisTrial['with_prac_mix_type'] == 'prac_with_mix_cont_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        # keep track of which components have finished
        prac_with_mix_cont_respComponents = [prac_with_mix_cont_resp_prompt, prac_with_mix_cont_resp_slider_shape, prac_with_mix_cont_resp_slider, prac_with_mix_cont_resp_left_calm, prac_with_mix_cont_resp_right_fear, prac_with_mix_cont_resp_mouse]
        for thisComponent in prac_with_mix_cont_respComponents:
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
        
        # --- Run Routine "prac_with_mix_cont_resp" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_with_mix_cont_resp_prompt* updates
            
            # if prac_with_mix_cont_resp_prompt is starting this frame...
            if prac_with_mix_cont_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_resp_prompt.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_resp_prompt.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_resp_prompt.started')
                # update status
                prac_with_mix_cont_resp_prompt.status = STARTED
                prac_with_mix_cont_resp_prompt.setAutoDraw(True)
            
            # if prac_with_mix_cont_resp_prompt is active this frame...
            if prac_with_mix_cont_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_cont_resp_slider_shape* updates
            
            # if prac_with_mix_cont_resp_slider_shape is starting this frame...
            if prac_with_mix_cont_resp_slider_shape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_resp_slider_shape.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_resp_slider_shape.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_resp_slider_shape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_resp_slider_shape, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_resp_slider_shape.started')
                # update status
                prac_with_mix_cont_resp_slider_shape.status = STARTED
                prac_with_mix_cont_resp_slider_shape.setAutoDraw(True)
            
            # if prac_with_mix_cont_resp_slider_shape is active this frame...
            if prac_with_mix_cont_resp_slider_shape.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_cont_resp_slider* updates
            
            # if prac_with_mix_cont_resp_slider is starting this frame...
            if prac_with_mix_cont_resp_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_resp_slider.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_resp_slider.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_resp_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_resp_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_resp_slider.started')
                # update status
                prac_with_mix_cont_resp_slider.status = STARTED
                prac_with_mix_cont_resp_slider.setAutoDraw(True)
            
            # if prac_with_mix_cont_resp_slider is active this frame...
            if prac_with_mix_cont_resp_slider.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_cont_resp_left_calm* updates
            
            # if prac_with_mix_cont_resp_left_calm is starting this frame...
            if prac_with_mix_cont_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_resp_left_calm.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_resp_left_calm.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_resp_left_calm.started')
                # update status
                prac_with_mix_cont_resp_left_calm.status = STARTED
                prac_with_mix_cont_resp_left_calm.setAutoDraw(True)
            
            # if prac_with_mix_cont_resp_left_calm is active this frame...
            if prac_with_mix_cont_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *prac_with_mix_cont_resp_right_fear* updates
            
            # if prac_with_mix_cont_resp_right_fear is starting this frame...
            if prac_with_mix_cont_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_resp_right_fear.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_resp_right_fear.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_with_mix_cont_resp_right_fear.started')
                # update status
                prac_with_mix_cont_resp_right_fear.status = STARTED
                prac_with_mix_cont_resp_right_fear.setAutoDraw(True)
            
            # if prac_with_mix_cont_resp_right_fear is active this frame...
            if prac_with_mix_cont_resp_right_fear.status == STARTED:
                # update params
                pass
            # *prac_with_mix_cont_resp_mouse* updates
            
            # if prac_with_mix_cont_resp_mouse is starting this frame...
            if prac_with_mix_cont_resp_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_with_mix_cont_resp_mouse.frameNStart = frameN  # exact frame index
                prac_with_mix_cont_resp_mouse.tStart = t  # local t and not account for scr refresh
                prac_with_mix_cont_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_with_mix_cont_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('prac_with_mix_cont_resp_mouse.started', t)
                # update status
                prac_with_mix_cont_resp_mouse.status = STARTED
                prac_with_mix_cont_resp_mouse.mouseClock.reset()
                prevButtonState = prac_with_mix_cont_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if prac_with_mix_cont_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = prac_with_mix_cont_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(prac_with_mix_cont_resp_slider_shape, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(prac_with_mix_cont_resp_mouse):
                                gotValidClick = True
                                prac_with_mix_cont_resp_mouse.clicked_name.append(obj.name)
                        x, y = prac_with_mix_cont_resp_mouse.getPos()
                        prac_with_mix_cont_resp_mouse.x.append(x)
                        prac_with_mix_cont_resp_mouse.y.append(y)
                        buttons = prac_with_mix_cont_resp_mouse.getPressed()
                        prac_with_mix_cont_resp_mouse.leftButton.append(buttons[0])
                        prac_with_mix_cont_resp_mouse.midButton.append(buttons[1])
                        prac_with_mix_cont_resp_mouse.rightButton.append(buttons[2])
                        prac_with_mix_cont_resp_mouse.time.append(prac_with_mix_cont_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from prac_with_mix_cont_resp_code
            
            # Create a slider_shape in this routine with the same size as the slider
            # Check if mouse is moving within slider_shape, if so change marker position accordingly
            if prac_with_mix_cont_resp_slider_shape.contains(prac_with_mix_cont_resp_mouse) and prac_with_mix_cont_resp_mouse.getPos()[slider_orientation] != mouseRec[slider_orientation]:
                mouseRec = prac_with_mix_cont_resp_mouse.getPos()
                prac_with_mix_cont_resp_slider.markerPos = mouseRec[slider_orientation]/slider_width*(slider_ticks[-1]-slider_ticks[0])+(slider_ticks[0]+slider_ticks[-1])/2
                
                # (new)Record marker position only on left mouse click
                slider_data.append([round(prac_with_mix_cont_resp_slider.markerPos, slider_decimals)])
            
            
            win.mouseVisible = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_with_mix_cont_respComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_with_mix_cont_resp" ---
        for thisComponent in prac_with_mix_cont_respComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_with_mix_cont_resp.stopped', globalClock.getTime())
        prac_with_mix.addData('prac_with_mix_cont_resp_slider.response', prac_with_mix_cont_resp_slider.getRating())
        prac_with_mix.addData('prac_with_mix_cont_resp_slider.rt', prac_with_mix_cont_resp_slider.getRT())
        # store data for prac_with_mix (TrialHandler)
        prac_with_mix.addData('prac_with_mix_cont_resp_mouse.x', prac_with_mix_cont_resp_mouse.x)
        prac_with_mix.addData('prac_with_mix_cont_resp_mouse.y', prac_with_mix_cont_resp_mouse.y)
        prac_with_mix.addData('prac_with_mix_cont_resp_mouse.leftButton', prac_with_mix_cont_resp_mouse.leftButton)
        prac_with_mix.addData('prac_with_mix_cont_resp_mouse.midButton', prac_with_mix_cont_resp_mouse.midButton)
        prac_with_mix.addData('prac_with_mix_cont_resp_mouse.rightButton', prac_with_mix_cont_resp_mouse.rightButton)
        prac_with_mix.addData('prac_with_mix_cont_resp_mouse.time', prac_with_mix_cont_resp_mouse.time)
        prac_with_mix.addData('prac_with_mix_cont_resp_mouse.clicked_name', prac_with_mix_cont_resp_mouse.clicked_name)
        # Run 'End Routine' code from prac_with_mix_cont_resp_code
        win.mouseVisible = False
        # the Routine "prac_with_mix_cont_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'prac_with_mix'
    
    
    # --- Prepare to start Routine "prac_end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('prac_end.started', globalClock.getTime())
    prac_end_key_resp.keys = []
    prac_end_key_resp.rt = []
    _prac_end_key_resp_allKeys = []
    # keep track of which components have finished
    prac_endComponents = [prac_end_text, prac_end_key_resp]
    for thisComponent in prac_endComponents:
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
    routineForceEnded = not continueRoutine
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
            theseKeys = prac_end_key_resp.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_end" ---
    for thisComponent in prac_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('prac_end.stopped', globalClock.getTime())
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
    with_mix = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('pics - 4.5.xlsx', selection='0:10'),
        seed=None, name='with_mix')
    thisExp.addLoop(with_mix)  # add the loop to the experiment
    thisWith_mix = with_mix.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisWith_mix.rgb)
    if thisWith_mix != None:
        for paramName in thisWith_mix:
            globals()[paramName] = thisWith_mix[paramName]
    
    for thisWith_mix in with_mix:
        currentLoop = with_mix
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisWith_mix.rgb)
        if thisWith_mix != None:
            for paramName in thisWith_mix:
                globals()[paramName] = thisWith_mix[paramName]
        
        # --- Prepare to start Routine "with_mix_begin" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('with_mix_begin.started', globalClock.getTime())
        with_mix_begin_key_resp.keys = []
        with_mix_begin_key_resp.rt = []
        _with_mix_begin_key_resp_allKeys = []
        # Run 'Begin Routine' code from with_mix_begin_code
        trialID = trialID + 1
        
        thisExp.addData('trialID', trialID)
        
        block_name = 'mix_anticipated'
        # keep track of which components have finished
        with_mix_beginComponents = [with_mix_begin_text, with_mix_begin_key_resp]
        for thisComponent in with_mix_beginComponents:
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
        
        # --- Run Routine "with_mix_begin" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *with_mix_begin_text* updates
            
            # if with_mix_begin_text is starting this frame...
            if with_mix_begin_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_begin_text.frameNStart = frameN  # exact frame index
                with_mix_begin_text.tStart = t  # local t and not account for scr refresh
                with_mix_begin_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_begin_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_begin_text.started')
                # update status
                with_mix_begin_text.status = STARTED
                with_mix_begin_text.setAutoDraw(True)
            
            # if with_mix_begin_text is active this frame...
            if with_mix_begin_text.status == STARTED:
                # update params
                pass
            
            # *with_mix_begin_key_resp* updates
            waitOnFlip = False
            
            # if with_mix_begin_key_resp is starting this frame...
            if with_mix_begin_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_begin_key_resp.frameNStart = frameN  # exact frame index
                with_mix_begin_key_resp.tStart = t  # local t and not account for scr refresh
                with_mix_begin_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_begin_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_begin_key_resp.started')
                # update status
                with_mix_begin_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(with_mix_begin_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(with_mix_begin_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if with_mix_begin_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = with_mix_begin_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _with_mix_begin_key_resp_allKeys.extend(theseKeys)
                if len(_with_mix_begin_key_resp_allKeys):
                    with_mix_begin_key_resp.keys = _with_mix_begin_key_resp_allKeys[-1].name  # just the last key pressed
                    with_mix_begin_key_resp.rt = _with_mix_begin_key_resp_allKeys[-1].rt
                    with_mix_begin_key_resp.duration = _with_mix_begin_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in with_mix_beginComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "with_mix_begin" ---
        for thisComponent in with_mix_beginComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('with_mix_begin.stopped', globalClock.getTime())
        # check responses
        if with_mix_begin_key_resp.keys in ['', [], None]:  # No response was made
            with_mix_begin_key_resp.keys = None
        with_mix.addData('with_mix_begin_key_resp.keys',with_mix_begin_key_resp.keys)
        if with_mix_begin_key_resp.keys != None:  # we had a response
            with_mix.addData('with_mix_begin_key_resp.rt', with_mix_begin_key_resp.rt)
            with_mix.addData('with_mix_begin_key_resp.duration', with_mix_begin_key_resp.duration)
        # the Routine "with_mix_begin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "with_mix_cate_fix" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('with_mix_cate_fix.started', globalClock.getTime())
        # Run 'Begin Routine' code from with_mix_cate_fix_Init_trial
        # 检查当前试次应该运行的例程
        if with_mix.thisTrial['with_mix_type'] == 'with_mix_cate_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        
        # keep track of which components have finished
        with_mix_cate_fixComponents = [with_mix_cate_fix_text, with_mix_cate_fix_left_calm, with_mix_cate_fix_right_fear]
        for thisComponent in with_mix_cate_fixComponents:
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
        
        # --- Run Routine "with_mix_cate_fix" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *with_mix_cate_fix_text* updates
            
            # if with_mix_cate_fix_text is starting this frame...
            if with_mix_cate_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_fix_text.frameNStart = frameN  # exact frame index
                with_mix_cate_fix_text.tStart = t  # local t and not account for scr refresh
                with_mix_cate_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_fix_text.started')
                # update status
                with_mix_cate_fix_text.status = STARTED
                with_mix_cate_fix_text.setAutoDraw(True)
            
            # if with_mix_cate_fix_text is active this frame...
            if with_mix_cate_fix_text.status == STARTED:
                # update params
                pass
            
            # if with_mix_cate_fix_text is stopping this frame...
            if with_mix_cate_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > with_mix_cate_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    with_mix_cate_fix_text.tStop = t  # not accounting for scr refresh
                    with_mix_cate_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'with_mix_cate_fix_text.stopped')
                    # update status
                    with_mix_cate_fix_text.status = FINISHED
                    with_mix_cate_fix_text.setAutoDraw(False)
            
            # *with_mix_cate_fix_left_calm* updates
            
            # if with_mix_cate_fix_left_calm is starting this frame...
            if with_mix_cate_fix_left_calm.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_fix_left_calm.frameNStart = frameN  # exact frame index
                with_mix_cate_fix_left_calm.tStart = t  # local t and not account for scr refresh
                with_mix_cate_fix_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_fix_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_fix_left_calm.started')
                # update status
                with_mix_cate_fix_left_calm.status = STARTED
                with_mix_cate_fix_left_calm.setAutoDraw(True)
            
            # if with_mix_cate_fix_left_calm is active this frame...
            if with_mix_cate_fix_left_calm.status == STARTED:
                # update params
                pass
            
            # if with_mix_cate_fix_left_calm is stopping this frame...
            if with_mix_cate_fix_left_calm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > with_mix_cate_fix_left_calm.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    with_mix_cate_fix_left_calm.tStop = t  # not accounting for scr refresh
                    with_mix_cate_fix_left_calm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'with_mix_cate_fix_left_calm.stopped')
                    # update status
                    with_mix_cate_fix_left_calm.status = FINISHED
                    with_mix_cate_fix_left_calm.setAutoDraw(False)
            
            # *with_mix_cate_fix_right_fear* updates
            
            # if with_mix_cate_fix_right_fear is starting this frame...
            if with_mix_cate_fix_right_fear.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_fix_right_fear.frameNStart = frameN  # exact frame index
                with_mix_cate_fix_right_fear.tStart = t  # local t and not account for scr refresh
                with_mix_cate_fix_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_fix_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_fix_right_fear.started')
                # update status
                with_mix_cate_fix_right_fear.status = STARTED
                with_mix_cate_fix_right_fear.setAutoDraw(True)
            
            # if with_mix_cate_fix_right_fear is active this frame...
            if with_mix_cate_fix_right_fear.status == STARTED:
                # update params
                pass
            
            # if with_mix_cate_fix_right_fear is stopping this frame...
            if with_mix_cate_fix_right_fear.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > with_mix_cate_fix_right_fear.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    with_mix_cate_fix_right_fear.tStop = t  # not accounting for scr refresh
                    with_mix_cate_fix_right_fear.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'with_mix_cate_fix_right_fear.stopped')
                    # update status
                    with_mix_cate_fix_right_fear.status = FINISHED
                    with_mix_cate_fix_right_fear.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in with_mix_cate_fixComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "with_mix_cate_fix" ---
        for thisComponent in with_mix_cate_fixComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('with_mix_cate_fix.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "with_mix_cont_fix" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('with_mix_cont_fix.started', globalClock.getTime())
        # Run 'Begin Routine' code from with_mix_cont_fix_Init_trial
        
        # 检查当前试次应该运行的例程
        if with_mix.thisTrial['with_mix_type'] == 'with_mix_cont_resp':
            continueRoutine = True
        else:
            continueRoutine = False
        # keep track of which components have finished
        with_mix_cont_fixComponents = [with_mix_cont_fix_text, with_mix_cont_fix_left_calm, with_mix_cont_fix_right_fear, with_mix_cont_fix_line]
        for thisComponent in with_mix_cont_fixComponents:
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
        
        # --- Run Routine "with_mix_cont_fix" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *with_mix_cont_fix_text* updates
            
            # if with_mix_cont_fix_text is starting this frame...
            if with_mix_cont_fix_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_fix_text.frameNStart = frameN  # exact frame index
                with_mix_cont_fix_text.tStart = t  # local t and not account for scr refresh
                with_mix_cont_fix_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_fix_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cont_fix_text.started')
                # update status
                with_mix_cont_fix_text.status = STARTED
                with_mix_cont_fix_text.setAutoDraw(True)
            
            # if with_mix_cont_fix_text is active this frame...
            if with_mix_cont_fix_text.status == STARTED:
                # update params
                pass
            
            # if with_mix_cont_fix_text is stopping this frame...
            if with_mix_cont_fix_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > with_mix_cont_fix_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    with_mix_cont_fix_text.tStop = t  # not accounting for scr refresh
                    with_mix_cont_fix_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'with_mix_cont_fix_text.stopped')
                    # update status
                    with_mix_cont_fix_text.status = FINISHED
                    with_mix_cont_fix_text.setAutoDraw(False)
            
            # *with_mix_cont_fix_left_calm* updates
            
            # if with_mix_cont_fix_left_calm is starting this frame...
            if with_mix_cont_fix_left_calm.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_fix_left_calm.frameNStart = frameN  # exact frame index
                with_mix_cont_fix_left_calm.tStart = t  # local t and not account for scr refresh
                with_mix_cont_fix_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_fix_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cont_fix_left_calm.started')
                # update status
                with_mix_cont_fix_left_calm.status = STARTED
                with_mix_cont_fix_left_calm.setAutoDraw(True)
            
            # if with_mix_cont_fix_left_calm is active this frame...
            if with_mix_cont_fix_left_calm.status == STARTED:
                # update params
                pass
            
            # if with_mix_cont_fix_left_calm is stopping this frame...
            if with_mix_cont_fix_left_calm.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > with_mix_cont_fix_left_calm.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    with_mix_cont_fix_left_calm.tStop = t  # not accounting for scr refresh
                    with_mix_cont_fix_left_calm.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'with_mix_cont_fix_left_calm.stopped')
                    # update status
                    with_mix_cont_fix_left_calm.status = FINISHED
                    with_mix_cont_fix_left_calm.setAutoDraw(False)
            
            # *with_mix_cont_fix_right_fear* updates
            
            # if with_mix_cont_fix_right_fear is starting this frame...
            if with_mix_cont_fix_right_fear.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_fix_right_fear.frameNStart = frameN  # exact frame index
                with_mix_cont_fix_right_fear.tStart = t  # local t and not account for scr refresh
                with_mix_cont_fix_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_fix_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cont_fix_right_fear.started')
                # update status
                with_mix_cont_fix_right_fear.status = STARTED
                with_mix_cont_fix_right_fear.setAutoDraw(True)
            
            # if with_mix_cont_fix_right_fear is active this frame...
            if with_mix_cont_fix_right_fear.status == STARTED:
                # update params
                pass
            
            # if with_mix_cont_fix_right_fear is stopping this frame...
            if with_mix_cont_fix_right_fear.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > with_mix_cont_fix_right_fear.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    with_mix_cont_fix_right_fear.tStop = t  # not accounting for scr refresh
                    with_mix_cont_fix_right_fear.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'with_mix_cont_fix_right_fear.stopped')
                    # update status
                    with_mix_cont_fix_right_fear.status = FINISHED
                    with_mix_cont_fix_right_fear.setAutoDraw(False)
            
            # *with_mix_cont_fix_line* updates
            
            # if with_mix_cont_fix_line is starting this frame...
            if with_mix_cont_fix_line.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_fix_line.frameNStart = frameN  # exact frame index
                with_mix_cont_fix_line.tStart = t  # local t and not account for scr refresh
                with_mix_cont_fix_line.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_fix_line, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cont_fix_line.started')
                # update status
                with_mix_cont_fix_line.status = STARTED
                with_mix_cont_fix_line.setAutoDraw(True)
            
            # if with_mix_cont_fix_line is active this frame...
            if with_mix_cont_fix_line.status == STARTED:
                # update params
                pass
            
            # if with_mix_cont_fix_line is stopping this frame...
            if with_mix_cont_fix_line.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > with_mix_cont_fix_line.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    with_mix_cont_fix_line.tStop = t  # not accounting for scr refresh
                    with_mix_cont_fix_line.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'with_mix_cont_fix_line.stopped')
                    # update status
                    with_mix_cont_fix_line.status = FINISHED
                    with_mix_cont_fix_line.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in with_mix_cont_fixComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "with_mix_cont_fix" ---
        for thisComponent in with_mix_cont_fixComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('with_mix_cont_fix.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "with_mix_face" ---
        continueRoutine = True
        # update component parameters for each repeat
        with_mix_face_image.setImage(mix)
        # keep track of which components have finished
        with_mix_faceComponents = [with_mix_face_image]
        for thisComponent in with_mix_faceComponents:
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
        
        # --- Run Routine "with_mix_face" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *with_mix_face_image* updates
            
            # if with_mix_face_image is starting this frame...
            if with_mix_face_image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_face_image.frameNStart = frameN  # exact frame index
                with_mix_face_image.tStart = t  # local t and not account for scr refresh
                with_mix_face_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_face_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_face_image.started')
                # update status
                with_mix_face_image.status = STARTED
                with_mix_face_image.setAutoDraw(True)
            
            # if with_mix_face_image is active this frame...
            if with_mix_face_image.status == STARTED:
                # update params
                pass
            
            # if with_mix_face_image is stopping this frame...
            if with_mix_face_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > with_mix_face_image.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    with_mix_face_image.tStop = t  # not accounting for scr refresh
                    with_mix_face_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'with_mix_face_image.stopped')
                    # update status
                    with_mix_face_image.status = FINISHED
                    with_mix_face_image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in with_mix_faceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "with_mix_face" ---
        for thisComponent in with_mix_faceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "with_mix_cate_resp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('with_mix_cate_resp.started', globalClock.getTime())
        # setup some python lists for storing info about the with_mix_cate_resp_mouse
        with_mix_cate_resp_mouse.x = []
        with_mix_cate_resp_mouse.y = []
        with_mix_cate_resp_mouse.leftButton = []
        with_mix_cate_resp_mouse.midButton = []
        with_mix_cate_resp_mouse.rightButton = []
        with_mix_cate_resp_mouse.time = []
        with_mix_cate_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        with_mix_cate_resp_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from with_mix_cate_resp_code
        # 设置Rectangle的初始状态为不可见
        with_mix_cate_resp_rect_left.opacity = 0
        with_mix_cate_resp_rect_right.opacity = 0
        
        with_mix_cate_resp_mouse.setPos((0, 0))
        
        # 检查当前试次应该运行的例程
        if with_mix.thisTrial['with_mix_type'] == 'with_mix_cate_resp':
            continueRoutine = True
        else:
            continueRoutine = False
            
        cate_resp = 'null'
        # keep track of which components have finished
        with_mix_cate_respComponents = [with_mix_cate_resp_prompt, with_mix_cate_resp_rect_left, with_mix_cate_resp_rect_right, with_mix_cate_resp_left_calm, with_mix_cate_resp_right_fear, with_mix_cate_resp_mouse]
        for thisComponent in with_mix_cate_respComponents:
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
        
        # --- Run Routine "with_mix_cate_resp" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *with_mix_cate_resp_prompt* updates
            
            # if with_mix_cate_resp_prompt is starting this frame...
            if with_mix_cate_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_resp_prompt.frameNStart = frameN  # exact frame index
                with_mix_cate_resp_prompt.tStart = t  # local t and not account for scr refresh
                with_mix_cate_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_resp_prompt.started')
                # update status
                with_mix_cate_resp_prompt.status = STARTED
                with_mix_cate_resp_prompt.setAutoDraw(True)
            
            # if with_mix_cate_resp_prompt is active this frame...
            if with_mix_cate_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *with_mix_cate_resp_rect_left* updates
            
            # if with_mix_cate_resp_rect_left is starting this frame...
            if with_mix_cate_resp_rect_left.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_resp_rect_left.frameNStart = frameN  # exact frame index
                with_mix_cate_resp_rect_left.tStart = t  # local t and not account for scr refresh
                with_mix_cate_resp_rect_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_resp_rect_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_resp_rect_left.started')
                # update status
                with_mix_cate_resp_rect_left.status = STARTED
                with_mix_cate_resp_rect_left.setAutoDraw(True)
            
            # if with_mix_cate_resp_rect_left is active this frame...
            if with_mix_cate_resp_rect_left.status == STARTED:
                # update params
                pass
            
            # *with_mix_cate_resp_rect_right* updates
            
            # if with_mix_cate_resp_rect_right is starting this frame...
            if with_mix_cate_resp_rect_right.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_resp_rect_right.frameNStart = frameN  # exact frame index
                with_mix_cate_resp_rect_right.tStart = t  # local t and not account for scr refresh
                with_mix_cate_resp_rect_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_resp_rect_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_resp_rect_right.started')
                # update status
                with_mix_cate_resp_rect_right.status = STARTED
                with_mix_cate_resp_rect_right.setAutoDraw(True)
            
            # if with_mix_cate_resp_rect_right is active this frame...
            if with_mix_cate_resp_rect_right.status == STARTED:
                # update params
                pass
            
            # *with_mix_cate_resp_left_calm* updates
            
            # if with_mix_cate_resp_left_calm is starting this frame...
            if with_mix_cate_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_resp_left_calm.frameNStart = frameN  # exact frame index
                with_mix_cate_resp_left_calm.tStart = t  # local t and not account for scr refresh
                with_mix_cate_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_resp_left_calm.started')
                # update status
                with_mix_cate_resp_left_calm.status = STARTED
                with_mix_cate_resp_left_calm.setAutoDraw(True)
            
            # if with_mix_cate_resp_left_calm is active this frame...
            if with_mix_cate_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *with_mix_cate_resp_right_fear* updates
            
            # if with_mix_cate_resp_right_fear is starting this frame...
            if with_mix_cate_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_resp_right_fear.frameNStart = frameN  # exact frame index
                with_mix_cate_resp_right_fear.tStart = t  # local t and not account for scr refresh
                with_mix_cate_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_resp_right_fear.started')
                # update status
                with_mix_cate_resp_right_fear.status = STARTED
                with_mix_cate_resp_right_fear.setAutoDraw(True)
            
            # if with_mix_cate_resp_right_fear is active this frame...
            if with_mix_cate_resp_right_fear.status == STARTED:
                # update params
                pass
            # *with_mix_cate_resp_mouse* updates
            
            # if with_mix_cate_resp_mouse is starting this frame...
            if with_mix_cate_resp_mouse.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cate_resp_mouse.frameNStart = frameN  # exact frame index
                with_mix_cate_resp_mouse.tStart = t  # local t and not account for scr refresh
                with_mix_cate_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cate_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cate_resp_mouse.started')
                # update status
                with_mix_cate_resp_mouse.status = STARTED
                prevButtonState = with_mix_cate_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if with_mix_cate_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = with_mix_cate_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([with_mix_cate_resp_left_calm, with_mix_cate_resp_right_fear], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(with_mix_cate_resp_mouse):
                                gotValidClick = True
                                with_mix_cate_resp_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = with_mix_cate_resp_mouse.getPos()
                            with_mix_cate_resp_mouse.x.append(x)
                            with_mix_cate_resp_mouse.y.append(y)
                            buttons = with_mix_cate_resp_mouse.getPressed()
                            with_mix_cate_resp_mouse.leftButton.append(buttons[0])
                            with_mix_cate_resp_mouse.midButton.append(buttons[1])
                            with_mix_cate_resp_mouse.rightButton.append(buttons[2])
                            with_mix_cate_resp_mouse.time.append(with_mix_cate_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from with_mix_cate_resp_code
            # 检测鼠标是否悬停在左边单词上方
            if with_mix_cate_resp_left_calm.contains(with_mix_cate_resp_mouse):
                with_mix_cate_resp_rect_left.opacity = 1  # 设置左边Rectangle可见
            else:
                with_mix_cate_resp_rect_left.opacity = 0  # 设置左边Rectangle不可见
            
            # 检测鼠标是否悬停在右边单词上方
            if with_mix_cate_resp_right_fear.contains(with_mix_cate_resp_mouse):
                with_mix_cate_resp_rect_right.opacity = 1  # 设置右边Rectangle可见
            else:
                with_mix_cate_resp_rect_right.opacity = 0  # 设置右边Rectangle不可见
            
            # 检测鼠标点击事件
            if with_mix_cate_resp_mouse.getPressed()[0]:
                # 检测鼠标点击是否在左边单词上方
                if with_mix_cate_resp_left_calm.contains(with_mix_cate_resp_mouse):
                    cate_resp = 0  # 记录被试选择左边单词
                # 检测鼠标点击是否在右边单词上方
                elif with_mix_cate_resp_right_fear.contains(with_mix_cate_resp_mouse):
                    cate_resp = 1  # 记录被试选择右边单词
                else:
                    cate_resp = 'None'
            else:
                cate_resp = 'None'
                
                
            win.mouseVisible = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in with_mix_cate_respComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "with_mix_cate_resp" ---
        for thisComponent in with_mix_cate_respComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('with_mix_cate_resp.stopped', globalClock.getTime())
        # store data for with_mix (TrialHandler)
        with_mix.addData('with_mix_cate_resp_mouse.x', with_mix_cate_resp_mouse.x)
        with_mix.addData('with_mix_cate_resp_mouse.y', with_mix_cate_resp_mouse.y)
        with_mix.addData('with_mix_cate_resp_mouse.leftButton', with_mix_cate_resp_mouse.leftButton)
        with_mix.addData('with_mix_cate_resp_mouse.midButton', with_mix_cate_resp_mouse.midButton)
        with_mix.addData('with_mix_cate_resp_mouse.rightButton', with_mix_cate_resp_mouse.rightButton)
        with_mix.addData('with_mix_cate_resp_mouse.time', with_mix_cate_resp_mouse.time)
        with_mix.addData('with_mix_cate_resp_mouse.clicked_name', with_mix_cate_resp_mouse.clicked_name)
        # Run 'End Routine' code from with_mix_cate_resp_code
        #trialID = trialID + 1
        
        thisExp.addData('Anticipation', block_name)
        #thisExp.addData('trialID', trialID)
        thisExp.addData('face_stim', mix)
        thisExp.addData('cate_resp', cate_resp)
        thisExp.addData('Thinking', thinking)
        
        win.mouseVisible = False
        
        # the Routine "with_mix_cate_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "with_mix_cont_resp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('with_mix_cont_resp.started', globalClock.getTime())
        with_mix_cont_resp_slider.reset()
        # setup some python lists for storing info about the with_mix_cont_resp_mouse
        with_mix_cont_resp_mouse.x = []
        with_mix_cont_resp_mouse.y = []
        with_mix_cont_resp_mouse.leftButton = []
        with_mix_cont_resp_mouse.midButton = []
        with_mix_cont_resp_mouse.rightButton = []
        with_mix_cont_resp_mouse.time = []
        with_mix_cont_resp_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        with_mix_cont_resp_mouse.mouseClock.reset()
        # Run 'Begin Routine' code from with_mix_cont_resp_code
        # Change the following parameters according to your own exp
        slider_width = 0.5
        slider_height = .05
        slider_orientation = 0
        #Failing to set ticks or labels in code
        slider_ticks=[0, 1]
        
        slider_granularity = .1
        slider_decimals = 2
        with_mix_cont_resp_slider.decimals = 2
        # 获取Slider组件的标记（marker）
        marker = with_mix_cont_resp_slider.marker
        # 将标记的形状更改为一个特定尺寸的方块
        marker.shape = 'square'
        marker.size = (0.01, 0.05)  # 设置方块的尺寸
        # Begin Routine
        
        # record the current position of the mouse
        mouseRec=with_mix_cont_resp_mouse.getPos()
        
        with_mix_cont_resp_mouse.setPos((0, 0))
        
        # 检查当前试次应该运行的例程
        if with_mix.thisTrial['with_mix_type'] == 'with_mix_cont_resp':
            continueRoutine = True
        else:
            continueRoutine = False
            
        cont_resp = 'null'
        # keep track of which components have finished
        with_mix_cont_respComponents = [with_mix_cont_resp_prompt, with_mix_cont_resp_slider_shape, with_mix_cont_resp_slider, with_mix_cont_resp_left_calm, with_mix_cont_resp_right_fear, with_mix_cont_resp_mouse]
        for thisComponent in with_mix_cont_respComponents:
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
        
        # --- Run Routine "with_mix_cont_resp" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *with_mix_cont_resp_prompt* updates
            
            # if with_mix_cont_resp_prompt is starting this frame...
            if with_mix_cont_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_resp_prompt.frameNStart = frameN  # exact frame index
                with_mix_cont_resp_prompt.tStart = t  # local t and not account for scr refresh
                with_mix_cont_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cont_resp_prompt.started')
                # update status
                with_mix_cont_resp_prompt.status = STARTED
                with_mix_cont_resp_prompt.setAutoDraw(True)
            
            # if with_mix_cont_resp_prompt is active this frame...
            if with_mix_cont_resp_prompt.status == STARTED:
                # update params
                pass
            
            # *with_mix_cont_resp_slider_shape* updates
            
            # if with_mix_cont_resp_slider_shape is starting this frame...
            if with_mix_cont_resp_slider_shape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_resp_slider_shape.frameNStart = frameN  # exact frame index
                with_mix_cont_resp_slider_shape.tStart = t  # local t and not account for scr refresh
                with_mix_cont_resp_slider_shape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_resp_slider_shape, 'tStartRefresh')  # time at next scr refresh
                # update status
                with_mix_cont_resp_slider_shape.status = STARTED
                with_mix_cont_resp_slider_shape.setAutoDraw(True)
            
            # if with_mix_cont_resp_slider_shape is active this frame...
            if with_mix_cont_resp_slider_shape.status == STARTED:
                # update params
                pass
            
            # *with_mix_cont_resp_slider* updates
            
            # if with_mix_cont_resp_slider is starting this frame...
            if with_mix_cont_resp_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_resp_slider.frameNStart = frameN  # exact frame index
                with_mix_cont_resp_slider.tStart = t  # local t and not account for scr refresh
                with_mix_cont_resp_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_resp_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cont_resp_slider.started')
                # update status
                with_mix_cont_resp_slider.status = STARTED
                with_mix_cont_resp_slider.setAutoDraw(True)
            
            # if with_mix_cont_resp_slider is active this frame...
            if with_mix_cont_resp_slider.status == STARTED:
                # update params
                pass
            
            # *with_mix_cont_resp_left_calm* updates
            
            # if with_mix_cont_resp_left_calm is starting this frame...
            if with_mix_cont_resp_left_calm.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_resp_left_calm.frameNStart = frameN  # exact frame index
                with_mix_cont_resp_left_calm.tStart = t  # local t and not account for scr refresh
                with_mix_cont_resp_left_calm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_resp_left_calm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cont_resp_left_calm.started')
                # update status
                with_mix_cont_resp_left_calm.status = STARTED
                with_mix_cont_resp_left_calm.setAutoDraw(True)
            
            # if with_mix_cont_resp_left_calm is active this frame...
            if with_mix_cont_resp_left_calm.status == STARTED:
                # update params
                pass
            
            # *with_mix_cont_resp_right_fear* updates
            
            # if with_mix_cont_resp_right_fear is starting this frame...
            if with_mix_cont_resp_right_fear.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_resp_right_fear.frameNStart = frameN  # exact frame index
                with_mix_cont_resp_right_fear.tStart = t  # local t and not account for scr refresh
                with_mix_cont_resp_right_fear.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_resp_right_fear, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_mix_cont_resp_right_fear.started')
                # update status
                with_mix_cont_resp_right_fear.status = STARTED
                with_mix_cont_resp_right_fear.setAutoDraw(True)
            
            # if with_mix_cont_resp_right_fear is active this frame...
            if with_mix_cont_resp_right_fear.status == STARTED:
                # update params
                pass
            # *with_mix_cont_resp_mouse* updates
            
            # if with_mix_cont_resp_mouse is starting this frame...
            if with_mix_cont_resp_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_mix_cont_resp_mouse.frameNStart = frameN  # exact frame index
                with_mix_cont_resp_mouse.tStart = t  # local t and not account for scr refresh
                with_mix_cont_resp_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_mix_cont_resp_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('with_mix_cont_resp_mouse.started', t)
                # update status
                with_mix_cont_resp_mouse.status = STARTED
                prevButtonState = with_mix_cont_resp_mouse.getPressed()  # if button is down already this ISN'T a new click
            if with_mix_cont_resp_mouse.status == STARTED:  # only update if started and not finished!
                buttons = with_mix_cont_resp_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(with_mix_cont_resp_slider_shape, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(with_mix_cont_resp_mouse):
                                gotValidClick = True
                                with_mix_cont_resp_mouse.clicked_name.append(obj.name)
                        if gotValidClick:
                            x, y = with_mix_cont_resp_mouse.getPos()
                            with_mix_cont_resp_mouse.x.append(x)
                            with_mix_cont_resp_mouse.y.append(y)
                            buttons = with_mix_cont_resp_mouse.getPressed()
                            with_mix_cont_resp_mouse.leftButton.append(buttons[0])
                            with_mix_cont_resp_mouse.midButton.append(buttons[1])
                            with_mix_cont_resp_mouse.rightButton.append(buttons[2])
                            with_mix_cont_resp_mouse.time.append(with_mix_cont_resp_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            # Run 'Each Frame' code from with_mix_cont_resp_code
            
            # Create a slider_shape in this routine with the same size as the slider
            # Check if mouse is moving within slider_shape, if so change marker position accordingly
            if with_mix_cont_resp_slider_shape.contains(with_mix_cont_resp_mouse):
                mouseRec = with_mix_cont_resp_mouse.getPos()
                with_mix_cont_resp_slider.markerPos = mouseRec[slider_orientation]/slider_width*(slider_ticks[-1]-slider_ticks[0])+(slider_ticks[0]+slider_ticks[-1])/2
                # (new)Record marker position only on left mouse click
                if with_mix_cont_resp_mouse.getPressed()[0]:
                    cont_resp = round(with_mix_cont_resp_slider.markerPos, slider_decimals)
                else:
                    cont_resp = 'None'
            else:
                cont_resp = 'None'
            
            win.mouseVisible = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in with_mix_cont_respComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "with_mix_cont_resp" ---
        for thisComponent in with_mix_cont_respComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('with_mix_cont_resp.stopped', globalClock.getTime())
        with_mix.addData('with_mix_cont_resp_slider.response', with_mix_cont_resp_slider.getRating())
        with_mix.addData('with_mix_cont_resp_slider.rt', with_mix_cont_resp_slider.getRT())
        # store data for with_mix (TrialHandler)
        with_mix.addData('with_mix_cont_resp_mouse.x', with_mix_cont_resp_mouse.x)
        with_mix.addData('with_mix_cont_resp_mouse.y', with_mix_cont_resp_mouse.y)
        with_mix.addData('with_mix_cont_resp_mouse.leftButton', with_mix_cont_resp_mouse.leftButton)
        with_mix.addData('with_mix_cont_resp_mouse.midButton', with_mix_cont_resp_mouse.midButton)
        with_mix.addData('with_mix_cont_resp_mouse.rightButton', with_mix_cont_resp_mouse.rightButton)
        with_mix.addData('with_mix_cont_resp_mouse.time', with_mix_cont_resp_mouse.time)
        with_mix.addData('with_mix_cont_resp_mouse.clicked_name', with_mix_cont_resp_mouse.clicked_name)
        # Run 'End Routine' code from with_mix_cont_resp_code
        #trialID = trialID + 1
        
        thisExp.addData('Anticipation', block_name)
        #thisExp.addData('trialID', trialID)
        thisExp.addData('face_stim', mix)
        
        #thisExp.addData('All Responses',slider_data)
        thisExp.addData('cont_resp',cont_resp)
        thisExp.addData('Thinking', thinking)
        
        print('cont_resp',cont_resp)
        
        win.mouseVisible = False
        
        # the Routine "with_mix_cont_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "with_mix_brea" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('with_mix_brea.started', globalClock.getTime())
        with_brea_key_resp.keys = []
        with_brea_key_resp.rt = []
        _with_brea_key_resp_allKeys = []
        # Run 'Begin Routine' code from with_mix_brea_code
        if trialID == 50:
            continueRoutine = True
        else:
            continueRoutine = False
        # keep track of which components have finished
        with_mix_breaComponents = [with_brea_text, with_brea_key_resp]
        for thisComponent in with_mix_breaComponents:
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
        
        # --- Run Routine "with_mix_brea" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *with_brea_text* updates
            
            # if with_brea_text is starting this frame...
            if with_brea_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_brea_text.frameNStart = frameN  # exact frame index
                with_brea_text.tStart = t  # local t and not account for scr refresh
                with_brea_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_brea_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'with_brea_text.started')
                # update status
                with_brea_text.status = STARTED
                with_brea_text.setAutoDraw(True)
            
            # if with_brea_text is active this frame...
            if with_brea_text.status == STARTED:
                # update params
                pass
            
            # *with_brea_key_resp* updates
            waitOnFlip = False
            
            # if with_brea_key_resp is starting this frame...
            if with_brea_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                with_brea_key_resp.frameNStart = frameN  # exact frame index
                with_brea_key_resp.tStart = t  # local t and not account for scr refresh
                with_brea_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(with_brea_key_resp, 'tStartRefresh')  # time at next scr refresh
                # update status
                with_brea_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(with_brea_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(with_brea_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if with_brea_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = with_brea_key_resp.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _with_brea_key_resp_allKeys.extend(theseKeys)
                if len(_with_brea_key_resp_allKeys):
                    with_brea_key_resp.keys = _with_brea_key_resp_allKeys[-1].name  # just the last key pressed
                    with_brea_key_resp.rt = _with_brea_key_resp_allKeys[-1].rt
                    with_brea_key_resp.duration = _with_brea_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in with_mix_breaComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "with_mix_brea" ---
        for thisComponent in with_mix_breaComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('with_mix_brea.stopped', globalClock.getTime())
        # check responses
        if with_brea_key_resp.keys in ['', [], None]:  # No response was made
            with_brea_key_resp.keys = None
        with_mix.addData('with_brea_key_resp.keys',with_brea_key_resp.keys)
        if with_brea_key_resp.keys != None:  # we had a response
            with_mix.addData('with_brea_key_resp.rt', with_brea_key_resp.rt)
            with_mix.addData('with_brea_key_resp.duration', with_brea_key_resp.duration)
        # the Routine "with_mix_brea" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'with_mix'
    
    
    # --- Prepare to start Routine "formal_wo_end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('formal_wo_end.started', globalClock.getTime())
    formal_wo_end_key_resp.keys = []
    formal_wo_end_key_resp.rt = []
    _formal_wo_end_key_resp_allKeys = []
    # keep track of which components have finished
    formal_wo_endComponents = [formal_wo_end_text, formal_wo_end_key_resp]
    for thisComponent in formal_wo_endComponents:
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
    routineForceEnded = not continueRoutine
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
            theseKeys = formal_wo_end_key_resp.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in formal_wo_endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "formal_wo_end" ---
    for thisComponent in formal_wo_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('formal_wo_end.stopped', globalClock.getTime())
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
    
    # --- Prepare to start Routine "intro_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_2.started', globalClock.getTime())
    intro_2_key_resp.keys = []
    intro_2_key_resp.rt = []
    _intro_2_key_resp_allKeys = []
    # keep track of which components have finished
    intro_2Components = [intro_2_text, intro_2_key_resp, intro_2_image]
    for thisComponent in intro_2Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_2" ---
    for thisComponent in intro_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_2.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_3.started', globalClock.getTime())
    intro_3_key_resp.keys = []
    intro_3_key_resp.rt = []
    _intro_3_key_resp_allKeys = []
    # keep track of which components have finished
    intro_3Components = [intro_3_text, intro_3_key_resp, intro_3_left_calm, intro_3_right_fear]
    for thisComponent in intro_3Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_3" ---
    for thisComponent in intro_3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_3.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_4.started', globalClock.getTime())
    intro_4_key_resp.keys = []
    intro_4_key_resp.rt = []
    _intro_4_key_resp_allKeys = []
    # keep track of which components have finished
    intro_4Components = [intro_4_text, intro_4_key_resp, intro_4_rect_left, intro_4_left_calm, intro_4_right_fear]
    for thisComponent in intro_4Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_4Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_4" ---
    for thisComponent in intro_4Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_4.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_5.started', globalClock.getTime())
    intro_5_key_resp.keys = []
    intro_5_key_resp.rt = []
    _intro_5_key_resp_allKeys = []
    # keep track of which components have finished
    intro_5Components = [intro_5_text, intro_5_key_resp, intro_5_left_calm, intro_5_right_fear, intro_5_cont_fix_line]
    for thisComponent in intro_5Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_5Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_5" ---
    for thisComponent in intro_5Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_5.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_6.started', globalClock.getTime())
    intro_6_key_resp.keys = []
    intro_6_key_resp.rt = []
    _intro_6_key_resp_allKeys = []
    # keep track of which components have finished
    intro_6Components = [intro_6_text, intro_6_key_resp, intro_6_left_calm, intro_6_right_fear, intro_6_marker, intro_6_cont_fix_line]
    for thisComponent in intro_6Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_6Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_6" ---
    for thisComponent in intro_6Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_6.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_7.started', globalClock.getTime())
    intro_7_key_resp.keys = []
    intro_7_key_resp.rt = []
    _intro_7_key_resp_allKeys = []
    # keep track of which components have finished
    intro_7Components = [intro_7_text, intro_7_key_resp]
    for thisComponent in intro_7Components:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_7Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_7" ---
    for thisComponent in intro_7Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_7.stopped', globalClock.getTime())
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
    prac_wo_mix = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('pics - 4.5.xlsx', selection='0:6'),
        seed=None, name='prac_wo_mix')
    thisExp.addLoop(prac_wo_mix)  # add the loop to the experiment
    thisPrac_wo_mix = prac_wo_mix.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPrac_wo_mix.rgb)
    if thisPrac_wo_mix != None:
        for paramName in thisPrac_wo_mix:
            globals()[paramName] = thisPrac_wo_mix[paramName]
    
    for thisPrac_wo_mix in prac_wo_mix:
        currentLoop = prac_wo_mix
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_wo_mix.rgb)
        if thisPrac_wo_mix != None:
            for paramName in thisPrac_wo_mix:
                globals()[paramName] = thisPrac_wo_mix[paramName]
        
        # --- Prepare to start Routine "prac_wo_mix_begin" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_wo_mix_begin.started', globalClock.getTime())
        prac_wo_mix_begin_key_resp.keys = []
        prac_wo_mix_begin_key_resp.rt = []
        _prac_wo_mix_begin_key_resp_allKeys = []
        # keep track of which components have finished
        prac_wo_mix_beginComponents = [prac_wo_mix_begin_text, prac_wo_mix_begin_key_resp]
        for thisComponent in prac_wo_mix_beginComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_beginComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_begin" ---
        for thisComponent in prac_wo_mix_beginComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_wo_mix_begin.stopped', globalClock.getTime())
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
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_wo_mix_fix.started', globalClock.getTime())
        # keep track of which components have finished
        prac_wo_mix_fixComponents = [prac_wo_mix_fix_text]
        for thisComponent in prac_wo_mix_fixComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_fixComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_fix" ---
        for thisComponent in prac_wo_mix_fixComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_wo_mix_fix.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "prac_wo_mix_face" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_wo_mix_face.started', globalClock.getTime())
        prac_wo_mix_face_image.setImage(mixprac)
        # keep track of which components have finished
        prac_wo_mix_faceComponents = [prac_wo_mix_face_image]
        for thisComponent in prac_wo_mix_faceComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_faceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_face" ---
        for thisComponent in prac_wo_mix_faceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_wo_mix_face.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "prac_wo_mix_cate_resp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_wo_mix_cate_resp.started', globalClock.getTime())
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
        
        
        # keep track of which components have finished
        prac_wo_mix_cate_respComponents = [prac_wo_mix_cate_resp_prompt, prac_wo_mix_cate_resp_rect_left, prac_wo_mix_cate_resp_rect_right, prac_wo_mix_cate_resp_left_calm, prac_wo_mix_cate_resp_right_fear, prac_wo_mix_cate_resp_mouse]
        for thisComponent in prac_wo_mix_cate_respComponents:
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
        routineForceEnded = not continueRoutine
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
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_cate_respComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_cate_resp" ---
        for thisComponent in prac_wo_mix_cate_respComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_wo_mix_cate_resp.stopped', globalClock.getTime())
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
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prac_wo_mix_cont_resp.started', globalClock.getTime())
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
        
        # keep track of which components have finished
        prac_wo_mix_cont_respComponents = [prac_wo_mix_cont_resp_prompt, prac_wo_mix_cont_resp_slider_shape, prac_wo_mix_cont_resp_slider, prac_wo_mix_cont_resp_left_calm, prac_wo_mix_cont_resp_right_fear, prac_wo_mix_cont_resp_mouse]
        for thisComponent in prac_wo_mix_cont_respComponents:
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
        routineForceEnded = not continueRoutine
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
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_wo_mix_cont_respComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_wo_mix_cont_resp" ---
        for thisComponent in prac_wo_mix_cont_respComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prac_wo_mix_cont_resp.stopped', globalClock.getTime())
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
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('prac_wo_end.started', globalClock.getTime())
    prac_wo_end_key_resp.keys = []
    prac_wo_end_key_resp.rt = []
    _prac_wo_end_key_resp_allKeys = []
    # keep track of which components have finished
    prac_wo_endComponents = [prac_wo_end_text, prac_wo_end_key_resp]
    for thisComponent in prac_wo_endComponents:
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
    routineForceEnded = not continueRoutine
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
            theseKeys = prac_wo_end_key_resp.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_wo_endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_wo_end" ---
    for thisComponent in prac_wo_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('prac_wo_end.stopped', globalClock.getTime())
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
    wo_mix = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('pics - 4.5.xlsx', selection='0:10'),
        seed=None, name='wo_mix')
    thisExp.addLoop(wo_mix)  # add the loop to the experiment
    thisWo_mix = wo_mix.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisWo_mix.rgb)
    if thisWo_mix != None:
        for paramName in thisWo_mix:
            globals()[paramName] = thisWo_mix[paramName]
    
    for thisWo_mix in wo_mix:
        currentLoop = wo_mix
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisWo_mix.rgb)
        if thisWo_mix != None:
            for paramName in thisWo_mix:
                globals()[paramName] = thisWo_mix[paramName]
        
        # --- Prepare to start Routine "wo_mix_begin" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('wo_mix_begin.started', globalClock.getTime())
        wo_mix_begin_key_resp.keys = []
        wo_mix_begin_key_resp.rt = []
        _wo_mix_begin_key_resp_allKeys = []
        # keep track of which components have finished
        wo_mix_beginComponents = [wo_mix_begin_text, wo_mix_begin_key_resp]
        for thisComponent in wo_mix_beginComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_beginComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_begin" ---
        for thisComponent in wo_mix_beginComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('wo_mix_begin.stopped', globalClock.getTime())
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
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('wo_mix_fix.started', globalClock.getTime())
        # Run 'Begin Routine' code from wo_mix_fix_Init_trial
        block_name = 'mix_unanticipated'
        
        # keep track of which components have finished
        wo_mix_fixComponents = [wo_mix_fix_text]
        for thisComponent in wo_mix_fixComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_fixComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_fix" ---
        for thisComponent in wo_mix_fixComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('wo_mix_fix.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "wo_mix_face" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('wo_mix_face.started', globalClock.getTime())
        wo_mix_face_image.setImage(mix)
        # keep track of which components have finished
        wo_mix_faceComponents = [wo_mix_face_image]
        for thisComponent in wo_mix_faceComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_faceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_face" ---
        for thisComponent in wo_mix_faceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('wo_mix_face.stopped', globalClock.getTime())
        # Run 'End Routine' code from wo_mix_face_code
        trialID = trialID + 1
        
        thisExp.addData('trialID', trialID)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "wo_mix_cate_resp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('wo_mix_cate_resp.started', globalClock.getTime())
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
        
        # keep track of which components have finished
        wo_mix_cate_respComponents = [wo_mix_cate_resp_prompt, wo_mix_cate_resp_rect_left, wo_mix_cate_resp_rect_right, wo_mix_cate_resp_left_calm, wo_mix_cate_resp_right_fear, wo_mix_cate_resp_mouse]
        for thisComponent in wo_mix_cate_respComponents:
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
        routineForceEnded = not continueRoutine
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
                # 检测鼠标点击是否在右边单词上方
                elif wo_mix_cate_resp_right_fear.contains(wo_mix_cate_resp_mouse):
                    cate_resp = 1  # 记录被试选择右边单词
                else:
                    cate_resp = 'None'
            else:
                cate_resp = 'None'
                
                
            win.mouseVisible = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_cate_respComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_cate_resp" ---
        for thisComponent in wo_mix_cate_respComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('wo_mix_cate_resp.stopped', globalClock.getTime())
        # store data for wo_mix (TrialHandler)
        wo_mix.addData('wo_mix_cate_resp_mouse.x', wo_mix_cate_resp_mouse.x)
        wo_mix.addData('wo_mix_cate_resp_mouse.y', wo_mix_cate_resp_mouse.y)
        wo_mix.addData('wo_mix_cate_resp_mouse.leftButton', wo_mix_cate_resp_mouse.leftButton)
        wo_mix.addData('wo_mix_cate_resp_mouse.midButton', wo_mix_cate_resp_mouse.midButton)
        wo_mix.addData('wo_mix_cate_resp_mouse.rightButton', wo_mix_cate_resp_mouse.rightButton)
        wo_mix.addData('wo_mix_cate_resp_mouse.time', wo_mix_cate_resp_mouse.time)
        wo_mix.addData('wo_mix_cate_resp_mouse.clicked_name', wo_mix_cate_resp_mouse.clicked_name)
        # Run 'End Routine' code from wo_mix_cate_resp_code
        
        thisExp.addData('Anticipation', block_name)
        thisExp.addData('face_stim', mix)
        thisExp.addData('cate_resp', cate_resp)
        
        #thisExp.addData('block_mix_type', mix_type)
        thisExp.addData('Thinking', thinking)
        win.mouseVisible = False
        
        # the Routine "wo_mix_cate_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wo_mix_cont_resp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('wo_mix_cont_resp.started', globalClock.getTime())
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
        
        # keep track of which components have finished
        wo_mix_cont_respComponents = [wo_mix_cont_resp_prompt, wo_mix_cont_resp_slider_shape, wo_mix_cont_resp_slider, wo_mix_cont_resp_left_calm, wo_mix_cont_resp_right_fear, wo_mix_cont_resp_mouse]
        for thisComponent in wo_mix_cont_respComponents:
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
        routineForceEnded = not continueRoutine
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
                else:
                    cont_resp = 'None'
            else:
                cont_resp = 'None'
                    
            
            win.mouseVisible = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_cont_respComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_cont_resp" ---
        for thisComponent in wo_mix_cont_respComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('wo_mix_cont_resp.stopped', globalClock.getTime())
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
        
        thisExp.addData('Anticipation', block_name)
        thisExp.addData('face_stim', mix)
        thisExp.addData('cont_resp',cont_resp)
        
        #thisExp.addData('block_mix_type', mix_type)
        thisExp.addData('Thinking', thinking)
        win.mouseVisible = False
        
        
        
        # the Routine "wo_mix_cont_resp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wo_mix_brea" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('wo_mix_brea.started', globalClock.getTime())
        wo_mix_brea_key_resp.keys = []
        wo_mix_brea_key_resp.rt = []
        _wo_mix_brea_key_resp_allKeys = []
        # Run 'Begin Routine' code from wo_mix_brea_code
        if trialID == 150:
            continueRoutine = True
        else:
            continueRoutine = False
        # keep track of which components have finished
        wo_mix_breaComponents = [wo_mix_brea_text, wo_mix_brea_key_resp]
        for thisComponent in wo_mix_breaComponents:
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
        routineForceEnded = not continueRoutine
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
                theseKeys = wo_mix_brea_key_resp.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wo_mix_breaComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wo_mix_brea" ---
        for thisComponent in wo_mix_breaComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('wo_mix_brea.stopped', globalClock.getTime())
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
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'wo_mix'
    
    
    # --- Prepare to start Routine "end_exp" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end_exp.started', globalClock.getTime())
    # keep track of which components have finished
    end_expComponents = [end_exp_text]
    for thisComponent in end_expComponents:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_expComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_exp" ---
    for thisComponent in end_expComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end_exp.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


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


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
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
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
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
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
