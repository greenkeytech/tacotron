#!/usr/bin/python

from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
from tqdm import tqdm

# Python2 check
if sys.version[0] == '2':
  import Queue as Queue
else:
  import queue as Queue

import numpy as np
from ffmpy import FFmpeg
from os.path import exists

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from util import audio
from multiprocessing import cpu_count
from hparams import hparams

num_workers = int(os.popen("grep -c proc /proc/cpuinfo").read().strip())
executor = ProcessPoolExecutor(max_workers=num_workers)
futures = []

MAX_TIME = 12.0  # seconds for max segment length
if "MAX_TIME" in os.environ:
  MAX_TIME = float(os.environ["MAX_TIME"])


def writeMetadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'a', encoding='utf-8') as f:
    for m in metadata:
      if len(m) > 0:
        f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[2] for m in metadata if len(m) > 0])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata if len(m) > 0))
  print('Max output length: %d' % max(m[2] for m in metadata if len(m) > 0))


# load stm
def loadText(ifl):
  """ Loads stm file and returns speakers, start/stop times, and text """
  speakers = []
  startStopTimes = []
  text = []

  with open(ifl) as f:
    data = f.read()

  for line in data.splitlines():
    line = line.split()
    speakers.append(line[2])
    startStopTimes.append([float(i) for i in line[3:5]])
    text.append(" ".join(line[6:]))

  return (speakers, startStopTimes, text)


def formatSecString(s):
  """ Formats time in seconds to time in hour:minutes:seconds """
  seconds = s % 60
  minutes = (s % 3600) // 60
  hours = s // 3600
  ret_str = ":".join(["{:}".format(i).zfill(2) for i in [int(hours), int(minutes), seconds]])
  return ret_str


def processAudioClip(listOfArguments):
  start, dur, wav_path, text = listOfArguments
  out_dir = "/".join(wav_path.split("/")[:-2]) + "/training/"
  spectrogram_filename = wav_path.split("/")[-1].replace(".wav", "") + '-spec.npy'
  mel_filename = wav_path.split("/")[-1].replace(".wav", "") + '-mel.npy'

  try:
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    if (len(wav) / 16000.0) > MAX_TIME:
      return ()

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    return (spectrogram_filename, mel_filename, n_frames, text)
  except:
    print(listOfArguments)
    return ()


def splitAudio(ifl):
  """ Loads sph file and splits with stm """
  speakers, times, texts = loadText(ifl.replace(".sph", ".stm"))

  for i, (speaker, startStops, text) in enumerate(zip(speakers, times, texts)):
    start, stop = startStops
    start, dur = map(formatSecString, [float(j) for j in [start, stop - start]])
    wav_path = ifl.replace(".sph", "_{:}.wav".format(str(i).zfill(4)))

    txt_path = ifl.replace(".sph", "_{:}.txt".format(str(i).zfill(4)))
    with open(txt_path, 'w') as f:
      f.write(text)

    if exists(wav_path):
      return ifl + " is done"

    # Hard-coding is dangerous, so do this instead
    try:
      rate = float(os.popen("soxi -r {:}".format(ifl)).read())
    except:
      rate = 16000

    ff = FFmpeg(
      inputs={
        ifl: '-y -f s16le -ar {:} -ac 1 -ss {:} -t {:} -loglevel panic -hide_banner -nostats '.format(rate, start, dur)
      },
      outputs={wav_path: " -filter:a loudnorm -af silenceremove=0:0:-40dB:-1:0.6:-40dB -ar 16k"}
    )
    ff.run()

  return ifl + " is done"


def procAudio(ifl):
  """ Loads sph file and splits with stm """
  global futures
  speakers, times, texts = loadText(ifl.replace(".sph", ".stm"))

  for i, (speaker, startStops, text) in enumerate(zip(speakers, times, texts)):
    start, stop = startStops
    start, dur = map(formatSecString, [float(j) for j in [start, stop - start]])
    wav_path = ifl.replace(".sph", "_{:}.wav".format(str(i).zfill(4)))

    futures.append(executor.submit(partial(processAudioClip, (start, dur, wav_path, text))))

  return


fl = os.popen("ls /data/raw/*.sph").read().split()
os.popen("mkdir -p /data/training ; echo -n \"\" > /data/training/train.txt").read()

futures = []
for i, ifl in enumerate(fl):
  futures.append(executor.submit(partial(splitAudio, ifl)))
print("Finished clipping {:} audio files.".format(len([future.result() for future in tqdm(futures)])))

futures = []
for i, ifl in enumerate(fl):
  procAudio(ifl)

res = [future.result() for future in tqdm(futures)]
print("Finished converting {:} audio clips.".format(len(res)))

writeMetadata(res, "/data/training/")
