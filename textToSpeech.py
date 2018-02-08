#!/bin/python3

import requests
import re
import sys
import os
import collections
import random

from tqdm import tqdm

# system selective imports
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# local selective imports
from dictionaries import CMUDict, GKTDict

cmu = CMUDict("dictionaries/cmudict-0.7b")
gkt = GKTDict("dictionaries/gkt.dict")

force_ours = ("feb", "sterling")
force_cmu = ("",)
always = collections.OrderedDict(
  [
    ("gasnap", "gas nap"),
    ("faurecia", "fauresea uh"),
    ("flix", "flicks"),
    ("telenet", "tele net"),
    ("stockee", "stocky"),
    ("nokee", "no key"),
    ("rbob", "r bob"),
    ("ebob", "e bob"),
    ("mopj", "mop j"),
    ("wti", "w t i"),
    ("k", "kay"),
    ("kc", "kay c"),
    ("df", "d f"),
    ("ql", "q l"),
    ("sm", "s m"),
    ("sb", "s b"),
    ("sz", "s z"),
  ]
)

sometimes = collections.OrderedDict([
  ("dec", "deece"),
])

forced = {
  "confirm": "confirm",
  "spot": "spot",
  "against": "against",
  "poll": "poll",
  "offer": "offer",
  "aussie": "{AO1 S IY2}",
  "two": "two",
  "adient": '{AE1 D IY0 EH2 N T}',
  "october": "{AA0 K T OW1 B ER2}",
  "ziggo": '{Z IH1 G} {OW1}',
  "mill": "mill.",
  "sixteen": "sixteen",
  "o": "{OW1}.",
  "kiwi": "{K IY1 W IY1}",
  "oh": "{OW1}.",
  "e": "{IY1}.",
  "eight": "{EY1 T}",
  "zero": "{Z IY1 R OW2}",
  "and": "{AE2 N D}",
  "nine": "nine",
  "rand": "{R AE2 N D}",
  "euro": "{Y UH1 R OW2}",
  "yen": "{Y EH1 N}",
  "huf": "{HH AH1 F}",
  "ones": "{W AH1 N Z}",
  "one": "{W AH1 N}",
  "dollar": "{D AA1 L ER2}",
  "cable": "{K EY1 B AH0 L}",
  "sterling": "{S T ER1 L IH2 NG}",
  "i": "{AY1}.",
  "ats": "{AE1 T S}",
  "bid": "{B IH1 D}.",
  "swissy": "{S W IH1 S IY2}",
  "loonie": "loonie",
  "naptha": "{N AE1 P F TH AH0}",
  "naphtha": "{N AE1 P F TH AH0}",
  "augie": "{AA1 G IY1}",
  "auggie": "{AA1 G IY1}",
  "fly": "{F L AA0 AY2}",
  "b.k.o.": "{B IY2} {K EY1} {OW1}",
  "buxel": "{B AH1 K S AH1 L}",
  "ozn": "{OW1} {Z IY1} {EH2 N}.",
  "obm": "{OW1} {B IY1} {EH2 M UH0}.",
}


def get_phonemes(sentence):
  for key, val in always.items():
    if key in sentence.split():
      sentence = " ".join([val if i in key else i for i in sentence.split()])

  for key, val in sometimes.items():
    if key in sentence.split() and random.random() < 0.5:
      sentence = " ".join([val if i in key else i for i in sentence.split()])

  sentence = sentence.rstrip().split()
  phonetic_sentence = []

  for word in sentence:
    phonemes = None
    if word in forced.keys():
      phonemes = forced[word]

    elif word in force_ours:
      phonemes = gkt.lookup(word)
    else:
      phonemes = cmu.lookup(word)

    if word.endswith("ber"):
      phonemes = word

    if not phonemes:
      phonemes = word
    elif type(phonemes) == list and len(phonemes) == 1:
      phonemes = "{" + phonemes[0] + "}"

    elif type(phonemes) == list and len(phonemes) > 1:
      # can add heat here later for variety
      phonemes = "{" + phonemes[0] + "}"

    # add space before certain words
    if word == "ats" or "huf" in word or "cad" in word:
      if len(phonetic_sentence) > 0 and phonetic_sentence[-1][-1] != "," and phonetic_sentence[-1][-1] != ".":
        phonetic_sentence[-1] = phonetic_sentence[-1] + ","
    if "week" in word or word.endswith("teen"):
      phonemes += ","
    if "kiwi" in word:
      phonemes += "."

    phonetic_sentence.append(phonemes)
  return " ".join(phonetic_sentence) + ("." if phonemes[-1] != "." else "")


def get_wav(text, filename="test.wav"):
  wav = requests.get("http://localhost:9000" + "/synthesize?text=" + text.replace(",", "%2c")).content
  with open(filename, 'wb') as f:
    f.write(wav)


def get_text(file):
  with open(file) as f:
    return " ".join(f.read().strip().splitlines())


def synthesize(textFile):
  wavFile = textFile.replace(".txt", ".wav")
  if os.path.exists(wavFile):
    return (textFile, "done")
  text = get_text(textFile)
  if wavFile == textFile:
    print("Watch your file extensions")
    return
  # print(text)
  text = get_phonemes(text)
  # print(text)
  get_wav(text, wavFile)
  return (textFile, "done")


if __name__ == "__main__":
  futures = []
  # GPU LIMITED EXECUTION HERE
  executor = ProcessPoolExecutor(max_workers=1)
  if len(sys.argv[1:]) > 0:
    for fl in sys.argv[1:]:
      futures.append(executor.submit(synthesize, fl))
  else:
    for fl in os.popen("ls *.txt").read().split():
      futures.append(executor.submit(synthesize, fl))
  res = [future.result() for future in tqdm(futures)]
  print("Processed {:} text files".format(len(res)))
