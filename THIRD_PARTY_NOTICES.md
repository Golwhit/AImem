# Third-party notices

This file lists third-party material used by the learning core's experiments.
It records notices; it is not legal clearance.

## SCOWL word list (Debian package `wamerican` 2020.12.07-2)

- Role: Layer 0 list of possible English words in `neural_skeletons.py`. It is
  used to decide whether a skeleton such as `th*t` can resolve to only one
  word. It is not training text.
- Source: `/usr/share/dict/american-english`, snapshotted with its SHA-256 in
  `.data/possible-words/` (not committed).
- License: SCOWL and its component lists permit use, copying, modification,
  distribution and sale provided the copyright and permission notices are kept.
  The full notice is in [licenses/SCOWL-wamerican-copyright.txt](licenses/SCOWL-wamerican-copyright.txt).

## World English Bible (eBible.org)

- Role: approved learning and test text for the neural experiments.
- Status: public domain. "World English Bible" is a trademark; changed or
  generated text must not be called the World English Bible.
