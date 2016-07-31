#!/bin/sh
TRANS=LibriSpeech_train_100.trans
FIELD=LibriSpeech_train_100.fileids
INDEX=train_100_index.txt
awk 'NR==FNR{a[NR]=$0;next}{printf "%s@%s@\n", a[FNR], $0}' $1/$FIELD $1/$TRANS > $INDEX

TRANS=LibriSpeech_test_clean.trans
FIELD=LibriSpeech_test_clean.fileids
INDEX=test_clean_index.txt
awk 'NR==FNR{a[NR]=$0;next}{printf "%s@%s@\n", a[FNR], $0}' $1/$FIELD $1/$TRANS > $INDEX
