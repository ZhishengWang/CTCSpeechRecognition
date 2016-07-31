#!/bin/sh
chmod u+rx  ./generateIndices.sh
ln -s ../Mapper.lua .
LibriSpeech_PATH='../../../../../media/vlsi/vlsi-data/sharedata/speech-corpus/LibriSpeech/LibriSpeech'
echo "ROOT_FOLDER: $LibriSpeech_PATH"
echo "Generating Indices..."
./generateIndices.sh $LibriSpeech_PATH
echo "Generating LMDB..."
th generateLMDB.lua $LibriSpeech_PATH/
