* Import works, recog works, produces mostly same WERs
* Training speedup (https://github.com/rwth-i6/returnn/issues/1402).
  Mostly done now, we get almost same speed as TF?
  (CTC still missing)
* [Check older experiments on Conformer](../exp2022_07_21_transducer/exp_fs_base/README.md)

TODO:

- aux48ff, or aux4812ff: aux CTC
- attdrop01
- posdrop01
- wdf?
- wdro
- lsxx01, or other lsxx
- specaugweia? yes but generalize... also could be optimized
- twarp?
- rndresize09_12, or other
- mhsapinit05
- chunk, some variant, unclear which is best...

TODO model changes:

- Second decoder LSTM
- ZoneoutLSTM use_zoneout_output=True
- cnnblstmf2