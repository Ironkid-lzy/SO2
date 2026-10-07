# T005 DroQ-style critic source

The hidden-block order Linear -> Dropout -> LayerNorm -> ReLU follows
Takuya Hiraoka et al., OriginalREDQCodebase/redq/algos/core.py:Mlp at
commit 4c7adddbaf7573ff9256b16091a3bbdbd536aad4 of
https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning.
The implementation is adapted to SO2 parallel heads; no DroQ training-rule code was copied.
The referenced project is MIT-licensed. Its license text is preserved in
DROQ_LICENSE_T005.txt. SO2 keeps its own root LICENSE.
