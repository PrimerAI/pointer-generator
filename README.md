This repository contains code for experimenting with seq to seq summarization models. It is inititally based off of https://github.com/abisee/pointer-generator (corresponding paper is at https://arxiv.org/abs/1704.04368).

## Notes

- CNN / DailyMail dataset from http://cs.nyu.edu/~kcho/DMQA/
- dependent on Tensorflow
- run using

```
python run_summarization.py --mode={train, eval, decode} --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```
