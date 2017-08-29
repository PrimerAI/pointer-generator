The file sample_output.tsv contains the reference summary, lexrank summary, and the seq-to-seq summary for 100 articles.

The following (with Tensorflow==1.1.0) should regenerate the file.
```
------------------------------------------------------------------
Reproducibility fingerprint:

# Use these shell commands to reproduce this run:
git checkout ae2be3e
SEED=7365955222154579289 python scripts.py results/sample_output.tsv

File hashes:
  results/articles/article_0.txt: 1e93c858e050f1d6b6fd1cc949944fb1c47f9796
------------------------------------------------------------------
Using TensorFlow backend.
/Users/michaelwu/dev/env/python_2.7.10/lib/python2.7/site-packages/sklearn/base.py:315: UserWarning: Trying to unpickle estimator LogisticRegression from version 0.18 when using version 0.18.1. This might lead to breaking code or invalid results. Use at your own risk.
  UserWarning)
2017-08-29 12:03:28.866668: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 12:03:28.866687: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 12:03:28.866691: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 12:03:28.866695: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 12:03:28.866699: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
####################
2013 will be a year of harsh change for the tech industry. Microsoft has been very candid about "missing a generation" of mobile innovation after Apple introduced the iPhone, and 2012 was all about the results of a furious catch - up effort. Apple found success in 2012 by introducing an iPhone with a bigger screen and an iPad with a smaller screen, but it'll have to focus on software in 2013 to stay ahead of the competition.
Time: 16.4391820431 | Score: -0.157654282735
```
