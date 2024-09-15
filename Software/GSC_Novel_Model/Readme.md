Here is where I show you some novel KWS models of mine. I can only share with you the architecture of TFResNet version 1, forgive me for that.

Because of technological limitations, GPU in particular, I couldn't go further. It should have many experiments with many random seeds and versions, but with limited free GPU I can not do it.
I can make a much more complicated model, but it's not necessary for the case of KWS. I mostly explored deeper in using CNNs which I believe is the perfect choice for this task.

The gaps in my models versus the baseline model are just relative for some reasons:
- The data I had to preprocess (add noise, transform to Audio Feature, ...) and export them to disk for later use for saving GPU time.
- The training setup was much fewer epochs than any KWS model proposed, just about 80 epochs. The total time for each model training was about 7-8 hours and needed at least 2 Google Drive accounts to handle. It's why I couldn't do more experiments.
