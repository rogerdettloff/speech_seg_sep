# speech_seg_sep
Tinkering with ideas about using an RNN to process audio signals for speech segmentation and separation from other acoustic sources.

This project has basic support for for speech segmentation using an rnn "auto encoder" type architecture.  It is written in python using tensorflow for the neural network.

I got the best results so far using BasicRNNCell rather than GRUCell.  I trained the model by giving it isolated speech and an 'ideal' speech mask based on the Short-Time Fourier Transform (STFT) analysis of 3 second long sound clips.  I also trained using other (non-speech) sounds with a mask of all zeros.  
Here's a few details of the model:
I used two layers of stacked rnn cells...a hidden layer and an output layer.  These are dynamically unrolled in time over 95 time steps. 
            n_steps=95,  # for a 3 second training clip.
            n_inputs=513,  # for a 1024 point long STFT window
            n_hidden=256,
            n_outputs=513,  
            keep_prob=0.5  # dropout applied to inputs and outputs of the hidden layer.

Here's a few examples:

### Example 1: Isolated speech
The following figure shows isolated speech from a female speaker, with no interfering sounds mixed into it.  The speech segmentation mask (third subplot) was estimated by the trained BasicRNNCell model.  The final subplot shows the result of applying the mask onto the original sound clip.   When I listen to the original vs masked audio, they sound substantially the same.  This shows that the model is capable of generating a mask to segment the time/frequency points of speech (from an isolated speaker).
![female1 none](https://user-images.githubusercontent.com/6138503/30874780-33c136d4-a2a6-11e7-9f5b-24e992eb3a08.png)
 
### Example 2: Isolated speech (different female speaker)
Similar to example 1 (above), but this is a different speaker reading a different text passage.  The speech segmentation mask is remarkably good.
![female2 none](https://user-images.githubusercontent.com/6138503/30875239-b2fd7006-a2a7-11e7-910a-8bcfc95eefd4.png)

### Example 3: isolated music (no speech)
This example presents a complex musical sound clip (Techno- or Synth-pop).  The speech segmentation mask is almost all zeros...i.,e., no speech.  This shows that the trained model can recognize the difference between speech and other complex sounds, and generate a mask that keeps the speech and reduces other sounds.  At least, for the isolated presentations shown so far.
![music none](https://user-images.githubusercontent.com/6138503/30875378-2e02290e-a2a8-11e7-8be2-ee3a5feee2ae.png)

### Example 4: "Cocktail Party"
This example shows the inference results for a complex auditory scene consisting of many voices chattering in the background, but no clear isolated speaker.  The speech mask generated is interesting...it identifies speech-like sounds even though I cannot tell what anyone is saying.  I was hoping the model would suppress these non-isolated speakers...I guess need a more sophisticated model to learn to do that.
![cocktail_party none](https://user-images.githubusercontent.com/6138503/30876353-3df32162-a2ab-11e7-8af4-e8ee4f38d8eb.png)

### Example 5: speech + music
This final example shows a female speaker mixed with Techno-music in the background.  The estimated speech mask looks pretty good, although after applying the mask to the mixed sound clip leaves room for improvement.  The background music in the post-masked sound clip is definitely much reduced, but the speech is also reduced and a bit muffled sounding--not crisp.  So far, I have only done training usung isolated speech or isolated other sounds...Perhaps I need to do some training using mixed clips like this one.
![female music](https://user-images.githubusercontent.com/6138503/30876728-53150e6a-a2ac-11e7-92aa-1d75dbff5725.png)

I developed and tested this using Enthought Python 3.6.0, and tensorflow 1.3.0.  See the edm_bundle.json file for all the details about how to recreate my development environment.
