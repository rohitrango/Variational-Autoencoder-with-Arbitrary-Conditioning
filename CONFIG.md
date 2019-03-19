Here is a sample config file and what each field means. I strongly recommend reading any one of the config files for examples.

```
DATASET:
  NAME: Name of the dataset (currently one of *mnist* or *celeba* )
  PATH: Path to the dataset
  MIN: Min value of the image (default = -1)
  MAX: Max value of the image (default =  1)
  H: Size of the mask to cut out from the image: In case of center or random square crop
  P: Probability of retaining a pixel: In case of random pixel drop
  TYPE: Type of corruption to apply. Current options : <None>, random, center
TRAIN:
  NUM-EPOCHS: Number of epochs to run for
  SAVE-FREQ: Number of *checkpoints* to save after
  BATCH-SIZE: Training batch size
  SHUFFLE: Shuffle examples? 
  STEP-LOG: Number of steps to print logs after
VAL:
  BATCH-SIZE: Validation batch size
  NUM-SAMPLES: Number of samples to generate for each image
  SHUFFLE: Shuffle validation set?
  SAVE-IMG: Save images?
MODEL:
  NAME: Name of model (Choices: EncoderDecoderNet, EncoderDecoderNetMini)
  LAST_LAYER: Activation to apply at last layer: choices: (tanh, sigmoid, <None>)
  INP_CHANNELS: Input channels (3 for celebA, 1 for mnist)
  N_HIDDEN: Number of channels for convolutions
  FC_HIDDEN: Number of hidden FC neurons
  FC_OUT: Size of latent variable 
  OPTIMIZER: Optimizer to use (Adam)
  LR: learning rate
  BETA1: beta1 for adam
  BETA2: beta2 for adam
  SCHEDULER: scheduler to use (step)
  DECAY-STEPS: number of steps to decay lr
  DECAY-FACTOR: lr decay factor 
  WEIGHT-DECAY: weight decay
  INIT: initialization (orthogonal)
  LOSS: loss function to use (mse for continuous, bce for binary)
  LR-OVERRIDE: Override the lr to the value of 'LR' from the config file
REG:
  LAMBDA_KL: coefficient of kl divergence term
  LAMBDA_REG: coefficient of regularizer term
  SIGMA_M: regularizer parameter for mean term
  SIGMA_S: regularizer parameter for variance term
  APPLY_REG: Apply regularizer? 
USE-CUDA: Use GPUs? (who uses CPUs anyway)
PRETRAINED: full path to pretrained model, leave blank for no model
SAVE-PATH: directory to save logs, models, and validation images in. If it doesn't exist, create it
PEEK-VALIDATION: Peek into validation set after every 'peek-validation' steps.
```