# Solar Panels Segmentation
Code from *A Comparative Evaluation of Deep Learning Techniques for Photovoltaic Panel Detection from Aerial Images* (IEEE Access).
 
> Paper available at: [https://ieeexplore.ieee.org/document/10122915](https://ieeexplore.ieee.org/document/10122915)

![Dataset image](/assets/dataset.png)

## Installation
- Clone the repository
- Create a new environment, e.g. `python3 -m venv .venv`
- Install requirements, `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

## Experiments
Everything is launched through the `run.py` file, using a combination of `click` and `pydantic`.
For more information about commands and their options, use `python run.py [command] --help`.

### Training
To launch a simple training, run `python run.py train-segmenter --data-folder=... [args]`.
This will generate an output folder with a specific name, if provided, or simply the current timestamp.
Inside each experiment directory, you'll find model checkpoints, output and tensorboard logs and the launch config.

Check [the launch script](/tools/launch.sh) for an example of how to launch an experiment.

### Testing
To test the same experiment, launch `python run.py test-segmenter --data-folder=... --output-folder=NAME_OF_THE_EXPERIMENT [other arguments, e.g. encoder type]`
The name is crucial, so that the task can find the right directory.

Check [the launch script](/tools/test.sh) for an example of how to launch a test on the test set.
Use instead [the prediction script](/tools/predict.sh) to generate predictions on a series of large rasters.
