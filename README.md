# DynVis - Visualization tool for dynprocs logs

This is a simple tool to visualize past runs using dynamic resources (adding and removing processes dynamically from an MPI job).

Here is a small video to demonstrate how the output of this tool looks like:



https://github.com/boi4/dynprocs_visualize/assets/33987679/416a6c83-0d67-454a-b101-95f064980af8



### Requirements

The visualization is based on [Manim](https://docs.manim.community/en/stable/installation.html), a visualization library for mathematical concepts.

Make sure to follow the [installation instrucions](https://docs.manim.community/en/stable/installation.html) to install Manim on your operating system.

### Usage

Clone the DynVis repo:

```bash
git clone https://github.com/boi4/dynprocs_visualize.git && cd dynprocs_visualize
```

Run the `dynvis.py` script with the path to the log file:

```bash
python3 ./dynvis.py path/to/log/file
```

This will create a rendered video at `media/videos/480p15/VisualizeDynProcs.mp4`.

There exist some command line flags to tweak the behavior of DynVis:

```
usage: dynvis.py [-h] [--quality {low_quality,medium_quality,high_quality}] [--preview] [--round-to ROUND_TO] logfile

positional arguments:
  logfile

options:
  -h, --help            show this help message and exit
  --quality {low_quality,medium_quality,high_quality}, -q {low_quality,medium_quality,high_quality}
  --preview, -p
  --round-to ROUND_TO, -r ROUND_TO
                        On how many 10^r miliseconds to round the time to when aligning events
  --save_last_frame, -s
                        Save as last frame as a picture
```
