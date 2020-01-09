# Traffic Prophet

Traffic Prophet is a suite of Python tools to transform Toronto's historical
traffic count data to estimates of annual average daily traffic (AADT) on all
city streets. It is based on the Traffic Emissions Prediction Scheme codebase
created by Dr. Arman Ganji of the University of Toronto ([paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12508)).

Traffic Prophet currently remains under development.  On initial release, it
will contain:

* A method, based on [Bagheri et al. 2014](
https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29TE.1943-5436.0000528), of
extrapolating short-term traffic counts into AADTs using nearby long-term
counts.
* Gaussian process regression method for estimating arterial AADTs from nearby
observed AADTs.
* Support vector regression method for estimating local road AADTs from nearby
observed AADTs.

## Usage

### Installation

To import `traffic_prophet`, add this folder to the Python PATH, eg. with

```
import sys
sys.path.append('<FULL PATH OF bdit_traffic_prophet>')
```

### Testing

To test the package, run the following in this folder:

```
pytest -s -v --pyargs traffic_prophet
```

## License

Traffic Prophet is licensed under the GNU General Public License v3.0 - see the
`LICENSE` file.
