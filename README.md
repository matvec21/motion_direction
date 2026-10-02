# Estimating the Direction of Motion of a Moving Object from Monocular Video Using Inertial Sensor Data

[![License](https://badgen.net/github/license/matvec21/motion_direction?color=green)](https://github.com/matvec21/motion_direction/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/matvec21/motion_direction)](https://github.com/matvec21/motion_direction/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/matvec21/motion_direction.svg?color=0088ff)](https://github.com/matvec21/motion_direction/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/matvec21/motion_direction.svg?color=7f29d6)](https://github.com/matvec21/motion_direction/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Глазунов Матвей </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Крыжановский Максим </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Воронцов Константин </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

We consider the problem of estimating the direction of motion of a moving object from a monocular onboard camera and inertial sensors. The translation direction is visible in the image as the focus of expansion and is usually recovered from the essential matrix of two frames. Such an estimate is noisy, lags behind after smoothing, and breaks down on degenerate frames: low parallax, a dominant plane, fast rotation. We combine the visual measurement with gyroscope and accelerometer data in a complementary filter on the unit sphere. The filter works in the body frame and does not use heading, which is unreliable in our recordings. We plan to add a learned estimate of the reliability of the visual measurement, which sets its weight in the filter. In a preliminary experiment on one recorded sequence, the gyroscope-based filter reduced the delay of the vision-only channel from about 350 ms to 30–130 ms at comparable jitter. The method will be compared with three baselines: a vision-only essential-matrix pipeline, vision with rotation taken from the gyroscope, and the visual-inertial odometry system OpenVINS.

## Problem Statement

**Input.** At each frame `t`: a monocular image `I_t`, angular velocity `ω_t` and acceleration `a_t` from the inertial sensors, and roll/pitch of the object. The camera intrinsics are known.

**Output.** A unit vector `d_t` on the sphere `S²` in the body frame: the direction of the object's velocity, equivalently two angles relative to the optical axis. The speed itself is not estimated and is taken from an external source.

**Quality criteria.**
- angular error with respect to a reference direction;
- delay of the channel relative to a zero-phase reference;
- jitter, the standard deviation of the derivative of the output (it goes into the control loop).

**Difficulties.** The visual measurement is delayed by tracking and smoothing, contains rare outliers, and degenerates on frames with a dominant plane, weak parallax, or rotation that is hard to separate from translation. The inertial integral drifts within seconds, so it can only bridge the gaps between visual corrections.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{glazunov2026motiondirection,
    title={Estimating the Direction of Motion of a Moving Object from Monocular Video Using Inertial Sensor Data},
    author={Matvey Glazunov, Maxim Kryzhanovsky (consultant), Konstantin Vorontsov (advisor)},
    year={2026}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
