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

We consider the problem of estimating the direction of motion of a moving object from a monocular onboard camera and inertial sensors. The translation direction is visible in the image as the focus of expansion and is usually recovered from the essential matrix of two frames. Such an estimate is noisy, lags behind after smoothing, and breaks down on degenerate frames: low parallax, a dominant plane, fast rotation. We combine the visual measurement with gyroscope and accelerometer data in a complementary filter on the unit sphere. The filter works in the body frame and does not use heading, which is unreliable in our recordings. In experiments on one recorded sequence, split into a tuning window and a separate validation window, the filter reduces the delay of the vision-only channel from 333–367 ms and 567–633 ms to 67–200 ms and 0–133 ms respectively. Jitter is lower on the horizontal angle in both windows; on the vertical angle it is 12% higher on the tuning window and 26% lower on the validation window. We plan to add a learned estimate of the reliability of the visual measurement and to compare the method with three baselines: a vision-only essential-matrix pipeline, vision with rotation taken from the gyroscope, and the visual-inertial odometry system OpenVINS.

## Problem Statement

**Given.** For each frame: a video frame and the gyroscope readings (plus the accelerometer, roll and pitch).

**Find.** The direction of motion of the object in each frame.

**Criterion.** Minimal delay `τ` (how far the estimate lags behind the real direction) with jitter not larger than that of the vision-only estimate: `θ* = argmin τ(θ)` subject to `J(θ) ≤ J_vis`, where `θ` are the filter parameters. The delay is measured against a zero-phase reference, the jitter is the standard deviation of the derivative of the output (it goes into the control loop).

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
