
#### harness position error (seed mean)

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| two_body err@5 | 0.00105 | 0.000999 | 0.00156 | 0.00108 |
| two_body err@10 | 0.0031 | 0.00302 | 0.00462 | 0.00318 |
| two_body err@20 | 0.0091 | 0.00809 | 0.0119 | 0.00833 |
| two_body err@100 | 0.131 | 0.153 | 0.139 | 0.131 |
| expB err@5 | 0.00301 | 0.00333 | 0.00498 | 0.00318 |
| expB err@10 | 0.00813 | 0.00888 | 0.0139 | 0.00884 |
| expB err@20 | 0.0239 | 0.0262 | 0.0394 | 0.027 |
| expB err@100 | 1.42 | 1.53 | 1.63 | 1.54 |
| bh300 err@5 | 0.00795 | 0.00946 | 0.0128 | 0.00905 |
| bh300 err@10 | 0.0209 | 0.0247 | 0.0346 | 0.0234 |
| bh300 err@20 | 0.0622 | 0.0759 | 0.107 | 0.0691 |
| bh300 err@100 | 4.73 | 4.97 | 6.95 | 5.08 |

#### tick-100 conservation (model; truth in last column of baseline)

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| two_body dE_rel_model | 0.129 | 0.331 | 0.14 | 0.131 |
| two_body dP_model | 3.49e-10 | 4.95e-10 | 1.02e-09 | 1.55e-10 |
| two_body dL_rel_model | 0.666 | 0.232 | 0.409 | 0.274 |
| expB dE_rel_model | 0.0107 | 0.0259 | 0.0269 | 0.0217 |
| expB dP_model | 3.18e-06 | 3.29e-06 | 3.29e-06 | 3.2e-06 |
| expB dL_rel_model | 0.393 | 0.374 | 2 | 0.877 |
| bh300 dE_rel_model | 0.0391 | 0.0461 | 0.186 | 0.0902 |
| bh300 dP_model | 4.37e-05 | 5e-05 | 8.27e-05 | 8.07e-05 |
| bh300 dL_rel_model | 0.303 | 0.171 | 0.554 | 0.0442 |

#### 1000-tick conservation rows

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| cons nr100 dE | 0.017 | 0.0742 | 0.167 | 0.287 |
| cons nr100 dL | 0.0378 | 0.0252 | 0.0426 | 0.0543 |
| cons nr100 dP | 2.01e-05 | 1.98e-05 | 1.43e-05 | 1.43e-05 |
| cons nr100 bound | 0.89 | 0.9 | 0.94 | 0.97 |
| cons nr100 vir | 1 | 1 | 0.904 | 0.835 |
| cons nr100 outgrid | 0.16 | 0.12 | 0.04 | 0.08 |
| cons nr100 (truth) dE | 4.36e-05 | - | - | - |
| cons nr100 bound_gap | 0.08 | 0.07 | 0.03 | 0 |
| cons nr100 vir_gap | 0.105 | 0.105 | 0.000242 | 0.0796 |
| cons bh300 dE | 0.481 | 0.666 | 2.25 | 0.648 |
| cons bh300 dL | 1.46 | 0.652 | 1.88 | 0.534 |
| cons bh300 dP | 0.000144 | 0.000103 | 0.000441 | 0.00032 |
| cons bh300 bound | 0.753 | 0.723 | 0.167 | 0.74 |
| cons bh300 vir | 0.966 | 0.94 | 53.8 | 1.22 |
| cons bh300 outgrid | 0.277 | 0.31 | 0.83 | 0.293 |
| cons bh300 (truth) dE | 0.000327 | - | - | - |
| cons bh300 bound_gap | 0.127 | 0.157 | 0.713 | 0.14 |
| cons bh300 vir_gap | 0.0372 | 0.0102 | 4.06 | 0.271 |
| bh300 vmax/c | 0.926 | 0.926 | 0.931 | 0.928 |

#### timing ms/tick

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| two_body | 4.72 | 4.81 | 4.8 | 4.81 |
| expB | 4.95 | 5.03 | 5.03 | 5.03 |
| bh300 | 2.49 | 2.6 | 2.61 | 2.56 |
| expB_batch1_n100 | 2.46 | 2.54 | 2.57 | 2.53 |

#### OOD battery err@2 (mass / eps / density / other)

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| ref/cluster_N100_s8_v0.3 | - | 0.0336 | 0.0486 | 0.0323 |
| mass_ratio_2body/binary_q10 | - | 0.0998 | 0.149 | 0.069 |
| mass_dominant/cluster_heavy30 | - | 0.618 | 1.82 | 0.732 |
| mass_spectrum/logunif_0.1_10 | - | 0.163 | 0.173 | 0.149 |
| mass_scale_equal/equal_m3.0 | - | 0.0962 | 0.136 | 0.0917 |
| mass_scale_equal/equal_m10.0 | - | 0.989 | 0.993 | 0.98 |
| density_sigma/sigma2.0 | - | 0.341 | 1.95 | 0.32 |
| density_plummer/plummer_a0.25 | - | 3.55 | 7.97 | 6.41 |
| softening_eps/cluster_eps0.1 | - | 0.482 | 0.535 | 0.416 |
| softening_eps/cluster_eps2.0 | - | 0.737 | 0.0243 | 0.0157 |
| boundary_offset/cluster_shift24.0 | - | 0.412 | 0.474 | 0.469 |
| dt/cluster_dt0.4 | - | 0.11 | 0.0984 | 0.101 |
| velocity_scale/vfac4.0 | - | 0.1 | 0.113 | 0.113 |
| ic_shape/filament | - | 0.331 | 0.413 | 0.334 |
| binary_tight/binary_d0.25 | - | 0.111 | 0.141 | 0.0994 |
| geomean ratio to ref | - | 7.91 | 6.29 | 6.39 |

#### OOD all axes err@2

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| ref/cluster_N100_s8_v0.3 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| ref/cluster_N300_s8_v0.3 | 0.067 | 0.0825 | 0.119 | 0.0745 |
| ref/bh300_c10 | 0.0622 | 0.0759 | 0.107 | 0.0691 |
| mass_ratio_2body/binary_q1 | 0.0106 | 0.00483 | 0.00917 | 0.00792 |
| mass_ratio_2body/binary_q3 | 0.0362 | 0.0172 | 0.0204 | 0.0176 |
| mass_ratio_2body/binary_q10 | 0.352 | 0.103 | 0.162 | 0.0695 |
| mass_ratio_2body/binary_q100 | 7.01 | 0.695 | 1.85 | 0.483 |
| mass_ratio_2body/binary_q1000 | 216 | 3.62 | 20.5 | 2.18 |
| mass_ratio_2body/binary_q10000 | 2.54e+03 | 106 | 257 | 97.3 |
| mass_dominant/cluster_heavy1 | 0.0303 | 0.0329 | 0.0528 | 0.0341 |
| mass_dominant/cluster_heavy3 | 0.0334 | 0.0349 | 0.0561 | 0.0361 |
| mass_dominant/cluster_heavy10 | 0.0625 | 0.054 | 0.088 | 0.0489 |
| mass_dominant/cluster_heavy30 | 1.47 | 0.648 | 1.78 | 0.668 |
| mass_dominant/cluster_heavy100 | 10.6 | 3.2 | 13.2 | 4.75 |
| mass_dominant/cluster_heavy300 | 42.6 | 13.2 | 58.4 | 20.8 |
| mass_dominant/cluster_heavy1000 | 694 | 46.1 | 475 | 68.3 |
| mass_dominant/cluster_heavy10000 | 5.5e+04 | 398 | 6.18e+03 | 577 |
| mass_spectrum/logunif_0.5_2 | 0.0432 | 0.042 | 0.0601 | 0.0421 |
| mass_spectrum/logunif_0.1_10 | 0.207 | 0.144 | 0.171 | 0.139 |
| mass_spectrum/logunif_0.01_100 | 20.6 | 6.16 | 5.11 | 4.61 |
| mass_spectrum/logunif_0.001_1000 | 3.99e+03 | 280 | 824 | 1.42e+03 |
| mass_scale_equal/equal_m0.01 | 0.0017 | 0.000742 | 0.00101 | 0.000757 |
| mass_scale_equal/equal_m0.1 | 0.00951 | 0.00605 | 0.00833 | 0.00607 |
| mass_scale_equal/equal_m0.3 | 0.016 | 0.0133 | 0.0187 | 0.0133 |
| mass_scale_equal/equal_m1.0 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| mass_scale_equal/equal_m3.0 | 0.119 | 0.103 | 0.136 | 0.0996 |
| mass_scale_equal/equal_m10.0 | 1.92 | 1.13 | 1.06 | 1.02 |
| mass_scale_equal/equal_m30.0 | 11.3 | 7.28 | 7.75 | 6.14 |
| mass_scale_equal/equal_m100.0 | 737 | 18.6 | 24.9 | 17.1 |
| velocity_scale/vfac0.0 | 0.0304 | 0.0308 | 0.0485 | 0.0325 |
| velocity_scale/vfac0.1 | 0.0307 | 0.0332 | 0.0495 | 0.0341 |
| velocity_scale/vfac0.3 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| velocity_scale/vfac0.6 | 0.0266 | 0.0294 | 0.0439 | 0.0314 |
| velocity_scale/vfac1.0 | 0.0257 | 0.0277 | 0.0405 | 0.0285 |
| velocity_scale/vfac1.5 | 0.0311 | 0.0291 | 0.0384 | 0.0292 |
| velocity_scale/vfac2.5 | 0.056 | 0.0467 | 0.0571 | 0.0511 |
| velocity_scale/vfac4.0 | 0.14 | 0.0981 | 0.107 | 0.108 |
| velocity_scale/vfac8.0 | 0.544 | 0.318 | 0.215 | 0.273 |
| relativistic_c/bh300_c1.5 | 0.0223 | 0.0273 | 0.0377 | 0.0249 |
| relativistic_c/bh300_c3.0 | 0.0412 | 0.0485 | 0.0676 | 0.0462 |
| relativistic_c/bh300_c5.0 | 0.0537 | 0.0642 | 0.0883 | 0.0602 |
| relativistic_c/bh300_c10.0 | 0.0622 | 0.0759 | 0.107 | 0.0691 |
| relativistic_c/bh300_c30.0 | 0.0668 | 0.0825 | 0.118 | 0.0743 |
| relativistic_c/bh300_c100.0 | 0.0673 | 0.0834 | 0.119 | 0.0748 |
| relativistic_c/bh300_c1000.0 | 0.0674 | 0.0836 | 0.119 | 0.075 |
| relativistic_c/bh300_c10_vfac0.0 | 0.0605 | 0.0686 | 0.114 | 0.0651 |
| relativistic_c/bh300_c10_vfac1.0 | 0.0486 | 0.0497 | 0.0691 | 0.0516 |
| relativistic_c/bh300_c10_vfac3.0 | 0.0306 | 0.026 | 0.0317 | 0.031 |
| density_sigma/sigma1.0 | 1.15 | 1.09 | 5.12 | 1.03 |
| density_sigma/sigma2.0 | 0.398 | 0.403 | 2.93 | 0.346 |
| density_sigma/sigma3.0 | 0.163 | 0.152 | 0.316 | 0.157 |
| density_sigma/sigma5.0 | 0.0505 | 0.0599 | 0.0962 | 0.059 |
| density_sigma/sigma8.0 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| density_sigma/sigma10.0 | 0.0251 | 0.0261 | 0.0421 | 0.0287 |
| density_sigma/sigma16.0 | 0.0312 | 0.0306 | 0.102 | 0.0349 |
| density_sigma/sigma32.0 | 0.0879 | 0.0834 | 0.358 | 0.0828 |
| density_plummer/plummer_a0.25 | 3.33 | 3.58 | 8.27 | 5.22 |
| density_plummer/plummer_a0.5 | 0.602 | 0.573 | 2.6 | 0.546 |
| density_plummer/plummer_a1.0 | 0.259 | 0.252 | 0.365 | 0.251 |
| density_plummer/plummer_a2.0 | 0.112 | 0.115 | 0.185 | 0.12 |
| density_plummer/plummer_a4.0 | 0.0474 | 0.0537 | 0.0785 | 0.0539 |
| density_plummer/plummer_a8.0 | 0.0272 | 0.0333 | 0.0562 | 0.0333 |
| close_pass_impact/pass_b0.0 | 0.0123 | 0.00632 | 0.0112 | 0.00709 |
| close_pass_impact/pass_b0.25 | 0.00988 | 0.00843 | 0.00921 | 0.00695 |
| close_pass_impact/pass_b0.5 | 0.00763 | 0.0102 | 0.00829 | 0.00657 |
| close_pass_impact/pass_b1.0 | 0.0103 | 0.00463 | 0.00683 | 0.00561 |
| close_pass_impact/pass_b2.0 | 0.00665 | 0.00493 | 0.00497 | 0.00449 |
| close_pass_impact/pass_b4.0 | 0.00823 | 0.00502 | 0.00733 | 0.00443 |
| close_pass_impact/pass_b8.0 | 0.0109 | 0.00697 | 0.00914 | 0.00464 |
| close_pass_impact/headon_fall_d1.0 | 0.00652 | 0.00806 | 0.00762 | 0.00653 |
| close_pass_impact/headon_fall_d4.0 | 0.0168 | 0.00868 | 0.0138 | 0.00805 |
| close_pass_impact/headon_fall_d12.0 | 0.0163 | 0.0101 | 0.013 | 0.0127 |
| binary_tight/binary_d0.125 | 0.0505 | 0.0738 | 0.0992 | 0.0679 |
| binary_tight/binary_d0.25 | 0.0746 | 0.112 | 0.14 | 0.0975 |
| binary_tight/binary_d0.5 | 0.0451 | 0.0369 | 0.0162 | 0.0414 |
| binary_tight/binary_d1.0 | 0.0448 | 0.0404 | 0.0537 | 0.0431 |
| binary_tight/binary_d2.0 | 0.0164 | 0.00819 | 0.027 | 0.0196 |
| binary_tight/binary_d4.0 | 0.0106 | 0.00483 | 0.00917 | 0.00792 |
| binary_tight/binary_d10.0 | 0.0101 | 0.0104 | 0.0109 | 0.00646 |
| hier_triple/triple_ain0.2 | 0.0457 | 0.0696 | 0.0882 | 0.0613 |
| hier_triple/triple_ain0.5 | 0.0164 | 0.0208 | 0.0179 | 0.0225 |
| hier_triple/triple_ain1.0 | 0.0325 | 0.0284 | 0.0405 | 0.0302 |
| hier_triple/triple_ain2.0 | 0.013 | 0.00587 | 0.0253 | 0.0125 |
| hier_triple/triple_ain0.5_m3x10 | 0.232 | 0.0907 | 0.149 | 0.12 |
| softening_eps/cluster_eps0.05 | 1.42 | 1.5 | 2.03 | 1.49 |
| softening_eps/cluster_eps0.1 | 0.453 | 0.534 | 0.579 | 0.421 |
| softening_eps/cluster_eps0.25 | 0.286 | 0.319 | 0.3 | 0.165 |
| softening_eps/cluster_eps0.5 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| softening_eps/cluster_eps1.0 | 0.393 | 0.42 | 0.0625 | 0.0214 |
| softening_eps/cluster_eps2.0 | 0.734 | 0.787 | 0.0261 | 0.016 |
| softening_eps/binary_d1_eps0.1 | 0.509 | 0.578 | 0.394 | 0.307 |
| softening_eps/binary_d1_eps0.25 | 0.373 | 0.395 | 0.391 | 0.0489 |
| softening_eps/binary_d1_eps0.5 | 0.0448 | 0.0404 | 0.0537 | 0.0431 |
| softening_eps/binary_d1_eps1.0 | 0.787 | 0.802 | 0.0601 | 0.0178 |
| dt/cluster_dt0.0125 | 0.0356 | 0.0423 | 0.0497 | 0.0385 |
| dt/cluster_dt0.025 | 0.0342 | 0.0408 | 0.0494 | 0.0376 |
| dt/cluster_dt0.05 | 0.032 | 0.0382 | 0.0489 | 0.0362 |
| dt/cluster_dt0.1 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| dt/cluster_dt0.2 | 0.0386 | 0.0427 | 0.0518 | 0.0393 |
| dt/cluster_dt0.4 | 0.12 | 0.118 | 0.107 | 0.112 |
| dt/bh300_c10_dt0.05 | 0.0649 | 0.0742 | 0.104 | 0.0721 |
| dt/bh300_c10_dt0.1 | 0.0622 | 0.0759 | 0.107 | 0.0691 |
| dt/bh300_c10_dt0.2 | 0.11 | 0.112 | 0.129 | 0.108 |
| dt/binary_d1_dt0.05 | 0.0475 | 0.0446 | 0.0573 | 0.0465 |
| dt/binary_d1_dt0.1 | 0.0448 | 0.0404 | 0.0537 | 0.0431 |
| dt/binary_d1_dt0.2 | 0.0344 | 0.0258 | 0.0425 | 0.0331 |
| dt/binary_d1_dt0.4 | 0.0335 | 0.0472 | 0.0325 | 0.0284 |
| ic_shape/uniform_L10 | 0.0465 | 0.0531 | 0.0786 | 0.0539 |
| ic_shape/uniform_L25 | 0.0191 | 0.0212 | 0.0297 | 0.0197 |
| ic_shape/shell_R8_rest | 0.11 | 0.118 | 0.151 | 0.126 |
| ic_shape/shell_R16_rest | 0.0609 | 0.0718 | 0.0943 | 0.0747 |
| ic_shape/shell_R12_rot | 0.0672 | 0.0797 | 0.112 | 0.0806 |
| ic_shape/disk_R12 | 0.0362 | 0.0409 | 0.0681 | 0.0389 |
| ic_shape/disk_R24 | 0.0163 | 0.0182 | 0.0301 | 0.0195 |
| ic_shape/two_cluster_v1 | 0.0456 | 0.0521 | 0.0807 | 0.0506 |
| ic_shape/two_cluster_v4 | 0.0432 | 0.0486 | 0.0811 | 0.049 |
| ic_shape/two_cluster_v0 | 0.0459 | 0.0508 | 0.0826 | 0.0513 |
| ic_shape/filament | 0.266 | 0.289 | 0.361 | 0.267 |
| boundary_offset/cluster_shift0.0 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| boundary_offset/cluster_shift8.0 | 0.0313 | 0.0375 | 0.0528 | 0.0373 |
| boundary_offset/cluster_shift16.0 | 0.0689 | 0.0983 | 0.143 | 0.115 |
| boundary_offset/cluster_shift24.0 | 0.427 | 0.424 | 0.47 | 0.457 |
| boundary_offset/cluster_shift32.0 | 1.28 | 1.18 | 1.17 | 1.23 |
| boundary_offset/cluster_shift40.0 | 1.45 | 1.37 | 1.34 | 1.38 |
| boundary_offset/cluster_vfac1.5_shift0.0 | 0.0311 | 0.0291 | 0.0384 | 0.0292 |
| boundary_offset/cluster_vfac1.5_shift20.0 | 0.209 | 0.234 | 0.425 | 0.332 |
| boundary_offset/cluster_vfac4.0_shift0.0 | 0.14 | 0.0981 | 0.107 | 0.108 |
| boundary_offset/cluster_vfac4.0_shift20.0 | 0.243 | 0.213 | 0.355 | 0.267 |
| mass_dtmatched/equal_m0.1_dt0.1/sqrt(m) | - | 0.0419 | 0.0488 | 0.0373 |
| mass_dtmatched/equal_m0.3_dt0.1/sqrt(m) | - | 0.0393 | 0.0487 | 0.0363 |
| mass_dtmatched/equal_m3.0_dt0.1/sqrt(m) | - | 0.0372 | 0.049 | 0.0359 |
| mass_dtmatched/equal_m10.0_dt0.1/sqrt(m) | - | 0.0538 | 0.0509 | 0.0483 |
| mass_dtmatched/equal_m30.0_dt0.1/sqrt(m) | - | 0.0929 | 0.0594 | 0.0859 |
| mass_dtmatched/equal_m100.0_dt0.1/sqrt(m) | - | 0.178 | 0.0883 | 0.17 |
| mass_dtmatched/cluster_heavy10_dt*sqrt(10/Mh) | - | 0.054 | 0.088 | 0.0489 |
| mass_dtmatched/cluster_heavy30_dt*sqrt(10/Mh) | - | 0.196 | 0.716 | 0.221 |
| mass_dtmatched/cluster_heavy100_dt*sqrt(10/Mh) | - | 0.363 | 3.15 | 0.613 |
| mass_dtmatched/cluster_heavy300_dt*sqrt(10/Mh) | - | 0.435 | 4.85 | 1.69 |
| mass_dtmatched/cluster_heavy1000_dt*sqrt(10/Mh) | - | 0.647 | 5.13 | 1.68 |
| mass_dtmatched/binary_q10_dt*sqrt(2/(1+q)) | - | 0.0252 | 0.0298 | 0.0129 |
| mass_dtmatched/binary_q100_dt*sqrt(2/(1+q)) | - | 0.0127 | 0.0206 | 0.00757 |
| mass_dtmatched/binary_q1000_dt*sqrt(2/(1+q)) | - | 0.0111 | 0.0243 | 0.00839 |
| precision/cluster_fp32 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| precision/bh300_c10_fp32 | 0.0622 | 0.0759 | 0.107 | 0.0691 |
| precision/cluster_fp64 | 0.031 | 0.0358 | 0.0487 | 0.035 |
| precision/bh300_c10_fp64 | 0.0622 | 0.0759 | 0.107 | 0.0691 |
| precision/cluster_tf32 | 0.031 | 0.0358 | 0.0488 | 0.0349 |
| precision/bh300_c10_tf32 | 0.0622 | 0.0759 | 0.108 | 0.0691 |
| precision/cluster_half16 | 0.031 | 0.0359 | 0.0487 | 0.035 |
| precision/bh300_c10_half16 | 0.0621 | 0.0761 | 0.107 | 0.0691 |
| precision/cluster_bf16 | 0.0331 | 0.0385 | 0.0568 | 0.0374 |
| precision/bh300_c10_bf16 | 0.0631 | 0.0758 | 0.111 | 0.07 |

#### scaling r46 err@5

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| R_G32 | 0.128 | 0.236 | 0.0599 | 0.148 |
| R_G64 | 0.0929 | 0.124 | 0.0243 | 0.0116 |
| R_G128 | 0.00756 | 0.00906 | 0.0129 | 0.00878 |
| R_G256 | 0.227 | 0.257 | 0.0188 | 0.138 |
| R_G512 | 0.858 | 0.712 | 0.222 | 0.402 |

#### scaling r46 err@20

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| R_G32 | 1.79 | 3.49 | 1.07 | 1.47 |
| R_G64 | 1.12 | 2.31 | 0.32 | 0.117 |
| R_G128 | 0.0633 | 0.0781 | 0.123 | 0.0714 |
| R_G256 | 1.93 | 2.04 | 0.166 | 0.967 |
| R_G512 | 7.9 | 6.38 | 1.93 | 6.2 |

#### scaling n46 err@5

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| cold_316 | 0.00797 | 0.00951 | 0.0135 | 0.00924 |
| cold_1000 | 0.016 | 0.0174 | 0.0297 | 0.0172 |
| cold_3162 | 0.0636 | 0.0551 | 0.12 | 0.0668 |
| cold_10000 | 1.77 | 2.02 | 2.44 | 2.62 |
| cold_31623 | 12.7 | 12.1 | 13.6 | 18.5 |
| warm_316 | 0.00929 | 0.00886 | 0.012 | 0.00881 |
| warm_1000 | 0.0307 | 0.0279 | 0.0359 | 0.028 |
| warm_3162 | 0.105 | 0.0971 | 0.106 | 0.0878 |
| warm_10000 | 0.338 | 0.29 | 0.27 | 0.231 |
| warm_31623 | 1.67 | 1.4 | 1.26 | 0.87 |

#### scaling n46 err@20

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| cold_316 | 0.0669 | 0.0828 | 0.13 | 0.0757 |
| cold_1000 | 0.583 | 0.536 | 1.28 | 0.526 |
| cold_3162 | 6.8 | 7.14 | 10.8 | 7.89 |
| cold_10000 | 42.3 | 48.6 | 58.9 | 66.8 |
| cold_31623 | 93.1 | 80.5 | 117 | 122 |
| warm_316 | 0.0718 | 0.071 | 0.0993 | 0.0707 |
| warm_1000 | 0.307 | 0.247 | 0.348 | 0.241 |
| warm_3162 | 1.41 | 0.964 | 1.12 | 0.776 |
| warm_10000 | 5.83 | 3.26 | 3.27 | 2.22 |
| warm_31623 | 20.9 | 11.4 | 12.1 | 8.65 |

#### scaling n46big err@5

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| cold_316 | 0.00797 | 0.00951 | 0.0135 | 0.00924 |
| cold_1000 | 0.016 | 0.0174 | 0.0297 | 0.0172 |
| cold_3162 | 0.0636 | 0.0551 | 0.12 | 0.0668 |
| cold_10000 | 1.77 | 2.02 | 2.44 | 2.62 |
| cold_31623 | 12.7 | 12.1 | 13.6 | 18.5 |
| cold_100000 | 35.1 | 26.7 | 44.8 | 32.5 |
| warm_316 | 0.00929 | 0.00886 | 0.012 | 0.00881 |
| warm_1000 | 0.0307 | 0.0279 | 0.0359 | 0.028 |
| warm_3162 | 0.105 | 0.0971 | 0.106 | 0.0878 |
| warm_10000 | 0.338 | 0.29 | 0.27 | 0.231 |
| warm_31623 | 1.67 | 1.4 | 1.26 | 0.87 |

#### scaling n46big err@20

| row | baseline_v2 | control | joint | int_full1 |
|---|---|---|---|---|
| cold_316 | 0.0669 | 0.0828 | 0.13 | 0.0757 |
| cold_1000 | 0.583 | 0.536 | 1.28 | 0.526 |
| cold_3162 | 6.8 | 7.14 | 10.8 | 7.89 |
| cold_10000 | 42.3 | 48.6 | 58.9 | 66.8 |
| cold_31623 | 93.1 | 80.5 | 117 | 122 |
| cold_100000 | 238 | 138 | 266 | 169 |
| warm_316 | 0.0718 | 0.071 | 0.0993 | 0.0707 |
| warm_1000 | 0.307 | 0.247 | 0.348 | 0.241 |
| warm_3162 | 1.41 | 0.964 | 1.12 | 0.776 |
| warm_10000 | 5.83 | 3.26 | 3.27 | 2.22 |
| warm_31623 | 20.9 | 11.5 | 12.1 | 8.65 |
