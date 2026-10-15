# Polaris job list (Augur / Bounce)

Source: Claude Code transcripts under ~/.claude/projects (not job logs). Times are the tool-call timestamp (UTC) of submission; dates are submission, not start. IDs 555, 777, 123456 (examples in docs) excluded.

Confirmed jobs: 410, range 2805-3423, 2026-09-21 18:41:42 UTC to 2026-10-08 12:39:42 UTC.

| Job ID | Submitted | Description | Source confidence |
|---|---|---|---|
| 2805 | 2026-09-21 18:41:42 UTC | polaris_mcp_test | confirmed (submit_job result); MCP smoke test |
| 2806 | 2026-09-21 18:44:06 UTC | polaris_mcp_gpu_test | confirmed (submit_job result); MCP smoke test |
| 2808 | 2026-09-21 21:50:53 UTC | bounce-stage1 | confirmed (submit_job result) |
| 2809 | 2026-09-21 21:55:52 UTC | bounce-stage1 | confirmed (submit_job result) |
| 2810 | 2026-09-21 22:10:48 UTC | bounce-stage1 | confirmed (submit_job result) |
| 2811 | 2026-09-21 22:16:04 UTC | bounce-pilot | confirmed (submit_job result) |
| 2812 | 2026-09-21 22:56:25 UTC | bounce-stage1 | confirmed (submit_job result) |
| 2813 | 2026-09-21 23:23:02 UTC | bounce-stage1-occloss | confirmed (submit_job result) |
| 2814 | 2026-09-21 23:33:56 UTC | scripts/polaris_train.sh | confirmed (sbatch output) |
| 2815 | 2026-09-22 00:26:38 UTC | bounce-stage2 | confirmed (submit_job result) |
| 2816 | 2026-09-22 01:28:07 UTC | bounce-stage2-fix | confirmed (submit_job result) |
| 2817 | 2026-09-22 02:42:20 UTC | bounce-stage2-200ep | confirmed (submit_job result) |
| 2818 | 2026-09-22 03:07:05 UTC | bounce-stage2-h12 | confirmed (submit_job result) |
| 2819 | 2026-09-22 04:00:12 UTC | bounce-stage2-h12-fixed | confirmed (submit_job result) |
| 2821 | 2026-09-22 04:38:15 UTC | bounce-stage2-h12-fixed-200ep | confirmed (submit_job result) |
| 2822 | 2026-09-22 05:26:45 UTC | bounce-stage2-flownet | confirmed (submit_job result) |
| 2823 | 2026-09-22 05:51:33 UTC | bounce-stage2-flownet-v2 | confirmed (submit_job result) |
| 2824 | 2026-09-22 06:23:52 UTC | bounce-stage2-flownet-v3 | confirmed (submit_job result) |
| 2825 | 2026-09-22 06:51:41 UTC | bounce-stage2-flownet-v4 | confirmed (submit_job result) |
| 2826 | 2026-09-22 07:27:13 UTC | bounce-stage2-flownet-v5 | confirmed (submit_job result) |
| 2827 | 2026-09-22 07:29:00 UTC | bounce-stage2-flownet-v5 | confirmed (submit_job result) |
| 2828 | 2026-09-22 08:07:04 UTC | bounce-stage2-flownet-v6 | confirmed (submit_job result) |
| 2829 | 2026-09-22 08:51:32 UTC | bounce-stage2-flownet-v7 | confirmed (submit_job result) |
| 2831 | 2026-09-22 17:19:59 UTC | bounce-stage2-flownet-v8 | confirmed (submit_job result) |
| 2833 | 2026-09-22 17:48:09 UTC | bounce-stage2-flownet-v9 | confirmed (submit_job result) |
| 2834 | 2026-09-22 17:49:16 UTC | bounce-stage2-flownet-v9 | confirmed (submit_job result) |
| 2835 | 2026-09-22 17:49:45 UTC | bounce-windowed-hist3 | confirmed (submit_job result) |
| 2840 | 2026-09-22 21:30:43 UTC | bounce-token-model-v1 | confirmed (submit_job result) |
| 2842 | 2026-09-22 22:38:52 UTC | bounce-token-model-v2 | confirmed (submit_job result) |
| 2843 | 2026-09-23 00:29:53 UTC | bounce-token-model-v3 HOME/bounce/scripts/polaris_train_token_v3.sh | confirmed (submit_job result) |
| 2844 | 2026-09-23 00:37:31 UTC | bounce-token-model-v4 HOME/bounce/scripts/polaris_train_token_v4.sh | confirmed (submit_job result) |
| 2845 | 2026-09-23 00:39:27 UTC | bounce-token-model-v5 HOME/bounce/scripts/polaris_train_token_v5.sh | confirmed (submit_job result) |
| 2846 | 2026-09-23 00:46:15 UTC | bounce-token-dataset-v1 HOME/bounce/scripts/polaris_generate_token_dataset_v1.sh | confirmed (submit_job result) |
| 2847 | 2026-09-23 00:53:04 UTC | bounce-token-model-v6 HOME/bounce/scripts/polaris_train_token_v6.sh | confirmed (submit_job result) |
| 2848 | 2026-09-23 01:08:22 UTC | bounce-token-model-v6 HOME/bounce/scripts/polaris_train_token_v6.sh | confirmed (submit_job result) |
| 2849 | 2026-09-23 01:15:24 UTC | bounce-token-model-v7 HOME/bounce/scripts/polaris_train_token_v7.sh | confirmed (submit_job result) |
| 2850 | 2026-09-23 03:37:05 UTC | bounce-token-model-v8 HOME/bounce/scripts/polaris_train_token_v8.sh | confirmed (submit_job result) |
| 2851 | 2026-09-23 03:49:11 UTC | bounce-token-model-v8 HOME/bounce/scripts/polaris_train_token_v8.sh | confirmed (submit_job result) |
| 2852 | 2026-09-23 04:52:55 UTC | bounce-token-model-v9 HOME/bounce/scripts/polaris_train_token_v9.sh | confirmed (submit_job result) |
| 2853 | 2026-09-23 07:25:55 UTC | bounce-token-model-v10 HOME/bounce/scripts/polaris_train_token_v10.sh | confirmed (submit_job result) |
| 2854 | 2026-09-23 07:30:27 UTC | bounce-token-model-v10 HOME/bounce/scripts/polaris_train_token_v10.sh | confirmed (submit_job result) |
| 2855 | 2026-09-23 07:35:05 UTC | bounce-token-model-v11 HOME/bounce/scripts/polaris_train_token_v11.sh | confirmed (submit_job result) |
| 2856 | 2026-09-23 08:31:46 UTC | bounce-token-model-v12 HOME/bounce/scripts/polaris_train_token_v12.sh | confirmed (submit_job result) |
| 2857 | 2026-09-23 19:18:57 UTC | bounce-token-model-v13 | confirmed (submit_job result) |
| 2858 | 2026-09-23 21:06:01 UTC | bounce-token-model-v14 | confirmed (submit_job result) |
| 2859 | 2026-09-23 21:57:26 UTC | bounce-token-model-v15 | confirmed (submit_job result) |
| 2861 | 2026-09-23 23:08:16 UTC | bounce-token-model-v16 bounce/scripts/polaris_train_token_v16.sh | confirmed (submit_job result) |
| 2862 | 2026-09-23 23:08:39 UTC | bounce-token-model-v16 | confirmed (submit_job result) |
| 2863 | 2026-09-23 23:19:47 UTC | bounce-token-model-v16 | confirmed (submit_job result) |
| 2868 | 2026-09-24 03:27:02 UTC | bounce-token-model-v17 | confirmed (submit_job result) |
| 2873 | 2026-09-24 03:40:02 UTC | bounce-token-model-v17 | confirmed (submit_job result) |
| 2874 | 2026-09-24 04:37:21 UTC | bounce-token-dataset-v2 scripts/generate_token_dataset.py | confirmed (submit_job result) |
| 2875 | 2026-09-24 04:44:58 UTC | bounce-token-model-v18 | confirmed (submit_job result) |
| 2876 | 2026-09-24 04:49:06 UTC | bounce-token-model-v18 | confirmed (submit_job result) |
| 2878 | 2026-09-24 09:05:03 UTC | bounce-token-model-v19 | confirmed (submit_job result) |
| 2879 | 2026-09-24 09:05:07 UTC | bounce-token-model-v20 | confirmed (submit_job result) |
| 2880 | 2026-09-24 09:51:01 UTC | bounce-token-model-v21 | confirmed (submit_job result) |
| 2881 | 2026-09-24 09:54:11 UTC | bounce-token-model-v22 | confirmed (submit_job result) |
| 2882 | 2026-09-24 09:54:15 UTC | bounce-token-model-v23 | confirmed (submit_job result) |
| 2884 | 2026-09-24 10:01:35 UTC | bounce-token-model-v24 | confirmed (submit_job result) |
| 2885 | 2026-09-24 10:32:25 UTC | bounce-token-model-v25 | confirmed (submit_job result) |
| 2886 | 2026-09-24 12:02:36 UTC | bounce-token-dataset-h44 scripts/generate_token_dataset.py | confirmed (submit_job result) |
| 2887 | 2026-09-24 12:03:48 UTC | bounce-token-model-v26 | confirmed (submit_job result) |
| 2888 | 2026-09-24 12:03:53 UTC | bounce-token-model-v27 | confirmed (submit_job result) |
| 2890 | 2026-09-24 12:46:05 UTC | bounce-token-model-v28 | confirmed (submit_job result) |
| 2891 | 2026-09-24 13:49:40 UTC | bounce-token-model-v29 | confirmed (submit_job result) |
| 2892 | 2026-09-24 13:49:44 UTC | bounce-token-model-v30 | confirmed (submit_job result) |
| 2893 | 2026-09-24 14:20:38 UTC | bounce-token-model-v28 | confirmed (submit_job result) |
| 2894 | 2026-09-24 14:29:45 UTC | bounce-token-model-v31 | confirmed (submit_job result) |
| 2895 | 2026-09-24 14:29:50 UTC | bounce-token-model-v32 | confirmed (submit_job result) |
| 2896 | 2026-09-24 14:44:05 UTC | bounce-token-model-v33 | confirmed (submit_job result) |
| 2897 | 2026-09-24 14:44:10 UTC | bounce-token-model-v34 | confirmed (submit_job result) |
| 2898 | 2026-09-24 14:58:31 UTC | bounce-token-model-v35 | confirmed (submit_job result) |
| 2899 | 2026-09-24 14:58:36 UTC | bounce-token-model-v36 | confirmed (submit_job result) |
| 2900 | 2026-09-24 15:03:48 UTC | bounce-token-model-v37 | confirmed (submit_job result) |
| 2901 | 2026-09-24 15:46:06 UTC | bounce-token-model-v38 | confirmed (submit_job result) |
| 2902 | 2026-09-24 15:46:10 UTC | bounce-token-model-v39 | confirmed (submit_job result) |
| 2903 | 2026-09-24 16:10:17 UTC | bounce-token-model-v40 | confirmed (submit_job result) |
| 2904 | 2026-09-24 16:10:22 UTC | bounce-token-model-v41 | confirmed (submit_job result) |
| 2905 | 2026-09-24 16:10:26 UTC | bounce-token-model-v42 | confirmed (submit_job result) |
| 2913 | 2026-09-24 18:19:29 UTC | bounce-bench-scaling | confirmed (submit_job result) |
| 2914 | 2026-09-24 18:20:38 UTC | bounce-bench-scaling | confirmed (submit_job result) |
| 2915 | 2026-09-24 18:29:38 UTC | bounce-bench-smoke | confirmed (submit_job result) |
| 2916 | 2026-09-24 18:32:42 UTC | bounce-bench-smoke | confirmed (submit_job result) |
| 2917 | 2026-09-24 18:34:11 UTC | bounce-bench-big | confirmed (submit_job result) |
| 2919 | 2026-09-24 19:57:35 UTC | gravity_sim.py | confirmed (sbatch output) |
| 2922 | 2026-09-24 20:30:22 UTC | scripts/polaris_tiled_video.sh | confirmed (sbatch output) |
| 2925 | 2026-09-24 21:03:40 UTC | scripts/polaris_tiled_video.sh | confirmed (sbatch output) |
| 2927 | 2026-09-24 21:20:15 UTC | scripts/polaris_train_gravity_v2.sh | confirmed (sbatch output) |
| 2928 | 2026-09-24 21:36:30 UTC | scripts/polaris_train_gravity_v2.sh | confirmed (sbatch output) |
| 2932 | 2026-09-24 23:44:24 UTC | scripts/polaris_gravity_long.sh | confirmed (sbatch output) |
| 2937 | 2026-09-25 00:29:14 UTC | scripts/polaris_train_gravity_central.sh | confirmed (sbatch output) |
| 2938 | 2026-09-25 00:34:16 UTC | scripts/polaris_train_gravity_central.sh | confirmed (sbatch output) |
| 2939 | 2026-09-25 00:42:04 UTC | scripts/polaris_gravity_long_central_far.sh | confirmed (sbatch output) |
| 2940 | 2026-09-25 00:59:32 UTC | scripts/polaris_gravity_1b.sh | confirmed (sbatch output) |
| 2941 | 2026-09-25 01:11:21 UTC | scripts/polaris_ball_1k.sh | confirmed (sbatch output) |
| 2942 | 2026-09-25 01:12:44 UTC | scripts/polaris_ball_1k.sh | confirmed (sbatch output) |
| 2943 | 2026-09-25 01:25:05 UTC | scripts/polaris_energy_probe.sh | confirmed (sbatch output) |
| 2944 | 2026-09-25 01:28:23 UTC | scripts/polaris_energy_probe2.sh | confirmed (sbatch output) |
| 2945 | 2026-09-25 01:30:48 UTC | scripts/polaris_energy_probe3.sh | confirmed (sbatch output) |
| 2946 | 2026-09-25 01:33:51 UTC | scripts/polaris_energy_finetune.sh | confirmed (sbatch output) |
| 2947 | 2026-09-25 01:48:09 UTC | scripts/polaris_energy_finetune2.sh | confirmed (sbatch output) |
| 2951 | 2026-09-25 03:02:23 UTC | scripts/polaris_fix1.sh | confirmed (sbatch output) |
| 2952 | 2026-09-25 03:03:07 UTC | scripts/polaris_fix1.sh | confirmed (sbatch output) |
| 2953 | 2026-09-25 03:04:10 UTC | scripts/polaris_train_conservative.sh | confirmed (sbatch output) |
| 2955 | 2026-09-25 03:43:27 UTC | scripts/polaris_eval_energy_fix.sh | confirmed (sbatch output) |
| 2956 | 2026-09-25 03:51:08 UTC | scripts/polaris_gdir_stage1.sh | confirmed (sbatch output) |
| 2957 | 2026-09-25 03:54:22 UTC | scripts/polaris_gdir_stage2.sh | confirmed (sbatch output) |
| 2960 | 2026-09-25 04:36:11 UTC | scripts/polaris_gdir_video.sh | confirmed (sbatch output) |
| 2965 | 2026-09-25 05:43:34 UTC | scripts/polaris_gravity_1b_clusters.sh | confirmed (sbatch output) |
| 2966 | 2026-09-25 05:44:43 UTC | scripts/polaris_gravity_1b_clusters.sh | confirmed (sbatch output) |
| 2967 | 2026-09-25 06:16:06 UTC | scripts/polaris_far_field_check.sh | confirmed (sbatch output) |
| 2968 | 2026-09-25 06:16:42 UTC | scripts/polaris_gravity_1b_far.sh | confirmed (sbatch output) |
| 2969 | 2026-09-25 06:17:13 UTC | scripts/polaris_gravity_1b_far.sh | confirmed (sbatch output) |
| 2970 | 2026-09-25 06:24:39 UTC | scripts/polaris_gravity_1m_far_10k.sh | confirmed (sbatch output) |
| 2971 | 2026-09-25 06:27:11 UTC | scripts/polaris_far_field_accuracy.sh | confirmed (sbatch output) |
| 2972 | 2026-09-25 06:55:23 UTC | scripts/polaris_gravity_collision.sh | confirmed (sbatch output) |
| 2973 | 2026-09-25 07:03:28 UTC | scripts/polaris_gravity_flyby.sh | confirmed (sbatch output) |
| 2975 | 2026-09-25 09:37:21 UTC | cd ~/bounce && sed -i 's/^#SBATCH --gres/#SBATCH --partition=gpu\n#SBA | confirmed (sbatch output) |
| 2976 | 2026-09-25 09:42:49 UTC | cd ~/bounce && sed -i 's#scripts/mesh_momentum_probe.py#scripts/mesh_m | confirmed (sbatch output) |
| 2977 | 2026-09-25 09:47:01 UTC | scripts/polaris_gravity_cic_reruns.sh | confirmed (sbatch output) |
| 2978 | 2026-09-25 09:51:46 UTC | scripts/polaris_lut_bench.sh | confirmed (sbatch output) |
| 2979 | 2026-09-25 10:11:18 UTC | scripts/polaris_flyby_100k_bench.sh | confirmed (sbatch output) |
| 2980 | 2026-09-25 10:13:24 UTC | scripts/polaris_flyby_100k_dt.sh | confirmed (sbatch output) |
| 2981 | 2026-09-25 10:14:49 UTC | scripts/polaris_flyby_100k.sh | confirmed (sbatch output) |
| 2987 | 2026-09-25 20:48:32 UTC | scripts/polaris_tile_box_test.sh | confirmed (sbatch output) |
| 2988 | 2026-09-25 20:52:18 UTC | scripts/polaris_shift_control.sh | confirmed (sbatch output) |
| 2989 | 2026-09-25 20:55:18 UTC | scripts/polaris_tile_vs_global.sh | confirmed (sbatch output) |
| 2990 | 2026-09-25 21:14:39 UTC | scripts/polaris_tile_speed_test.sh | confirmed (sbatch output) |
| 2991 | 2026-09-25 21:42:36 UTC | bounce-oracle-smoke | confirmed (submit_job result) |
| 2992 | 2026-09-25 21:43:22 UTC | scripts/polaris_adaptive_oracle.sh | confirmed (sbatch output) |
| 2996 | 2026-09-25 21:49:32 UTC | scripts/polaris_granularity_estimator.sh | confirmed (sbatch output) |
| 2999 | 2026-09-25 21:50:59 UTC | scripts/polaris_granularity_estimator_ablate.sh | confirmed (sbatch output) |
| 3000 | 2026-09-25 22:09:41 UTC | bounce-dual-smoke | confirmed (submit_job result) |
| 3001 | 2026-09-25 22:10:23 UTC | scripts/polaris_dual_tree.sh | confirmed (sbatch output) |
| 3002 | 2026-09-25 22:15:40 UTC | scripts/polaris_dual_estimator.sh | confirmed (sbatch output) |
| 3003 | 2026-09-25 22:19:18 UTC | scripts/polaris_dual_kernels.sh | confirmed (sbatch output) |
| 3004 | 2026-09-25 22:21:48 UTC | scripts/polaris_dual_kernels2.sh | confirmed (sbatch output) |
| 3005 | 2026-09-25 22:29:53 UTC | scripts/polaris_dual_kernels3.sh | confirmed (sbatch output) |
| 3006 | 2026-09-25 23:02:33 UTC | scripts/polaris_rollout_smoke.sh | confirmed (sbatch output) |
| 3007 | 2026-09-25 23:06:53 UTC | genic-exA | confirmed (submit_job result) |
| 3008 | 2026-09-25 23:07:11 UTC | rollout-smoke | confirmed (submit_job result) |
| 3009 | 2026-09-25 23:08:21 UTC | fallback-smoke scripts/fallback_run.py | confirmed (submit_job result) |
| 3010 | 2026-09-25 23:08:31 UTC | genic-exB | confirmed (submit_job result) |
| 3011 | 2026-09-25 23:08:33 UTC | genic-scal scripts/genic_scaling.py | confirmed (submit_job result) |
| 3012 | 2026-09-25 23:08:34 UTC | genic-rollA /bounce/scripts/genic_roll_job.sh | confirmed (submit_job result) |
| 3013 | 2026-09-25 23:08:35 UTC | genic-rollB /bounce/scripts/genic_roll_job.sh | confirmed (submit_job result) |
| 3014 | 2026-09-25 23:09:09 UTC | scripts/polaris_genker_a.sh | confirmed (sbatch output) |
| 3016 | 2026-09-25 23:24:07 UTC | sleep 115; tail -c 150 ~/bounce/interp_submit.out; squeue -u $USER / h | confirmed (sbatch output) |
| 3017 | 2026-09-25 23:28:49 UTC | tail -2 ~/bounce/tiling_submit.out; squeue -u adamwebb / awk '{print $ | confirmed (sbatch output) |
| 3018 | 2026-09-25 23:28:49 UTC | tail -2 ~/bounce/tiling_submit.out; squeue -u adamwebb / awk '{print $ | confirmed (sbatch output) |
| 3020 | 2026-09-25 23:45:55 UTC | sleep 100; tail -1 ~/fallback_submit.out; squeue -u $USER / grep -E "  | confirmed (sbatch output) |
| 3022 | 2026-09-26 00:18:29 UTC | genic-stat scripts/genic_static.py | confirmed (submit_job result) |
| 3023 | 2026-09-26 00:21:51 UTC | scripts/polaris_genker_all.sh | confirmed (sbatch output) |
| 3025 | 2026-09-26 00:33:55 UTC | polaris_interp_maps2.sh | confirmed (sbatch output) |
| 3026 | 2026-09-26 01:25:28 UTC | tail -5 ~/rollout_submit.log; squeue -u $USER / awk '{print $1,$3,$5,$ | confirmed (sbatch output) |
| 3028 | 2026-09-26 01:25:28 UTC | tail -5 ~/rollout_submit.log; squeue -u $USER / awk '{print $1,$3,$5,$ | confirmed (sbatch output) |
| 3029 | 2026-09-26 01:35:17 UTC | tail -3 ~/rollout_submit.log; squeue -u $USER / awk '{print $1,$3,$5,$ | confirmed (sbatch output) |
| 3030 | 2026-09-26 01:45:30 UTC | tail -3 ~/rollout_submit.log; squeue -u $USER / awk '{print $1,$3,$5,$ | confirmed (sbatch output) |
| 3032 | 2026-09-26 01:55:20 UTC | tail -3 ~/rollout_submit.log; squeue -u $USER / awk '{print $1,$3,$5,$ | confirmed (sbatch output) |
| 3033 | 2026-09-26 02:05:10 UTC | tail -3 ~/rollout_submit.log; squeue -u $USER / awk '{print $1,$3,$5,$ | confirmed (sbatch output) |
| 3034 | 2026-09-26 02:23:25 UTC | squeue -u $USER -n rollout-q; tail -3 ~/rollout_submit.log; ls ~/bounc | confirmed (sbatch output) |
| 3038 | 2026-09-26 02:33:19 UTC | squeue -u $USER -n rollout-q; tail -3 ~/rollout_submit.log; ls ~/bounc | confirmed (sbatch output) |
| 3041 | 2026-09-26 03:12:37 UTC | squeue -u $USER -n rollout-q; ls ~/bounce/results / grep "^rollout_\(f | confirmed (sbatch output) |
| 3044 | 2026-09-26 04:15:26 UTC | squeue -u $USER -n rollout-q; ls ~/bounce/results / grep "^rollout_uni | confirmed (sbatch output) |
| 3047 | 2026-09-26 04:25:12 UTC | squeue -u $USER -n rollout-q; ls ~/bounce/results / grep "^rollout_uni | confirmed (sbatch output) |
| 3050 | 2026-09-26 04:40:29 UTC | ps aux/grep scaling_submit/grep -v grep/wc -l; tail -3 ~/scaling_submi | confirmed (sbatch output) |
| 3051 | 2026-09-26 04:40:37 UTC | cd ~/bounce && cat > scripts/scaling_queue2.txt <<'EOF' scaling-c-uni- | confirmed (sbatch output) |
| 3054 | 2026-09-26 08:49:34 UTC | clumpy-1m-5k | confirmed (submit_job result) |
| 3055 | 2026-09-26 08:49:46 UTC | clumpy-1m-5k | confirmed (submit_job result) |
| 3056 | 2026-09-26 09:01:49 UTC | clumpy-1m-5k-mesh | confirmed (submit_job result) |
| 3057 | 2026-09-26 09:03:07 UTC | clumpy-1m-5k-resume | confirmed (submit_job result) |
| 3058 | 2026-09-26 09:30:26 UTC | clumpy-1m-5k-resume | confirmed (submit_job result) |
| 3059 | 2026-09-26 09:59:40 UTC | orbit-smoke | confirmed (submit_job result) |
| 3060 | 2026-09-26 09:59:41 UTC | binary-mesh | confirmed (submit_job result) |
| 3061 | 2026-09-26 09:59:43 UTC | bh-capped | confirmed (submit_job result) |
| 3062 | 2026-09-26 09:59:45 UTC | exact-and-control | confirmed (submit_job result) |
| 3063 | 2026-09-26 11:01:24 UTC | binary-v2-mesh | confirmed (submit_job result) |
| 3064 | 2026-09-26 11:01:26 UTC | v2-exact-bhctrl | confirmed (submit_job result) |
| 3065 | 2026-09-26 11:02:32 UTC | cmp-clumpy | confirmed (submit_job result) |
| 3066 | 2026-09-26 12:01:03 UTC | binary-dt-test | confirmed (submit_job result) |
| 3067 | 2026-09-26 14:00:54 UTC | binary-d1200 | confirmed (submit_job result) |
| 3068 | 2026-09-26 14:00:55 UTC | binary-d900 | confirmed (submit_job result) |
| 3069 | 2026-09-26 15:01:12 UTC | binary-d1200-box | confirmed (submit_job result) |
| 3072 | 2026-09-26 18:16:24 UTC | orbit3d-smoke | confirmed (submit_job result) |
| 3073 | 2026-09-26 18:17:13 UTC | orbit3d-binary | confirmed (submit_job result) |
| 3074 | 2026-09-26 18:27:20 UTC | bh-relativistic | confirmed (submit_job result) |
| 3075 | 2026-09-26 20:08:23 UTC | bh-learned-smoke | confirmed (submit_job result) |
| 3076 | 2026-09-26 20:10:16 UTC | bench-learned-force | confirmed (submit_job result) |
| 3078 | 2026-09-26 20:12:41 UTC | bench-learned-tab | confirmed (submit_job result) |
| 3079 | 2026-09-26 20:14:11 UTC | bench-learned-tab | confirmed (submit_job result) |
| 3080 | 2026-09-26 20:15:38 UTC | bh-learned | confirmed (submit_job result) |
| 3086 | 2026-09-26 22:39:16 UTC |  | confirmed (submit_job result) |
| 3087 | 2026-09-26 22:39:58 UTC | bh-learned-r25 | confirmed (submit_job result) |
| 3088 | 2026-09-26 22:41:47 UTC | cancel-3087 | confirmed (submit_job result) |
| 3089 | 2026-09-26 23:21:30 UTC | bh-learned-r5 | confirmed (submit_job result) |
| 3090 | 2026-09-27 03:21:41 UTC |  | confirmed (submit_job result) |
| 3091 | 2026-09-27 03:28:13 UTC |  | confirmed (submit_job result) |
| 3092 | 2026-09-27 03:32:21 UTC |  | confirmed (submit_job result) |
| 3093 | 2026-09-27 03:36:54 UTC |  | confirmed (submit_job result) |
| 3094 | 2026-09-27 06:43:21 UTC |  | confirmed (submit_job result) |
| 3095 | 2026-09-27 06:46:45 UTC |  | confirmed (submit_job result) |
| 3096 | 2026-09-27 06:47:51 UTC |  | confirmed (submit_job result) |
| 3097 | 2026-09-27 07:19:21 UTC |  | confirmed (submit_job result) |
| 3098 | 2026-09-27 07:20:49 UTC |  | confirmed (submit_job result) |
| 3105 | 2026-09-27 18:09:58 UTC | sacct-check | confirmed (submit_job result) |
| 3106 | 2026-09-27 20:58:45 UTC | sacct_query | confirmed (submit_job result) |
| 3107 | 2026-09-27 20:58:55 UTC | sacct_query2 | confirmed (submit_job result) |
| 3108 | 2026-09-27 20:59:13 UTC | sacct_sum | confirmed (submit_job result) |
| 3111 | 2026-09-28 00:25:56 UTC |  | confirmed (submit_job result) |
| 3112 | 2026-09-28 00:30:05 UTC |  | confirmed (submit_job result) |
| 3113 | 2026-09-28 00:31:21 UTC |  | confirmed (submit_job result) |
| 3114 | 2026-09-28 01:25:53 UTC | bounce-cons100 scripts/train_conservative.py | confirmed (submit_job result) |
| 3116 | 2026-09-28 02:01:52 UTC | bounce-50k | confirmed (submit_job result) |
| 3117 | 2026-09-28 02:25:15 UTC | bounce-ball-50k-rollout | confirmed (submit_job result) |
| 3118 | 2026-09-28 03:28:42 UTC | bounce-orbit-bh-100k-250steps | confirmed (submit_job result) |
| 3119 | 2026-09-28 06:02:58 UTC |  | confirmed (submit_job result) |
| 3120 | 2026-09-28 07:23:42 UTC | bounce-gravity-relativistic-export | confirmed (submit_job result) |
| 3122 | 2026-09-28 17:54:49 UTC |  | confirmed (submit_job result) |
| 3124 | 2026-09-28 18:45:26 UTC | bounce-probe-bh-ic | confirmed (submit_job result) |
| 3125 | 2026-09-28 18:46:44 UTC | scripts/polaris_orbit_bh_100k_collapse_coarse.sh | confirmed (submit_job result) |
| 3126 | 2026-09-28 18:57:05 UTC | scripts/polaris_orbit_bh_100k_collapse_slowmo.sh | confirmed (submit_job result) |
| 3127 | 2026-09-28 19:43:02 UTC |  | confirmed (submit_job result) |
| 3128 | 2026-09-28 19:44:47 UTC | scancel-3127 | confirmed (submit_job result) |
| 3129 | 2026-09-28 19:44:58 UTC |  | confirmed (submit_job result) |
| 3131 | 2026-09-28 20:05:00 UTC |  | confirmed (submit_job result) |
| 3132 | 2026-09-29 00:18:00 UTC | bounce-orbit-bh-100k-first20-8substeps | confirmed (submit_job result) |
| 3133 | 2026-09-29 02:01:19 UTC |  | confirmed (submit_job result) |
| 3134 | 2026-09-29 02:44:39 UTC | bounce-contact-gen-eval-v1 | confirmed (submit_job result) |
| 3135 | 2026-09-29 17:01:02 UTC |  | confirmed (submit_job result) |
| 3136 | 2026-09-29 17:33:49 UTC |  | confirmed (submit_job result) |
| 3138 | 2026-09-29 21:22:25 UTC |  | confirmed (submit_job result) |
| 3139 | 2026-09-29 22:14:30 UTC |  | confirmed (submit_job result) |
| 3142 | 2026-09-29 22:54:38 UTC | bounce-contact-gen-symlog-capped | confirmed (submit_job result) |
| 3143 | 2026-09-29 23:28:09 UTC | bounce-contact-eval-symlog-capped | confirmed (submit_job result) |
| 3144 | 2026-09-29 23:31:37 UTC | bounce-contact-gen-symlog-densershape | confirmed (submit_job result) |
| 3145 | 2026-09-30 00:05:20 UTC | bounce-contact-eval-symlog-densershape | confirmed (submit_job result) |
| 3146 | 2026-09-30 00:43:24 UTC | bounce-orbit-bh-100k-500frames | confirmed (submit_job result) |
| 3150 | 2026-09-30 04:24:49 UTC |  | confirmed (submit_job result) |
| 3151 | 2026-09-30 04:43:13 UTC |  | confirmed (submit_job result) |
| 3177 | 2026-09-30 23:50:47 UTC |  | confirmed (submit_job result) |
| 3178 | 2026-10-01 00:24:22 UTC |  | confirmed (submit_job result) |
| 3192 | 2026-10-02 01:16:48 UTC | bounce-causal-field-smoke | confirmed (submit_job result) |
| 3193 | 2026-10-02 01:17:30 UTC | bounce-causal-field-smoke2 | confirmed (submit_job result) |
| 3194 | 2026-10-02 01:18:12 UTC | sf-smoke scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3195 | 2026-10-02 01:18:40 UTC | bounce-causal-field-2body | confirmed (submit_job result) |
| 3196 | 2026-10-02 01:18:47 UTC | sf-smoke scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3197 | 2026-10-02 01:23:36 UTC | sf-A scripts/polaris_scatter_field.sh | confirmed (submit_job result) |
| 3198 | 2026-10-02 01:23:36 UTC | sf-C scripts/polaris_scatter_field.sh | confirmed (submit_job result) |
| 3199 | 2026-10-02 02:11:00 UTC | sf-smoke2 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3200 | 2026-10-02 02:31:18 UTC | sf-D scripts/polaris_scatter_field.sh | confirmed (submit_job result) |
| 3201 | 2026-10-02 02:31:19 UTC | sf-B scripts/polaris_scatter_field.sh | confirmed (submit_job result) |
| 3202 | 2026-10-02 03:17:40 UTC | sf-DA-fix scripts/polaris_scatter_field.sh | confirmed (submit_job result) |
| 3208 | 2026-10-02 05:04:27 UTC | sf-D-fix2 scripts/polaris_scatter_field.sh | confirmed (submit_job result) |
| 3209 | 2026-10-02 05:06:40 UTC | sf-nodm scripts/polaris_scatter_field.sh | confirmed (submit_job result) |
| 3211 | 2026-10-02 07:21:49 UTC | sf-iter1-pot scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3212 | 2026-10-02 07:30:59 UTC | sf-iter2 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3213 | 2026-10-02 07:50:55 UTC | sf-iter3-ker scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3214 | 2026-10-02 07:51:36 UTC | sf-iter3-ker scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3215 | 2026-10-02 08:07:37 UTC | sf-probe1 | confirmed (submit_job result) |
| 3216 | 2026-10-02 08:08:09 UTC | sf-probe1 | confirmed (submit_job result) |
| 3217 | 2026-10-02 08:08:29 UTC | sf-probe1 | confirmed (submit_job result) |
| 3218 | 2026-10-02 08:08:57 UTC | sf-probe1 | confirmed (submit_job result) |
| 3219 | 2026-10-02 08:09:46 UTC | sf-iter4-pp scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3220 | 2026-10-02 08:10:23 UTC | chk | confirmed (submit_job result) |
| 3221 | 2026-10-02 08:10:31 UTC | sf-iter4-pp scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3222 | 2026-10-02 08:27:29 UTC | sf-iter5-kp scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3223 | 2026-10-02 08:37:27 UTC | sf-iter6 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3225 | 2026-10-02 08:50:12 UTC | sf-iter7 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3226 | 2026-10-02 09:18:57 UTC | sf-iter8-fw scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3227 | 2026-10-02 09:25:36 UTC | sf-iter9-30k scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3228 | 2026-10-02 10:09:07 UTC | sf-iter10 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3229 | 2026-10-02 11:29:15 UTC | cancel3228 | confirmed (submit_job result) |
| 3230 | 2026-10-02 11:29:26 UTC | sf-iter11 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3231 | 2026-10-02 11:30:11 UTC | scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3232 | 2026-10-02 12:10:34 UTC | cancel-nan | confirmed (submit_job result) |
| 3233 | 2026-10-02 12:10:51 UTC | sf-iter11b scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3234 | 2026-10-02 12:10:57 UTC | scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3235 | 2026-10-02 16:15:08 UTC | scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3236 | 2026-10-02 16:28:04 UTC | scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3238 | 2026-10-02 18:27:53 UTC | cancel3236 | confirmed (submit_job result) |
| 3239 | 2026-10-02 18:28:43 UTC | sf-budget1 scripts/train_gravity_dynamics.py | confirmed (submit_job result) |
| 3244 | 2026-10-02 20:35:31 UTC | sf-profile | confirmed (submit_job result) |
| 3245 | 2026-10-02 20:35:51 UTC | sf-budget2 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3246 | 2026-10-02 20:52:21 UTC | sf-profile2 | confirmed (submit_job result) |
| 3247 | 2026-10-02 20:53:24 UTC | sf-graphtest | confirmed (submit_job result) |
| 3248 | 2026-10-02 20:53:47 UTC | sf-graphtest2 | confirmed (submit_job result) |
| 3249 | 2026-10-02 20:54:17 UTC | sf-graphtest3 | confirmed (submit_job result) |
| 3250 | 2026-10-02 20:54:49 UTC | sf-budget3 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3251 | 2026-10-02 20:55:29 UTC | sf-intfloor | confirmed (submit_job result) |
| 3252 | 2026-10-02 20:56:24 UTC | sf-budget4-verlet scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3255 | 2026-10-02 22:15:53 UTC | sf-probe-v | confirmed (submit_job result) |
| 3257 | 2026-10-02 22:16:24 UTC | sf-budgetB scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3260 | 2026-10-03 01:14:31 UTC | gap-floors-e2 | confirmed (submit_job result) |
| 3261 | 2026-10-03 01:15:23 UTC | gap-floors-e2 | confirmed (submit_job result) |
| 3262 | 2026-10-03 01:20:01 UTC | gap-ss-e2 | confirmed (submit_job result) |
| 3263 | 2026-10-03 01:20:05 UTC | gap-e1-base | confirmed (submit_job result) |
| 3265 | 2026-10-03 01:25:03 UTC | cancel3263 | confirmed (submit_job result) |
| 3266 | 2026-10-03 01:25:13 UTC | gap-pp-e2 | confirmed (submit_job result) |
| 3267 | 2026-10-03 01:25:14 UTC | gap-e1-base | confirmed (submit_job result) |
| 3268 | 2026-10-03 01:27:56 UTC | gap-e1-fix | confirmed (submit_job result) |
| 3272 | 2026-10-03 06:10:27 UTC |  | confirmed (submit_job result) |
| 3273 | 2026-10-03 06:12:10 UTC |  | confirmed (submit_job result) |
| 3274 | 2026-10-03 08:21:11 UTC | tail-log | confirmed (submit_job result) |
| 3275 | 2026-10-03 08:33:24 UTC | tail-log | confirmed (submit_job result) |
| 3276 | 2026-10-03 08:41:03 UTC | tail-log | confirmed (submit_job result) |
| 3277 | 2026-10-03 08:41:15 UTC |  | confirmed (submit_job result) |
| 3278 | 2026-10-03 08:43:45 UTC | tail-log | confirmed (submit_job result) |
| 3288 | 2026-10-03 23:43:46 UTC | scatter-node-probe | confirmed (submit_job result) |
| 3289 | 2026-10-03 23:43:55 UTC | scatter-node-install | confirmed (submit_job result) |
| 3290 | 2026-10-03 23:46:46 UTC | scatter-precompute | confirmed (submit_job result) |
| 3296 | 2026-10-04 20:38:38 UTC |  | confirmed (submit_job result) |
| 3297 | 2026-10-04 20:38:41 UTC | bounce-chaos-floor | confirmed (submit_job result) |
| 3298 | 2026-10-04 20:39:30 UTC | scatter-regress-rerun | confirmed (submit_job result) |
| 3299 | 2026-10-04 20:43:43 UTC | opt-setup | confirmed (submit_job result) |
| 3300 | 2026-10-04 20:44:24 UTC | opt-base | confirmed (submit_job result) |
| 3301 | 2026-10-04 20:45:52 UTC | opt-proftrain | confirmed (submit_job result) |
| 3302 | 2026-10-04 20:47:21 UTC | scal-setup | confirmed (submit_job result) |
| 3303 | 2026-10-04 20:47:23 UTC | opt-r1 | confirmed (submit_job result) |
| 3304 | 2026-10-04 20:48:09 UTC | scal-smoke HOME/bounce_scaling/scripts/polaris_scatter_scaling.sh | confirmed (submit_job result) |
| 3305 | 2026-10-04 20:49:13 UTC | scal-main HOME/bounce_scaling/scripts/polaris_scatter_scaling.sh | confirmed (submit_job result) |
| 3306 | 2026-10-04 20:49:18 UTC | opt-r2 | confirmed (submit_job result) |
| 3307 | 2026-10-04 20:50:00 UTC | scal-cancel | confirmed (submit_job result) |
| 3308 | 2026-10-04 20:50:07 UTC | scal-main2 HOME/bounce_scaling/scripts/polaris_scatter_scaling.sh | confirmed (submit_job result) |
| 3309 | 2026-10-04 20:53:19 UTC | opt-r3 scripts/train_scatter_field.py | confirmed (submit_job result) |
| 3311 | 2026-10-04 21:31:29 UTC | scal-dump HOME/bounce_scaling/scripts/polaris_scatter_scaling.sh | confirmed (submit_job result) |
| 3312 | 2026-10-04 21:33:38 UTC | scal-setup2 | confirmed (submit_job result) |
| 3313 | 2026-10-04 21:35:03 UTC | scal-fprobe HOME/bounce_scaling/scripts/polaris_scatter_scaling.sh | confirmed (submit_job result) |
| 3314 | 2026-10-04 21:35:04 UTC | scal-cell HOME/bounce_scaling_opt/scripts/polaris_scatter_scaling_opt.sh | confirmed (submit_job result) |
| 3315 | 2026-10-04 21:51:59 UTC | scenes-setup | confirmed (submit_job result) |
| 3316 | 2026-10-04 21:52:52 UTC | cons-setup | confirmed (submit_job result) |
| 3317 | 2026-10-04 21:53:10 UTC |  | confirmed (submit_job result) |
| 3318 | 2026-10-04 21:53:44 UTC | cons-smoke HOME/bounce-cons/scripts/polaris_scatter_conservation.sh | confirmed (submit_job result) |
| 3319 | 2026-10-04 21:56:58 UTC | bounce-scenes-zs | confirmed (submit_job result) |
| 3320 | 2026-10-04 21:57:28 UTC | scal-cellL HOME/bounce_scaling_opt/scripts/polaris_scatter_scaling_opt.sh | confirmed (submit_job result) |
| 3321 | 2026-10-04 21:57:28 UTC | scal-chunkL HOME/bounce_scaling/scripts/polaris_scatter_scaling.sh | confirmed (submit_job result) |
| 3323 | 2026-10-04 22:35:57 UTC | cons-full HOME/bounce-cons/scripts/polaris_scatter_conservation.sh | confirmed (submit_job result) |
| 3324 | 2026-10-04 22:36:19 UTC |  | confirmed (submit_job result) |
| 3325 | 2026-10-04 22:39:54 UTC | bounce-scenes-ft | confirmed (submit_job result) |
| 3326 | 2026-10-04 22:41:11 UTC | opt-r4 | confirmed (submit_job result) |
| 3327 | 2026-10-04 22:50:05 UTC | scal-cancel2 | confirmed (submit_job result) |
| 3328 | 2026-10-05 00:24:30 UTC | bounce-scenes-ft2 | confirmed (submit_job result) |
| 3333 | 2026-10-06 17:17:17 UTC | scatter-bh-1body | confirmed (submit_job result) |
| 3334 | 2026-10-06 17:18:04 UTC | scatter-bh-1moving | confirmed (submit_job result) |
| 3335 | 2026-10-06 17:26:32 UTC | scatter-bh-threshold | confirmed (submit_job result) |
| 3336 | 2026-10-06 17:31:44 UTC | scatter-bh-sweep | confirmed (submit_job result) |
| 3337 | 2026-10-06 17:46:23 UTC | delete-star-E | confirmed (submit_job result) |
| 3338 | 2026-10-06 17:48:29 UTC | delete-star-field | confirmed (submit_job result) |
| 3339 | 2026-10-06 17:49:17 UTC | delete-star-field2 | confirmed (submit_job result) |
| 3340 | 2026-10-06 20:43:55 UTC | augur-rename-move | confirmed (submit_job result) |
| 3341 | 2026-10-06 20:48:59 UTC | augur-code-sync scripts/polaris_train_token_v17.sh | confirmed (submit_job result) |
| 3345 | 2026-10-07 00:41:20 UTC | augur-fix-setup | confirmed (submit_job result) |
| 3346 | 2026-10-07 00:41:44 UTC | d3-setup | confirmed (submit_job result) |
| 3347 | 2026-10-07 00:42:45 UTC | scatter-energy-log | confirmed (submit_job result) |
| 3348 | 2026-10-07 00:43:02 UTC | augur-fix-baseline-v2 HOME/augur-fix/scripts/polaris_scatter_fix.sh | confirmed (submit_job result) |
| 3349 | 2026-10-07 00:43:18 UTC | pp8-smoke HOME/augur_pp8_up/polaris_scatter_pp8.sh | confirmed (submit_job result) |
| 3350 | 2026-10-07 00:43:20 UTC | find-node | confirmed (submit_job result) |
| 3351 | 2026-10-07 00:43:28 UTC | scatter-energy-log | confirmed (submit_job result) |
| 3352 | 2026-10-07 00:43:36 UTC |  | confirmed (submit_job result) |
| 3353 | 2026-10-07 00:43:45 UTC | d3-probe-env | confirmed (submit_job result) |
| 3354 | 2026-10-07 00:44:01 UTC | augur-fix2-far-mono HOME/augur-fix/scripts/polaris_scatter_fix.sh | confirmed (submit_job result) |
| 3355 | 2026-10-07 00:44:02 UTC | augur-fix2-far-quad HOME/augur-fix/scripts/polaris_scatter_fix.sh | confirmed (submit_job result) |
| 3356 | 2026-10-07 00:44:29 UTC | augur-fix4-sub2 HOME/augur-fix/scripts/polaris_scatter_fix.sh | confirmed (submit_job result) |
| 3357 | 2026-10-07 00:44:30 UTC | augur-fix3-smoke HOME/augur-fix/scripts/polaris_scatter_longpen.sh | confirmed (submit_job result) |
| 3358 | 2026-10-07 00:45:06 UTC |  | confirmed (submit_job result) |
| 3359 | 2026-10-07 00:46:54 UTC | augur-scatter-contact | confirmed (submit_job result) |
| 3360 | 2026-10-07 00:47:16 UTC | s46-seed | confirmed (submit_job result) |
| 3361 | 2026-10-07 00:50:27 UTC | s46-smoke scripts/train_scatter_ft.py | confirmed (submit_job result) |
| 3362 | 2026-10-07 00:54:43 UTC | augur-fix-cmp | confirmed (submit_job result) |
| 3363 | 2026-10-07 00:55:17 UTC | augur-fix2-far-b05 HOME/augur-fix/scripts/polaris_scatter_fix.sh | confirmed (submit_job result) |
| 3364 | 2026-10-07 00:56:55 UTC | augur-d3-cellgraph | confirmed (submit_job result) |
| 3365 | 2026-10-07 01:01:14 UTC | augur-fix-cmp | confirmed (submit_job result) |
| 3366 | 2026-10-07 01:05:28 UTC | scatter-band-probe | confirmed (submit_job result) |
| 3367 | 2026-10-07 01:11:43 UTC | augur-fix-cmp | confirmed (submit_job result) |
| 3368 | 2026-10-07 01:12:24 UTC | augur-ft-ctrl HOME/augur-fix/scripts/polaris_scatter_longpen.sh | confirmed (submit_job result) |
| 3369 | 2026-10-07 01:12:26 UTC | augur-ft-longpen HOME/augur-fix/scripts/polaris_scatter_longpen.sh | confirmed (submit_job result) |
| 3370 | 2026-10-07 01:12:27 UTC | augur-ft-sub2 HOME/augur-fix/scripts/polaris_scatter_longpen.sh | confirmed (submit_job result) |
| 3371 | 2026-10-07 01:15:29 UTC |  | confirmed (submit_job result) |
| 3372 | 2026-10-07 01:22:01 UTC | augur-scatter-contact | confirmed (submit_job result) |
| 3373 | 2026-10-07 01:26:07 UTC |  | confirmed (submit_job result) |
| 3374 | 2026-10-07 01:32:00 UTC | augur-fix-cmp | confirmed (submit_job result) |
| 3375 | 2026-10-07 01:32:23 UTC |  | confirmed (submit_job result) |
| 3376 | 2026-10-07 01:32:23 UTC |  | confirmed (submit_job result) |
| 3377 | 2026-10-07 02:08:16 UTC | augur-fix-cmp | confirmed (submit_job result) |
| 3378 | 2026-10-07 02:08:26 UTC |  | confirmed (submit_job result) |
| 3379 | 2026-10-07 02:50:44 UTC | augur-fix-cmp | confirmed (submit_job result) |
| 3380 | 2026-10-07 03:35:19 UTC | augur-scatter-contact | confirmed (submit_job result) |
| 3382 | 2026-10-07 04:06:05 UTC | augur-scatter-contact | confirmed (submit_job result) |
| 3397 | 2026-10-07 08:07:29 UTC | augur-scatter-pp8 HOME/augur_pp8_up/polaris_scatter_pp8.sh | confirmed (submit_job result) |
| 3398 | 2026-10-07 08:07:49 UTC |  | confirmed (submit_job result) |
| 3399 | 2026-10-07 08:09:48 UTC |  | confirmed (submit_job result) |
| 3400 | 2026-10-07 08:10:06 UTC |  | confirmed (submit_job result) |
| 3401 | 2026-10-07 09:56:06 UTC | augur-pp8-regress | confirmed (submit_job result) |
| 3406 | 2026-10-08 00:09:17 UTC | int-setup | confirmed (submit_job result) |
| 3407 | 2026-10-08 00:09:49 UTC | augur-scatter-int-pilot HOME/augur_int/scripts/polaris_scatter_int.sh | confirmed (submit_job result) |
| 3408 | 2026-10-08 00:10:11 UTC |  | confirmed (submit_job result) |
| 3409 | 2026-10-08 00:10:59 UTC | augur-scatter-int-pilot HOME/augur_int/scripts/polaris_scatter_int.sh | confirmed (submit_job result) |
| 3410 | 2026-10-08 00:12:42 UTC | augur-scatter-int-pilot HOME/augur_int/scripts/polaris_scatter_int.sh | confirmed (submit_job result) |
| 3411 | 2026-10-08 00:26:15 UTC | int-grep | confirmed (submit_job result) |
| 3412 | 2026-10-08 00:27:18 UTC | augur-scatter-int-pilot2 HOME/augur_int/scripts/polaris_scatter_int.sh | confirmed (submit_job result) |
| 3413 | 2026-10-08 00:39:35 UTC | augur-scatter-int-full HOME/augur_int/scripts/polaris_scatter_int.sh | confirmed (submit_job result) |
| 3414 | 2026-10-08 02:25:17 UTC | int-grep | confirmed (submit_job result) |
| 3415 | 2026-10-08 03:11:59 UTC | int-grep | confirmed (submit_job result) |
| 3416 | 2026-10-08 03:24:39 UTC | augur-int-ctrlcons | confirmed (submit_job result) |
| 3417 | 2026-10-08 08:21:30 UTC | joint-setup | confirmed (submit_job result) |
| 3418 | 2026-10-08 08:22:11 UTC | joint-sbatch scripts/polaris_scatter_joint.sh | confirmed (submit_job result) |
| 3419 | 2026-10-08 08:22:21 UTC |  | confirmed (sbatch output) |
| 3420 | 2026-10-08 08:36:35 UTC | joint-sbatch scripts/polaris_scatter_joint.sh | confirmed (submit_job result) |
| 3421 | 2026-10-08 08:36:38 UTC |  | confirmed (sbatch output) |
| 3423 | 2026-10-08 12:39:42 UTC | augur-joint-snaps | confirmed (submit_job result) |

## Text-only mentions inside the range (not confirmed submissions; first mention time)

3019 (2026-09-25 23:53:57 UTC), 3021 (2026-09-26 00:06:18 UTC), 3039 (2026-09-26 02:32:24 UTC), 3040 (2026-09-26 02:32:31 UTC), 3042 (2026-09-26 03:16:52 UTC), 3043 (2026-09-26 03:16:52 UTC), 3045 (2026-09-26 04:20:56 UTC), 3279 (2026-10-03 22:06:31 UTC)

## TIDE submissions (excluded)

none

## IDs in range with no transcript record (other users' jobs on the shared cluster or untracked)

209 IDs; first/last: [2807, 2820, 2830]...[3404, 3405, 3422]
