# Example A: test a single jsonl as one multi-dispatch scenario
# TEST_CFG = {
#     "target_path": "test_case/case_04_train_stream_000.jsonl",
#     "model_path": "ddqn_policy_rainbow.pt",
#     "case_mode": "full_stream",
# }

# Example B: test a single jsonl line-by-line (each line as one single-dispatch case)
# TEST_CFG = {
#     "target_path": "test_case/case_04_train_stream_000.jsonl",
#     "model_path": "ddqn_policy_rainbow.pt",
#     "case_mode": "each_line",
# }

# Example C: test all jsonl in test_case folder
TEST_CFG = {
    "target_path": "test_case",
    "model_path": "ddqn_policy_rainbow.pt",
    "case_mode": "each_line",      # "full_stream" | "each_line"
    "max_files": None,                 # e.g. 3
    "max_cases_per_file": None,        # only used when each_line
    "plot": True,
    "save_plots": True,
    "plot_dir": "test_plots",
    "show_route_map": False,
    "show_plotly": False,            # True => generate HTML via viz_plotly.py
    "plotly_html_dir": "test_plots_html",
    "plotly_window": 80.0,
    "plotly_step": 5.0,
    "save_route_jsonl": True,       # True => export per-second AMR route log jsonl
    "route_jsonl_dir": "test_route_logs",
    "route_time_step": 1.0,
    "save_stats_txt": True,         # True => export case + summary stats to txt
    "stats_txt_path": "test_output/test_stats.txt",
    "stats_txt_append": True,
    "allow_proactive_replenish": True,
    "proactive_replenish_bias_weight": 1.5,
    "proactive_full_load_bias_weight": 0.8,
    "proactive_waiting_replenish_bias_weight": 1.2,
    "enable_collision_avoidance": False,
    "print_predict_time": True,        # True => print elapsed time for every prediction step
}

results = run_tests(**TEST_CFG)
print(f"result_rows = {len(results)}")
