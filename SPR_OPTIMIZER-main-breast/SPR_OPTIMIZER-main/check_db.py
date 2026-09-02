import optuna

study = optuna.load_study(
    study_name="spr_opt",
    storage="sqlite:///results/spr_opt.db"
)

print("Trials:", len(study.trials))
