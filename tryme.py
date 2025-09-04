import mlflow
run = mlflow.active_run()
print(run)
print('by')
print(run.info.run_id)