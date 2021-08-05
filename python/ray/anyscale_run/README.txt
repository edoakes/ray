Environment is described in runtime_env.yaml.

Supported commands:
- anyscale run driver.py # Run the command on my local machine in the local env. Env will be cached across runs.
- anyscale shell # Open a shell running in the local env. This could be used to e.g., open IPython and do some dev.
- anyscale exec "python driver.py" # Run an arbitrary shell command in the local env. This could be how we implement job submission.

This is primarily concerned with the environment for the driver.
By default this would be inherited by driver/job, but you can still specify the runtime_env within the ray.init() call or per-actor/per-task (see driver-with-deps.py).
A use case for this would be calling Ray client from within another application (e.g., a hosted notebook).
