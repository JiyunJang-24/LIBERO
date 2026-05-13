import os
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.bddl_utils as BDDLUtils


# xyg add get libero path function
def get_libero_path(query_key):
    if query_key == "bddl_files":
        return "libero/libero/bddl_files"
    else:
        raise ValueError(f"Invalid query key: {query_key}")


def main():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, libero_10, etc.
    task_suite = benchmark_dict[task_suite_name]()      # task_suite.get_num_tasks()

    # retrieve a specific task
    # task_id = 0
    task_problem_info_list = []
    for task_id in range(task_suite.get_num_tasks()):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
            f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

        # task_problem_info = BDDLUtils.get_problem_info(task_bddl_file)
        task_problem_info = BDDLUtils.robosuite_parse_problem(task_bddl_file)
        task_problem_info_list.append(task_problem_info)
        # step over the environment
        # BDDLUtils.get_problem_info(bddl_file_name)
        if False:
            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)
            env.reset()
            init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
            init_state_id = 0
            env.set_init_state(init_states[init_state_id])

            dummy_action = [0.] * 7
            obs_list = []       # observation存储的东西特别之多，各种物体的pos, vel, quat, img全部都存储了
            for step in range(10):
                obs, reward, done, info = env.step(dummy_action)
                obs_list.append(obs)
            env.close()

    import ipdb; ipdb.set_trace()
    print('this is a test')
    # ['problem_name', 'fixtures', 'regions', 'objects', 'scene_properties', 'initial_state', 'goal_state', 'language_instruction', 'obj_of_interest']
    # problem_name
    # language_instruction
    # fixtures, objects, regions
    # goal_state
    for i in range(len(task_problem_info_list)): print(task_problem_info_list[i]['regions'], '\n', '\n')


if __name__ == "__main__":
    main()
    
